"""YOLOv8 para objetos e áreas críticas da sessão.

Três verificações, todas a partir das detecções:

1. **Objetos em cena** (YOLOv8n, COCO-80, 1 quadro por segundo): o que há no
   ambiente — cadeira, mesa, cama — para o relatório e para checar se o
   material do exercício está presente.
2. **Outra pessoa na área do paciente**: a "área crítica" é a região que o
   paciente ocupa durante a sessão (envoltória das caixas dele, com margem).
   Outra pessoa entrando ali é interferência ou assistência — o relatório
   registra, e é avaliado contra o rótulo `extra_person_in_cam` do REHAB24-6.
3. **Queda** (pela pose): o quadril desce até perto da altura dos tornozelos e
   a caixa do paciente fica mais larga que alta por ≥ 1 s. O REHAB24-6 não tem
   quedas — o detector é testado com séries sintéticas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import ROOT, VideoSettings
from .pose import SeriePose

MODELOS = ROOT / "models"
NOMES_PT = {"person": "pessoa", "chair": "cadeira", "dining table": "mesa", "bed": "cama", "couch": "sofá",
            "bottle": "garrafa", "cell phone": "celular", "laptop": "notebook", "tv": "monitor/TV",
            "sports ball": "bola", "bench": "banco", "backpack": "mochila", "handbag": "bolsa", "clock": "relógio",
            "potted plant": "planta", "book": "livro", "cup": "copo", "remote": "controle"}


@dataclass(frozen=True)
class Evento:
    tipo: str            # "pessoa_na_area" | "queda"
    inicio_s: float
    fim_s: float
    detalhe: str

    def as_dict(self) -> dict:
        return asdict(self)


def modelo_objetos(nome: str | None = None):
    from ultralytics import YOLO
    import torch

    torch.backends.nnpack.set_flags(False)   # sem suporte nesta CPU: evita um aviso por inferência

    MODELOS.mkdir(parents=True, exist_ok=True)
    return YOLO(str(MODELOS / (nome or VideoSettings().yolo_model)))


def objetos_em_cena(video: Path, a_cada_s: float = 1.0, modelo=None, conf: float = 0.4) -> dict[str, float]:
    """Classe → fração dos quadros amostrados em que aparece (pessoa inclusa)."""
    import cv2

    modelo = modelo or modelo_objetos()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    passo = max(1, int(round(fps * a_cada_s)))
    contagem: dict[str, int] = {}
    amostras, idx = 0, 0
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        if idx % passo == 0:
            r = modelo.predict(quadro, imgsz=640, conf=conf, verbose=False)[0]
            nomes = {r.names[int(c)] for c in r.boxes.cls.cpu().numpy()} if r.boxes is not None else set()
            for n in nomes:
                contagem[n] = contagem.get(n, 0) + 1
            amostras += 1
        idx += 1
    cap.release()
    return {NOMES_PT.get(k, k): round(v / max(1, amostras), 3)
            for k, v in sorted(contagem.items(), key=lambda kv: -kv[1])}


def area_do_paciente(s: SeriePose, margem: float = 0.15) -> np.ndarray | None:
    """Envoltória (p5–p95) das caixas do paciente, com margem relativa."""
    ok = ~np.isnan(s.caixa).any(axis=1)
    if ok.sum() < 5:
        return None
    c = s.caixa[ok]
    x1, y1 = np.percentile(c[:, 0], 5), np.percentile(c[:, 1], 5)
    x2, y2 = np.percentile(c[:, 2], 95), np.percentile(c[:, 3], 95)
    mx, my = (x2 - x1) * margem, (y2 - y1) * margem
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my])


def _intervalos(mascara: np.ndarray, quadros: np.ndarray, fps_video: float, min_s: float) -> list[tuple[float, float]]:
    bordas = np.diff(np.concatenate([[0], mascara.astype(int), [0]]))
    saida = []
    for a, b in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)):
        ini, fim = quadros[a] / fps_video, quadros[b - 1] / fps_video
        if fim - ini >= min_s:
            saida.append((round(float(ini), 1), round(float(fim), 1)))
    return saida


CONF_PESSOA_EXTRA = 0.6
SOBREPOSICAO_MIN = 0.2     # fração da caixa da outra pessoa que precisa estar dentro da área do paciente


def pessoa_visivel(s: SeriePose, conf: float = CONF_PESSOA_EXTRA) -> np.ndarray:
    """(T,) bool: alguma outra pessoa detectada com confiança, em qualquer lugar do quadro."""
    if s.outras is None:
        return s.pessoas >= 2
    return np.nan_to_num(s.outras[..., 4] >= conf, nan=0).astype(bool).any(axis=1)


def pessoa_na_area(s: SeriePose, area: np.ndarray | None = None) -> np.ndarray:
    """(T,) bool: alguma *outra* pessoa, com confiança, dentro da área crítica do paciente.

    Contar pessoas em qualquer lugar do quadro gera falso alarme com quem passa ao
    fundo ou com vultos cortados na borda; o que importa é a área do paciente.
    """
    if s.outras is None:
        return s.pessoas >= 2            # cache antigo, sem as caixas: cai na contagem simples
    area = area_do_paciente(s) if area is None else area
    if area is None:
        return np.zeros(len(s.quadros), bool)
    o = s.outras
    ix = np.clip(np.minimum(o[..., 2], area[2]) - np.maximum(o[..., 0], area[0]), 0, None)
    iy = np.clip(np.minimum(o[..., 3], area[3]) - np.maximum(o[..., 1], area[1]), 0, None)
    area_o = (o[..., 2] - o[..., 0]) * (o[..., 3] - o[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        dentro = (ix * iy / area_o >= SOBREPOSICAO_MIN) & (o[..., 4] >= CONF_PESSOA_EXTRA)
    return np.nan_to_num(dentro, nan=0).astype(bool).any(axis=1)


def eventos_pessoas_extras(s: SeriePose, fps_video: float = 30.0, min_s: float = 1.0) -> list[Evento]:
    """Intervalos com outra pessoa dentro da área crítica do paciente."""
    return [Evento("pessoa_na_area", a, b, "outra pessoa na área do paciente")
            for a, b in _intervalos(pessoa_na_area(s), s.quadros, fps_video, min_s)]


def queda(s: SeriePose, fps_video: float = 30.0, min_s: float = 1.0) -> list[Evento]:
    """Quadril perto da altura dos tornozelos **e** caixa deitada (larga > alta), sustentado."""
    quadril = np.nanmean(np.stack([s.ponto("quadril_e")[:, 1], s.ponto("quadril_d")[:, 1]]), axis=0)
    tornozelo = np.nanmean(np.stack([s.ponto("tornozelo_e")[:, 1], s.ponto("tornozelo_d")[:, 1]]), axis=0)
    ombro = np.nanmean(np.stack([s.ponto("ombro_e")[:, 1], s.ponto("ombro_d")[:, 1]]), axis=0)
    altura_corpo = np.abs(tornozelo - ombro)
    largura = s.caixa[:, 2] - s.caixa[:, 0]
    altura = s.caixa[:, 3] - s.caixa[:, 1]
    with np.errstate(invalid="ignore"):
        no_chao = (np.abs(tornozelo - quadril) < 0.25 * s.altura) & (largura > altura)
    no_chao &= ~np.isnan(altura_corpo)
    return [Evento("queda", a, b, "paciente no chão (quadril na altura dos pés, corpo deitado)")
            for a, b in _intervalos(no_chao, s.quadros, fps_video, min_s)]
