"""Pose por quadro com YOLOv8-pose (17 keypoints COCO — o formato do OpenPose COCO).

Decisão D1 do plano: o OpenPose original (CMU, Caffe) exige build com CUDA, e a
máquina não tem GPU. O YOLOv8-pose produz os mesmos 17 pontos do modelo COCO do
OpenPose, roda em CPU e vem no mesmo pacote do YOLOv8 de objetos.

**Qual pessoa é o paciente.** O REHAB24-6 tem quadros com outras pessoas (o
fisioterapeuta, quem opera a câmera). O paciente é escolhido pela continuidade:
no primeiro quadro, a maior pessoa detectada; depois, a detecção cujo centro
está mais perto do paciente no quadro anterior (rastreamento simples por
proximidade). As demais pessoas não somem: o número de pessoas por quadro é
guardado e alimenta o detector de áreas críticas.

A série é guardada em cache (`data/processed/pose/`) — extrair pose em CPU é a
etapa cara; as métricas e o classificador rodam sobre o cache em segundos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PROCESSED_DIR, ROOT, VideoSettings

CACHE = PROCESSED_DIR / "pose"
MODELOS = ROOT / "models"

KEYPOINTS = ("nariz", "olho_e", "olho_d", "orelha_e", "orelha_d", "ombro_e", "ombro_d", "cotovelo_e",
             "cotovelo_d", "punho_e", "punho_d", "quadril_e", "quadril_d", "joelho_e", "joelho_d",
             "tornozelo_e", "tornozelo_d")
K = {nome: i for i, nome in enumerate(KEYPOINTS)}
MAX_OUTRAS = 4
VERSAO_CACHE = 2       # v2: guarda as caixas das demais pessoas (área crítica)


@dataclass
class SeriePose:
    video: str
    fps: float                 # fps efetivo da série (fps do vídeo / passo)
    quadros: np.ndarray        # (T,) índice do quadro no vídeo original
    xy: np.ndarray             # (T, 17, 2) em pixels; NaN quando não há paciente
    conf: np.ndarray           # (T, 17)
    caixa: np.ndarray          # (T, 4) x1, y1, x2, y2 do paciente
    pessoas: np.ndarray        # (T,) pessoas detectadas no quadro
    largura: int
    altura: int
    outras: np.ndarray | None = None   # (T, MAX_OUTRAS, 5) x1, y1, x2, y2, conf das demais pessoas (NaN = vazio)

    def salvar(self, caminho: Path) -> Path:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(caminho, video=self.video, fps=self.fps, quadros=self.quadros, xy=self.xy,
                            conf=self.conf, caixa=self.caixa, pessoas=self.pessoas,
                            dims=np.array([self.largura, self.altura]),
                            outras=self.outras if self.outras is not None else np.full((len(self.quadros), MAX_OUTRAS, 5), np.nan))
        return caminho

    @classmethod
    def carregar(cls, caminho: Path) -> "SeriePose":
        d = np.load(caminho)
        return cls(str(d["video"]), float(d["fps"]), d["quadros"], d["xy"], d["conf"], d["caixa"],
                   d["pessoas"], int(d["dims"][0]), int(d["dims"][1]),
                   d["outras"] if "outras" in d.files else None)

    def ponto(self, nome: str, conf_min: float = 0.3) -> np.ndarray:
        """(T, 2) de um keypoint; NaN onde a confiança é baixa."""
        p = self.xy[:, K[nome]].copy()
        p[self.conf[:, K[nome]] < conf_min] = np.nan
        return p


def modelo_pose(nome: str | None = None):
    from ultralytics import YOLO
    import torch

    torch.backends.nnpack.set_flags(False)   # sem suporte nesta CPU: evita um aviso por inferência

    nome = nome or VideoSettings().pose_model
    MODELOS.mkdir(parents=True, exist_ok=True)
    return YOLO(str(MODELOS / nome))


def _escolher(caixas: np.ndarray, anterior: np.ndarray | None) -> int:
    areas = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    if anterior is None:
        return int(np.argmax(areas))
    centros = (caixas[:, :2] + caixas[:, 2:]) / 2
    ref = (anterior[:2] + anterior[2:]) / 2
    dist = np.linalg.norm(centros - ref, axis=1)
    # proximidade manda; área desempata entre candidatos igualmente próximos
    return int(np.argmin(dist - 1e-3 * np.sqrt(areas)))


def extrair(video: Path, passo: int = 2, lote: int = 16, modelo=None, imgsz: int = 640,
            limite_quadros: int | None = None) -> SeriePose:
    import cv2

    modelo = modelo or modelo_pose()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura, altura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    quadros, xy, conf, caixas, pessoas, outras = [], [], [], [], [], []
    anterior = None
    pendentes: list[tuple[int, np.ndarray]] = []

    def processar(lote_quadros):
        nonlocal anterior
        resultados = modelo.predict([q for _, q in lote_quadros], imgsz=imgsz, verbose=False, conf=0.25)
        for (idx, _), r in zip(lote_quadros, resultados):
            quadros.append(idx)
            n = 0 if r.boxes is None else len(r.boxes)
            pessoas.append(n)
            demais = np.full((MAX_OUTRAS, 5), np.nan)
            if n == 0 or r.keypoints is None:
                xy.append(np.full((17, 2), np.nan)); conf.append(np.zeros(17)); caixas.append(np.full(4, np.nan))
                outras.append(demais)
                continue
            bxs = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            i = _escolher(bxs, anterior)
            resto = [j for j in np.argsort(-confs) if j != i][:MAX_OUTRAS]
            for k, j in enumerate(resto):
                demais[k] = [*bxs[j], confs[j]]
            outras.append(demais)
            anterior = bxs[i]
            xy.append(r.keypoints.xy[i].cpu().numpy())
            conf.append(r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17))
            caixas.append(bxs[i])

    idx = 0
    while True:
        ok, quadro = cap.read()
        if not ok or (limite_quadros is not None and idx >= limite_quadros):
            break
        if idx % passo == 0:
            pendentes.append((idx, quadro))
            if len(pendentes) == lote:
                processar(pendentes)
                pendentes = []
        idx += 1
    if pendentes:
        processar(pendentes)
    cap.release()

    return SeriePose(Path(video).stem, fps / passo, np.array(quadros), np.array(xy, dtype=float),
                     np.array(conf, dtype=float), np.array(caixas, dtype=float), np.array(pessoas),
                     largura, altura, np.array(outras, dtype=float).reshape(-1, MAX_OUTRAS, 5))


def extrair_com_cache(video: Path, passo: int = 2, **kw) -> SeriePose:
    caminho = CACHE / f"{Path(video).stem}_p{passo}_v{VERSAO_CACHE}.npz"
    if caminho.exists():
        return SeriePose.carregar(caminho)
    serie = extrair(video, passo=passo, **kw)
    serie.salvar(caminho)
    return serie
