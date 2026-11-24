"""REHAB24-6 — vídeos de fisioterapia com execuções corretas e incorretas.

Černek, Sedmidubský, Budíková — *REHAB24-6: Physical Therapy Dataset for
Analyzing Pose Estimation Methods*, SISAP 2024. Zenodo 13305826, CC BY-NC 4.0.

65 gravações, 6 exercícios, 10 pessoas, duas câmeras (17 e 18, ortogonais).
`Segmentation.csv` traz, para cada uma das 1.072 repetições, o quadro inicial
e final, o exercício, o lado, a orientação da pessoa e o rótulo de **correção**
dado por fisioterapeuta — o *ground truth* do detector de desvios.

**Download seletivo.** O `videos.zip` tem 2,65 GB e o Zenodo entrega a ~340
KB/s (≈ 2 h). Como o índice de um zip fica no fim do arquivo e o Zenodo aceita
requisições por faixa de bytes, `RemoteZip` lê o índice e baixa **só os vídeos
escolhidos** — dezenas de MB em vez de gigabytes.
"""

from __future__ import annotations

import csv
import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import EXTERNAL_DIR

DESTINO = EXTERNAL_DIR / "rehab24-6"
BASE = "https://zenodo.org/records/13305826/files"
URL_VIDEOS = f"{BASE}/videos.zip?download=1"

EXERCICIOS = {
    1: "abdução de braço",
    2: "arm VW",
    3: "flexão apoiada na mesa",
    4: "abdução de perna",
    5: "avanço (lunge)",
    6: "agachamento",
}


class _ArquivoRemoto(io.RawIOBase):
    """Arquivo somente-leitura sobre HTTP Range — o suficiente para o `zipfile` navegar."""

    def __init__(self, url: str, timeout: float = 60):
        self.timeout = timeout
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            self.tamanho = int(r.headers["Content-Length"])
            self.url = r.url                      # segue o redirecionamento uma vez só
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, deslocamento: int, origem: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self.tamanho}[origem]
        self._pos = base + deslocamento
        return self._pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.tamanho - self._pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{self._pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            dados = r.read()
        buffer[:len(dados)] = dados
        self._pos += len(dados)
        return len(dados)


class RemoteZip:
    def __init__(self, url: str = URL_VIDEOS):
        self._zip = zipfile.ZipFile(io.BufferedReader(_ArquivoRemoto(url), buffer_size=1 << 20))

    def nomes(self) -> list[str]:
        return [i.filename for i in self._zip.infolist() if not i.is_dir()]

    def tamanho(self, nome: str) -> int:
        return self._zip.getinfo(nome).file_size

    def extrair(self, nome: str, destino: Path) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_suffix(destino.suffix + ".part")
        with self._zip.open(nome) as origem, open(parcial, "wb") as saida:
            shutil.copyfileobj(origem, saida, length=1 << 20)
        parcial.rename(destino)
        return destino


@dataclass(frozen=True)
class Repeticao:
    video_id: str
    numero: int
    exercicio: int
    pessoa: int
    quadro_inicio: int
    quadro_fim: int
    orientacao_cam17: str
    lado: str
    correta: bool
    pessoa_extra_cam17: int = 0    # 0 nenhuma · 1 desprezível · 2 perceptível · 3 grande parte visível
    pessoa_extra_cam18: int = 0

    def pessoa_extra(self, camera: int) -> int:
        return self.pessoa_extra_cam17 if camera == 17 else self.pessoa_extra_cam18

    @property
    def orientacao_cam18(self) -> str:
        return {"front": "side", "side": "front", "profile": "front"}.get(self.orientacao_cam17, self.orientacao_cam17)


def baixar_segmentacao() -> Path:
    caminho = DESTINO / "Segmentation.csv"
    if not caminho.exists():
        DESTINO.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/Segmentation.csv?download=1", caminho)
    return caminho


def repeticoes() -> list[Repeticao]:
    with open(baixar_segmentacao(), encoding="utf-8") as fh:
        return [Repeticao(r["video_id"], int(r["repetition_number"]), int(r["exercise_id"]), int(r["person_id"]),
                          int(r["first_frame"]), int(r["last_frame"]), r["cam17_orientation"],
                          r["exercise_subtype"], r["correctness"] == "1",
                          int(r["extra_person_in_cam17"] or 0), int(r["extra_person_in_cam18"] or 0))
                for r in csv.DictReader(fh, delimiter=";")]


def caminho_video(video_id: str, camera: int) -> Path:
    return DESTINO / "videos" / f"{video_id}_c{camera}.mp4"


def baixar_videos(video_ids: list[str], camera: int = 17, zip_remoto: RemoteZip | None = None) -> list[Path]:
    """Baixa só os vídeos pedidos, de uma câmera, a partir do zip remoto."""
    faltando = [v for v in video_ids if not caminho_video(v, camera).exists()]
    if not faltando:
        return [caminho_video(v, camera) for v in video_ids]
    zr = zip_remoto or RemoteZip()
    indice = {Path(n).name: n for n in zr.nomes()}
    for v in faltando:
        candidatos = [n for nome, n in indice.items() if nome.startswith(f"{v}-Camera{camera}")]
        if not candidatos:
            raise FileNotFoundError(f"{v} (câmera {camera}) não está no zip")
        zr.extrair(candidatos[0], caminho_video(v, camera))
    return [caminho_video(v, camera) for v in video_ids]


# Plano do movimento → câmera que o enxerga melhor. A abdução de braço acontece
# no plano frontal (câmera de frente); o agachamento, no sagital (câmera de lado).
PLANO = {1: "frontal", 2: "frontal", 3: "sagital", 4: "frontal", 5: "sagital", 6: "sagital"}
EXERCICIOS_AVALIADOS = (1, 6)


def camera_ideal(video_id: str, reps: list[Repeticao] | None = None) -> int:
    reps = [r for r in (reps or repeticoes()) if r.video_id == video_id]
    if not reps:
        raise KeyError(video_id)
    from collections import Counter

    ori17 = Counter(r.orientacao_cam17 for r in reps).most_common(1)[0][0]
    frontal = PLANO[reps[0].exercicio] == "frontal"
    if ori17 == "front":
        return 17 if frontal else 18
    if ori17 in ("side", "profile"):
        return 18 if frontal else 17
    return 17          # meio-perfil: as duas câmeras veem em diagonal


def selecao(exercicios=EXERCICIOS_AVALIADOS) -> list[tuple[str, int]]:
    """(video_id, câmera) de todos os vídeos dos exercícios, do menor para o maior id."""
    reps = repeticoes()
    vids = sorted({r.video_id for r in reps if r.exercicio in exercicios})
    return [(v, camera_ideal(v, reps)) for v in vids]
