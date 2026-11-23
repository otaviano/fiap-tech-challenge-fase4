"""Consultas sintéticas em pt-BR com alterações vocais controladas (espeak-ng).

Não há dataset público de consultas em português com rótulo de fadiga ou
dispneia. Para desenvolver e avaliar o detector com *ground truth*, geramos
falas com a alteração conhecida:

| Condição | Como é sintetizada | O que imita |
|---|---|---|
| `normal` | 165 palavras/min, tom e volume padrão | fala habitual (basal) |
| `fadiga` | 115 palavras/min, tom mais grave, volume −5 dB, mais espaço entre palavras | cansaço: fala lenta, grave, fraca |
| `dispneia` | pausa de ~550 ms a cada 3 palavras + ruído de inspiração nas pausas | falta de ar: frases curtas, interrompidas para respirar |
| `fadiga_leve`, `dispneia_leve` | as mesmas alterações, pela metade | casos limítrofes, para medir sensibilidade |

Todas recebem um piso de ruído de sala (−55 dBFS), para que o detector não se
apoie em silêncio digital perfeito. Cada `.wav` vai com seu roteiro `.txt` —
a "transcrição de referência" usada quando o Azure não está configurado.

Isto valida a **mecânica** do detector, não sua acurácia clínica: a voz é
sintética e as alterações são exageradas. Para a demonstração, o ideal é o
grupo gravar o mesmo roteiro com a própria voz.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RAW_DIR
from .acustica import SR

DESTINO_PADRAO = RAW_DIR / "audio"

CONDICOES = {
    "normal": {"s": 165, "p": 50, "a": 100, "g": 0},
    "fadiga": {"s": 115, "p": 32, "a": 55, "g": 6},
    "fadiga_leve": {"s": 140, "p": 42, "a": 75, "g": 3},
    "dispneia": {"s": 160, "p": 50, "a": 100, "g": 0, "pausa": (3, 550)},
    "dispneia_leve": {"s": 160, "p": 50, "a": 100, "g": 0, "pausa": (5, 380)},
}

ROTEIROS = {
    "basal_1": "Bom dia doutor. Hoje acordei bem disposto, consegui tomar o café da manhã inteiro. "
               "Dormi a noite toda e a enfermeira disse que a pressão está boa. Estou esperando a visita "
               "da minha filha à tarde e queria saber quando posso voltar a caminhar no corredor.",
    "basal_2": "Boa tarde. Estou me sentindo melhor que ontem, a dor na perna diminuiu bastante. "
               "Comi bem no almoço e tomei os remédios no horário. Queria perguntar se amanhã já dá "
               "para tirar o soro e se vou fazer aquele exame de sangue de novo.",
    "basal_3": "Oi doutora, tudo bem. A noite foi tranquila, só acordei uma vez para ir ao banheiro. "
               "O fisioterapeuta veio cedo e fizemos os exercícios na cadeira. Estou animado para "
               "conversar sobre a alta e sobre a dieta que vou seguir em casa.",
    "dispneia": "Doutor, estou com falta de ar desde cedo. Qualquer coisa que eu falo já canso. "
                "Tentei levantar para ir ao banheiro e precisei parar no caminho, o peito apertado. "
                "Parece que o ar não chega, e à noite piorou quando deitei.",
    "fadiga": "Estou muito cansada hoje, doutor. Sem força nenhuma, nem consegui comer direito. "
              "Tive calafrio de madrugada e a enfermeira disse que eu estava com febre. "
              "Só quero ficar deitada, uma fraqueza no corpo todo.",
    "estavel": "Estou bem, doutor. Não tenho falta de ar, nem dor no peito. Dormi bem e não tive febre. "
               "Só quero saber se a cirurgia continua marcada para quinta e se posso tomar banho sozinho.",
}


@dataclass(frozen=True)
class Amostra:
    paciente_id: str
    voz: str                 # variante do espeak-ng pelo nome do arquivo: "pt-br+m3", "pt-br+f2" (não "male3")
    basal: tuple[str, str]   # (roteiro, condição)
    atual: tuple[str, str]


# Pacientes dos leitos do simulador: a consulta conversa com o que o monitor mostra.
AMOSTRAS_PADRAO = (
    Amostra("PRT-2026-200102", "pt-br+m3", ("basal_1", "normal"), ("dispneia", "dispneia")),   # UTI-02 dessaturação
    Amostra("PRT-2026-200103", "pt-br+f2", ("basal_2", "normal"), ("fadiga", "fadiga")),     # UTI-03 deterioração
    Amostra("PRT-2026-200105", "pt-br+m1", ("basal_3", "normal"), ("estavel", "normal")),      # UTI-05 estável
)


def _ssml_dispneia(texto: str, a_cada: int = 3, pausa_ms: int = 550) -> str:
    palavras = texto.split()
    partes = []
    for i, p in enumerate(palavras, 1):
        partes.append(p)
        if i % a_cada == 0 and i < len(palavras):
            partes.append(f'<break time="{pausa_ms}ms"/>')
    return "<speak>" + " ".join(partes) + "</speak>"


def _ruido_rosa(n: int, rng: np.random.Generator) -> np.ndarray:
    branco = rng.normal(0, 1, n)
    espectro = np.fft.rfft(branco)
    f = np.arange(len(espectro)) + 1
    rosa = np.fft.irfft(espectro / np.sqrt(f), n)
    return rosa / (np.max(np.abs(rosa)) + 1e-12)


def _db(nivel_dbfs: float) -> float:
    return 10 ** (nivel_dbfs / 20)


def _inspiracoes(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ruído de inspiração (banda 300–2500 Hz, ~350 ms) no começo de cada pausa longa."""
    from scipy.signal import butter, sosfilt

    from .acustica import _com_energia, _corridas, _suavizar, PASSO
    import librosa

    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=400, hop_length=PASSO)[0] + 1e-10)
    pausas = _corridas(~_suavizar(_com_energia(rms_db)))
    sos = butter(4, [300, 2500], btype="band", fs=SR, output="sos")
    saida = y.copy()
    for a, b in pausas[1:-1]:
        ini, fim = a * PASSO, b * PASSO
        n = min(int(0.35 * SR), fim - ini)
        if n < int(0.15 * SR):
            continue
        sopro = sosfilt(sos, rng.normal(0, 1, n)) * np.hanning(n)
        saida[ini:ini + n] += sopro / (np.max(np.abs(sopro)) + 1e-12) * _db(-42)
    return saida


def sintetizar(texto: str, condicao: str, voz: str, semente: int = 0) -> np.ndarray:
    import librosa
    import soundfile as sf

    p = CONDICOES[condicao]
    cmd = ["espeak-ng", "-v", voz, "-s", str(p["s"]), "-p", str(p["p"]), "-a", str(p["a"]), "-g", str(p["g"])]
    entrada = texto
    if "pausa" in p:
        cmd.append("-m")
        entrada = _ssml_dispneia(texto, *p["pausa"])
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "tts.wav"
        subprocess.run([*cmd, "-w", str(wav), entrada], check=True, capture_output=True)
        y, sr = sf.read(wav)
    y = librosa.resample(np.asarray(y, dtype=float), orig_sr=sr, target_sr=SR)
    y = y / (np.max(np.abs(y)) + 1e-12) * _db(-3) * (p["a"] / 100)
    rng = np.random.default_rng(semente)
    if "pausa" in p:
        y = _inspiracoes(y, rng)
    y = np.concatenate([np.zeros(int(0.5 * SR)), y, np.zeros(int(0.5 * SR))])
    y = y + _ruido_rosa(len(y), rng) * _db(-55)
    return np.clip(y, -1, 1)


def gravar(y: np.ndarray, texto: str, destino: Path) -> Path:
    import soundfile as sf

    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, y, SR, subtype="PCM_16")
    destino.with_suffix(".txt").write_text(texto + "\n", encoding="utf-8")
    return destino


def gerar_padrao(destino: Path = DESTINO_PADRAO) -> list[tuple[Amostra, Path, Path]]:
    """Gera basal + consulta atual para cada paciente de `AMOSTRAS_PADRAO`."""
    saida = []
    for i, a in enumerate(AMOSTRAS_PADRAO):
        caminhos = []
        for rotulo, (roteiro, condicao) in (("basal", a.basal), ("atual", a.atual)):
            texto = ROTEIROS[roteiro]
            y = sintetizar(texto, condicao, a.voz, semente=i)
            caminhos.append(gravar(y, texto, destino / f"{a.paciente_id}_{rotulo}.wav"))
        saida.append((a, *caminhos))
    return saida
