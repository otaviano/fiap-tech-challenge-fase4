"""Consultas sintéticas em pt-BR com alterações vocais controladas.

Não há dataset público de consultas em português com rótulo de fadiga ou
dispneia. Para desenvolver e avaliar o detector com *ground truth*, geramos
falas com a alteração conhecida, por um de dois motores:

- **Vozes neurais do Azure Speech** (`pt-BR-AntonioNeural`, `pt-BR-FranciscaNeural`…),
  quando o `.env` tem a chave do Speech. Fala natural, que o Speech to Text
  transcreve bem — a voz robótica do espeak saía com confiança de 0,5–0,6 e
  trechos inteiros perdidos.
- **espeak-ng** (local, sem rede), como alternativa offline.

| Condição | Azure (SSML) | espeak-ng | O que imita |
|---|---|---|---|
| `normal` | ritmo e tom da voz | 165 palavras/min | fala habitual (basal) |
| `fadiga` | ritmo −30%, tom −12%, volume −5 dB | 115 palavras/min, tom grave, volume −5 dB | cansaço: fala lenta, grave, fraca |
| `dispneia` | pausa de 550 ms a cada 3 palavras | idem | falta de ar: frases curtas |
| `fadiga_leve`, `dispneia_leve` | as mesmas alterações, pela metade | idem | casos limítrofes |

Na dispneia, um ruído de inspiração é somado no começo de cada pausa. Todas
recebem um piso de ruído de sala (−55 dBFS). Cada `.wav` vai com seu roteiro
`.txt` — a "transcrição de referência" quando o Azure não está configurado.

A síntese neural fica em cache (`data/external/tts/`): regerar as amostras
não gasta de novo a cota gratuita (0,5 milhão de caracteres/mês).

Isto valida a **mecânica** do detector, não sua acurácia clínica: a voz é
sintética e as alterações são controladas. Para o vídeo, o ideal é o grupo
gravar os mesmos roteiros.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from ..config import EXTERNAL_DIR, RAW_DIR, AzureSettings
from .acustica import SR

DESTINO_PADRAO = RAW_DIR / "audio"
CACHE_TTS = EXTERNAL_DIR / "tts"

# Parâmetros do espeak-ng (s: palavras/min, p: tom 0–99, a: amplitude %, g: espaço entre palavras)
CONDICOES = {
    "normal": {"s": 165, "p": 50, "a": 100, "g": 0},
    "fadiga": {"s": 115, "p": 32, "a": 55, "g": 6},
    "fadiga_leve": {"s": 140, "p": 42, "a": 75, "g": 3},
    "dispneia": {"s": 160, "p": 50, "a": 100, "g": 0, "pausa": (3, 550)},
    "dispneia_leve": {"s": 160, "p": 50, "a": 100, "g": 0, "pausa": (5, 380)},
}
# Prosódia SSML equivalente para as vozes neurais (o volume é aplicado depois, igual ao espeak)
PROSODIA = {
    "normal": {"rate": "0%", "pitch": "0%"},
    "fadiga": {"rate": "-30%", "pitch": "-12%"},
    "fadiga_leve": {"rate": "-15%", "pitch": "-6%"},
    "dispneia": {"rate": "0%", "pitch": "0%"},
    "dispneia_leve": {"rate": "0%", "pitch": "0%"},
}

# Voz neural → variante do espeak-ng equivalente (usada sem Azure)
VOZ_OFFLINE = {
    "pt-BR-AntonioNeural": "pt-br+m3", "pt-BR-FranciscaNeural": "pt-br+f2",
    "pt-BR-DonatoNeural": "pt-br+m1", "pt-BR-BrendaNeural": "pt-br+f4",
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
    voz: str                 # voz neural do Azure ("pt-BR-AntonioNeural"); sem Azure, vira a do VOZ_OFFLINE
    basal: tuple[str, str]   # (roteiro, condição)
    atual: tuple[str, str]


# Pacientes dos leitos do simulador: a consulta conversa com o que o monitor mostra.
AMOSTRAS_PADRAO = (
    Amostra("PRT-2026-200102", "pt-BR-AntonioNeural", ("basal_1", "normal"), ("dispneia", "dispneia")),   # UTI-02
    Amostra("PRT-2026-200103", "pt-BR-FranciscaNeural", ("basal_2", "normal"), ("fadiga", "fadiga")),     # UTI-03
    Amostra("PRT-2026-200105", "pt-BR-DonatoNeural", ("basal_3", "normal"), ("estavel", "normal")),      # UTI-05
)


def _com_pausas(texto: str, a_cada: int = 3, pausa_ms: int = 550, escapar: bool = False) -> str:
    palavras = texto.split()
    partes = []
    for i, p in enumerate(palavras, 1):
        partes.append(escape(p) if escapar else p)
        if i % a_cada == 0 and i < len(palavras):
            partes.append(f'<break time="{pausa_ms}ms"/>')
    return " ".join(partes)


def _ssml_dispneia(texto: str, a_cada: int = 3, pausa_ms: int = 550) -> str:
    return "<speak>" + _com_pausas(texto, a_cada, pausa_ms) + "</speak>"


def ssml_azure(texto: str, condicao: str, voz: str) -> str:
    p = CONDICOES[condicao]
    corpo = _com_pausas(texto, *p["pausa"], escapar=True) if "pausa" in p else escape(texto)
    pr = PROSODIA[condicao]
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">'
            f'<voice name="{voz}"><prosody rate="{pr["rate"]}" pitch="{pr["pitch"]}">{corpo}</prosody>'
            '</voice></speak>')


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


def _tts_espeak(texto: str, condicao: str, voz: str) -> tuple[np.ndarray, int]:
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
    return np.asarray(y, dtype=float), sr


def _tts_azure(texto: str, condicao: str, voz: str, settings: AzureSettings) -> tuple[np.ndarray, int]:
    """Síntese neural com cache em disco (chave: SSML)."""
    import soundfile as sf

    ssml = ssml_azure(texto, condicao, voz)
    cache = CACHE_TTS / f"{hashlib.sha1(ssml.encode()).hexdigest()[:16]}.wav"
    if not cache.exists():
        import azure.cognitiveservices.speech as sdk

        cfg = sdk.SpeechConfig(subscription=settings.speech_key, region=settings.speech_region)
        cfg.set_speech_synthesis_output_format(sdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
        resultado = sdk.SpeechSynthesizer(speech_config=cfg, audio_config=None).speak_ssml_async(ssml).get()
        if resultado.reason != sdk.ResultReason.SynthesizingAudioCompleted:
            detalhe = getattr(resultado, "cancellation_details", None)
            raise RuntimeError(f"síntese falhou: {resultado.reason} {getattr(detalhe, 'error_details', '')}")
        CACHE_TTS.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resultado.audio_data)
    y, sr = sf.read(io.BytesIO(cache.read_bytes()))
    return np.asarray(y, dtype=float), sr


def motor_padrao(settings: AzureSettings | None = None) -> str:
    return "azure" if (settings or AzureSettings()).speech_configurado else "espeak"


def sintetizar(texto: str, condicao: str, voz: str, semente: int = 0, motor: str | None = None,
               settings: AzureSettings | None = None) -> np.ndarray:
    import librosa

    settings = settings or AzureSettings()
    motor = motor or motor_padrao(settings)
    neural = voz.endswith("Neural")
    if motor == "azure" and neural:
        y, sr = _tts_azure(texto, condicao, voz, settings)
    else:
        y, sr = _tts_espeak(texto, condicao, VOZ_OFFLINE.get(voz, voz) if neural else voz)
    p = CONDICOES[condicao]
    y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
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


def gerar_padrao(destino: Path = DESTINO_PADRAO, motor: str | None = None) -> list[tuple[Amostra, Path, Path]]:
    """Gera basal + consulta atual para cada paciente de `AMOSTRAS_PADRAO`."""
    saida = []
    for i, a in enumerate(AMOSTRAS_PADRAO):
        caminhos = []
        for rotulo, (roteiro, condicao) in (("basal", a.basal), ("atual", a.atual)):
            texto = ROTEIROS[roteiro]
            y = sintetizar(texto, condicao, a.voz, semente=i, motor=motor)
            caminhos.append(gravar(y, texto, destino / f"{a.paciente_id}_{rotulo}.wav"))
        saida.append((a, *caminhos))
    return saida
