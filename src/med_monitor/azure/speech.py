"""Azure AI Speech — Speech to Text das gravações de consulta (pt-BR).

Usa reconhecimento contínuo sobre o arquivo inteiro (consultas passam de 15 s,
o limite do reconhecimento de uma frase só) e pede carimbo de tempo por
palavra: além do texto, o relatório sabe *quando* cada frase foi dita, o que
permite cruzar termos críticos com o trecho do áudio.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import AzureSettings

_TICKS_POR_SEGUNDO = 10_000_000   # o SDK mede offset/duração em ticks de 100 ns


@dataclass(frozen=True)
class Segmento:
    inicio_s: float
    fim_s: float
    texto: str
    confianca: float | None = None


@dataclass
class Transcricao:
    texto: str
    fonte: str                                   # "azure-speech" | "referencia"
    idioma: str = "pt-BR"
    segmentos: list[Segmento] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class ClienteSpeech:
    def __init__(self, settings: AzureSettings | None = None):
        self.settings = settings or AzureSettings()

    @property
    def disponivel(self) -> bool:
        return self.settings.speech_configurado

    def transcrever(self, caminho: Path, timeout_s: float = 300) -> Transcricao:
        if not self.disponivel:
            raise RuntimeError("Azure Speech não configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)")
        import azure.cognitiveservices.speech as sdk

        cfg = sdk.SpeechConfig(subscription=self.settings.speech_key, region=self.settings.speech_region)
        cfg.speech_recognition_language = self.settings.speech_language
        cfg.request_word_level_timestamps()
        cfg.output_format = sdk.OutputFormat.Detailed
        reconhecedor = sdk.SpeechRecognizer(speech_config=cfg,
                                            audio_config=sdk.audio.AudioConfig(filename=str(caminho)))

        segmentos: list[Segmento] = []
        erros: list[str] = []
        fim = threading.Event()

        def reconhecido(evt):
            r = evt.result
            if r.reason == sdk.ResultReason.RecognizedSpeech and r.text:
                confianca = None
                try:   # formato detalhado: a confiança vem na melhor hipótese (NBest[0])
                    confianca = round(float(json.loads(r.json)["NBest"][0]["Confidence"]), 3)
                except (KeyError, IndexError, ValueError, TypeError):
                    pass
                ini = r.offset / _TICKS_POR_SEGUNDO
                segmentos.append(Segmento(round(ini, 2), round(ini + r.duration / _TICKS_POR_SEGUNDO, 2),
                                          r.text, confianca))

        def cancelado(evt):
            if evt.cancellation_details.reason == sdk.CancellationReason.Error:
                erros.append(evt.cancellation_details.error_details)
            fim.set()

        reconhecedor.recognized.connect(reconhecido)
        reconhecedor.canceled.connect(cancelado)
        reconhecedor.session_stopped.connect(lambda _: fim.set())
        reconhecedor.start_continuous_recognition()
        fim.wait(timeout_s)
        reconhecedor.stop_continuous_recognition()
        if erros:
            raise RuntimeError(f"Azure Speech falhou: {erros[0]}")
        return Transcricao(" ".join(s.texto for s in segmentos), "azure-speech",
                           self.settings.speech_language, segmentos)


def transcricao_de_referencia(caminho_wav: Path) -> Transcricao | None:
    """Sem Azure: usa o roteiro `<arquivo>.txt` ao lado do áudio, se existir (amostras sintéticas)."""
    txt = Path(caminho_wav).with_suffix(".txt")
    if not txt.exists():
        return None
    return Transcricao(txt.read_text(encoding="utf-8").strip(), "referencia")
