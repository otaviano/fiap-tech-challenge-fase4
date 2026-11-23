"""Configuração central (caminhos e variáveis de ambiente)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # o .env é conveniência local; sem python-dotenv, valem só as variáveis do ambiente
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

RESULTS_DIR = ROOT / "results"
VIDEO_RESULTS = RESULTS_DIR / "video"
AUDIO_RESULTS = RESULTS_DIR / "audio"
ANOMALIAS_RESULTS = RESULTS_DIR / "anomalias"
LOGS_DIR = RESULTS_DIR / "logs"

AUDIT_FILE = LOGS_DIR / "auditoria.jsonl"
APP_LOG = LOGS_DIR / "med_monitor.log"

HOSPITAL = "Hospital Santa Helena"
SEED = 42


@dataclass(frozen=True)
class AzureSettings:
    """Credenciais dos serviços gerenciados. Nunca versionadas — vêm do `.env`."""

    speech_key: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", ""))
    speech_region: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", "brazilsouth"))
    speech_language: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_LANGUAGE", "pt-BR"))
    language_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_LANGUAGE_ENDPOINT", ""))
    language_key: str = field(default_factory=lambda: os.getenv("AZURE_LANGUAGE_KEY", ""))

    @property
    def speech_configurado(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_configurado(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


@dataclass(frozen=True)
class VideoSettings:
    yolo_model: str = field(default_factory=lambda: os.getenv("YOLO_MODEL", "yolov8n.pt"))
    pose_model: str = field(default_factory=lambda: os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt"))


def ensure_dirs() -> None:
    """Cria as pastas de saída que o pipeline assume existirem."""
    for path in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, VIDEO_RESULTS, AUDIO_RESULTS,
                 ANOMALIAS_RESULTS, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
