"""Contratos compartilhados entre as modalidades.

É o ponto de encontro do fluxo multimodal. Cada analisador (vídeo, áudio,
sinais vitais, prescrições, movimentação) produz `Achado`s — observações com
evidência e escore — sem saber da existência dos outros. A fusão consome
achados de qualquer origem e decide os `Alerta`s.

Mesma filosofia da Fase 3: o modelo *observa*, a regra *decide*. Um alerta
sempre aponta para os achados que o fundamentam, para que a equipe médica (e a
auditoria) consiga responder "por que este alerta disparou?".
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Modalidade(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXTO = "texto"                  # transcrição, evolução clínica
    SINAIS_VITAIS = "sinais_vitais"
    PRESCRICAO = "prescricao"
    MOVIMENTACAO = "movimentacao"


class Severidade(str, Enum):
    """Mesma escala da Fase 3, ordenada da mais grave para a menos grave."""

    CRITICO = "critico"
    ALTO = "alto"
    MODERADO = "moderado"
    INFORMATIVO = "informativo"

    @property
    def peso(self) -> int:
        return {"critico": 3, "alto": 2, "moderado": 1, "informativo": 0}[self.value]

    def elevar(self) -> "Severidade":
        """Um nível acima — usado quando modalidades distintas concordam."""
        ordem = [Severidade.INFORMATIVO, Severidade.MODERADO, Severidade.ALTO, Severidade.CRITICO]
        return ordem[min(ordem.index(self) + 1, len(ordem) - 1)]


def _agora() -> datetime:
    return datetime.now(timezone.utc)


def _novo_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Achado:
    """Uma observação anômala produzida por um analisador de uma modalidade."""

    paciente_id: str
    modalidade: Modalidade
    tipo: str                         # ex.: "taquicardia", "disartria", "queda_postural"
    escore: float                     # 0..1 — quão fora do padrão (não é probabilidade clínica)
    fonte: str                        # modelo/serviço que gerou: "yolov8n-pose", "azure-speech"...
    evidencia: dict[str, Any] = field(default_factory=dict)
    momento: datetime = field(default_factory=_agora)
    id: str = field(default_factory=_novo_id)

    def __post_init__(self) -> None:
        if not 0.0 <= self.escore <= 1.0:
            raise ValueError(f"escore deve estar em [0, 1], recebido {self.escore}")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["modalidade"] = self.modalidade.value
        d["momento"] = self.momento.isoformat()
        return d


@dataclass(frozen=True)
class Alerta:
    """Alerta destinado à equipe médica, rastreável até os achados que o originaram."""

    paciente_id: str
    codigo: str
    severidade: Severidade
    mensagem: str
    achados: tuple[str, ...]          # ids dos `Achado`s que fundamentam o alerta
    modalidades: tuple[Modalidade, ...]
    destinatario: str = "equipe assistencial"
    momento: datetime = field(default_factory=_agora)
    id: str = field(default_factory=_novo_id)

    @property
    def multimodal(self) -> bool:
        return len(set(self.modalidades)) > 1

    def as_dict(self) -> dict:
        d = asdict(self)
        d["severidade"] = self.severidade.value
        d["modalidades"] = [m.value for m in self.modalidades]
        d["momento"] = self.momento.isoformat()
        d["multimodal"] = self.multimodal
        return d
