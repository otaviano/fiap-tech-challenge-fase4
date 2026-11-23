"""Logging detalhado e trilha de auditoria.

Duas saídas, com propósitos distintos:

- `results/logs/med_monitor.log` — log de aplicação legível, para operação e demo.
- `results/logs/auditoria.jsonl` — trilha estruturada, uma linha JSON por evento,
  para rastreamento e auditoria clínica. É append-only: eventos não são
  reescritos nem removidos.

Portado da Fase 3. Cada interação (uma análise de vídeo, um áudio processado,
uma janela de sinais vitais) recebe um `trace_id`, compartilhado por todos os
eventos dela — achados, chamadas ao Azure, alertas emitidos —, então é possível
reconstruir *por que* um alerta chegou à equipe médica.

TODO(fase4): `trace()` ainda tem a assinatura do assistente da Fase 3
(`usuario`, `pergunta`); generalizar para `origem`/`descricao` quando o primeiro
analisador for plugado.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import APP_LOG, AUDIT_FILE, LOGS_DIR

_configured = False


def setup_logging(level: int = logging.INFO, log_file: Path = APP_LOG) -> logging.Logger:
    """Configura o log de aplicação uma única vez por processo."""
    global _configured
    logger = logging.getLogger("med_monitor")
    if _configured:
        return logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    arquivo = logging.FileHandler(log_file, encoding="utf-8")
    arquivo.setFormatter(formato)
    console = logging.StreamHandler()
    console.setFormatter(formato)

    logger.setLevel(level)
    logger.handlers = [arquivo, console]
    logger.propagate = False
    _configured = True
    return logger


def get_logger(nome: str = "med_monitor") -> logging.Logger:
    setup_logging()
    return logging.getLogger(nome if nome.startswith("med_monitor") else f"med_monitor.{nome}")


class AuditTrail:
    """Escreve a trilha estruturada. Um objeto por processo é suficiente."""

    def __init__(self, caminho: Path = AUDIT_FILE):
        self.caminho = caminho
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("auditoria")

    def event(self, trace_id: str, evento: str, **campos: Any) -> dict:
        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace_id": trace_id,
            "evento": evento,
            **campos,
        }
        with self.caminho.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        self._logger.info("%s | trace=%s | %s", evento, trace_id[:8],
                          " ".join(f"{k}={v}" for k, v in campos.items() if k != "resposta"))
        return registro

    @contextmanager
    def trace(self, usuario: str, pergunta: str, paciente_id: str | None = None) -> Iterator[str]:
        """Abre uma interação: emite início e fim, com latência medida."""
        trace_id = str(uuid.uuid4())
        inicio = time.perf_counter()
        self.event(trace_id, "interacao_iniciada", usuario=usuario,
                   paciente_id=paciente_id, pergunta=pergunta[:500])
        try:
            yield trace_id
        except Exception as exc:  # a falha também é auditável
            self.event(trace_id, "interacao_falhou", erro=type(exc).__name__, detalhe=str(exc)[:300],
                       latencia_ms=round((time.perf_counter() - inicio) * 1000, 1))
            raise
        else:
            self.event(trace_id, "interacao_concluida",
                       latencia_ms=round((time.perf_counter() - inicio) * 1000, 1))

    def read(self, trace_id: str | None = None, limite: int = 50) -> list[dict]:
        """Releitura da trilha — usada pela API `/audit` e pela demonstração."""
        if not self.caminho.exists():
            return []
        registros = [json.loads(linha) for linha in
                     self.caminho.read_text(encoding="utf-8").splitlines() if linha.strip()]
        if trace_id:
            registros = [r for r in registros if r["trace_id"] == trace_id]
        return registros[-limite:]


_trail: AuditTrail | None = None


def get_audit_trail(caminho: Path | None = None) -> AuditTrail:
    """Trilha compartilhada do processo.

    Sem argumento, devolve a trilha em uso — inclusive uma que tenha sido
    substituída (testes, ou um destino configurado no bootstrap da aplicação).
    Só recria quando um caminho diferente é pedido explicitamente.
    """
    global _trail
    if _trail is None or (caminho is not None and _trail.caminho != caminho):
        _trail = AuditTrail(caminho or AUDIT_FILE)
    return _trail
