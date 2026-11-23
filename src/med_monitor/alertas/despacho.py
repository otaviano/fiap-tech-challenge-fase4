"""Despacho de alertas: auditoria, barramento em tempo real e canais externos.

Todo alerta emitido pela fusão passa por aqui, nesta ordem:

1. **Trilha de auditoria** (`alerta_emitido`), com os achados que o
   fundamentam — antes de qualquer canal, para que nenhum alerta chegue à
   equipe sem registro.
2. **Barramento**: publicado para cada assinante (o painel, via SSE).
3. **Canais externos** opcionais (`CanalWebhook` para Teams/Slack/pager),
   desligados por padrão. Falha de canal externo é registrada, não derruba o
   despacho.

O ciclo fecha com o **reconhecimento**: um profissional marca o alerta como
ciente, e a trilha guarda quem e em quanto tempo — a "resposta à anomalia"
que o enunciado pede para demonstrar.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from collections import deque
from datetime import datetime, timezone
from typing import Protocol

from ..audit import AuditTrail, get_audit_trail, get_logger
from ..fusao.motor import Decisao

_log = get_logger("alertas")


class Canal(Protocol):
    nome: str

    def enviar(self, registro: dict) -> None: ...


class CanalWebhook:
    """POST JSON para um webhook (Teams, Slack, pager). Só alertas a partir de `minima`."""

    nome = "webhook"
    _PESO = {"critico": 3, "alto": 2, "moderado": 1, "informativo": 0}

    def __init__(self, url: str, minima: str = "alto", timeout: float = 5.0):
        self.url, self.minima, self.timeout = url, minima, timeout

    def enviar(self, registro: dict) -> None:
        if self._PESO[registro["severidade"]] < self._PESO[self.minima]:
            return
        corpo = json.dumps({"text": f"[{registro['severidade'].upper()}] {registro['paciente_id']} — "
                                    f"{registro['mensagem']}", "alerta": registro},
                           ensure_ascii=False, default=str).encode()
        req = urllib.request.Request(self.url, data=corpo, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=self.timeout).close()


class Despachante:
    def __init__(self, trilha: AuditTrail | None = None, canais: tuple[Canal, ...] = (),
                 historico: int = 500):
        self.trilha = trilha or get_audit_trail()
        self.canais = canais
        self.historico: deque[dict] = deque(maxlen=historico)
        self._por_id: dict[str, dict] = {}
        self._assinantes: set[asyncio.Queue] = set()

    # ------------------------------------------------------------ barramento
    def assinar(self, tamanho: int = 1000) -> asyncio.Queue:
        fila: asyncio.Queue = asyncio.Queue(maxsize=tamanho)
        self._assinantes.add(fila)
        return fila

    def cancelar(self, fila: asyncio.Queue) -> None:
        self._assinantes.discard(fila)

    def publicar(self, tipo: str, dados: dict) -> None:
        """Entrega a todos os assinantes; quem estiver lento perde mensagens, não trava os demais."""
        for fila in list(self._assinantes):
            try:
                fila.put_nowait((tipo, dados))
            except asyncio.QueueFull:
                pass

    # --------------------------------------------------------------- alertas
    def despachar(self, decisao: Decisao, trace_id: str, contexto: dict | None = None) -> dict:
        registro = decisao.alerta.as_dict()
        registro["achados_detalhe"] = [a.as_dict() for a in decisao.achados]
        registro["contexto"] = contexto or {}
        registro["reconhecido"] = None
        registro["emitido_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self.trilha.event(trace_id, "alerta_emitido", alerta_id=registro["id"],
                          paciente_id=registro["paciente_id"], codigo=registro["codigo"],
                          severidade=registro["severidade"], destinatario=registro["destinatario"],
                          achados=list(registro["achados"]), mensagem=registro["mensagem"])
        self.historico.append(registro)
        self._por_id[registro["id"]] = registro
        self.publicar("alerta", registro)

        for canal in self.canais:
            try:
                canal.enviar(registro)
            except Exception as exc:  # canal externo nunca derruba o despacho
                self.trilha.event(trace_id, "canal_falhou", canal=canal.nome, alerta_id=registro["id"],
                                  erro=type(exc).__name__)
                _log.warning("canal %s falhou para alerta %s: %s", canal.nome, registro["id"], exc)
        return registro

    def reconhecer(self, alerta_id: str, profissional: str, trace_id: str, nota: str = "") -> dict:
        registro = self._por_id.get(alerta_id)
        if registro is None:
            raise KeyError(alerta_id)
        if registro["reconhecido"]:
            return registro
        agora = datetime.now(timezone.utc)
        emitido = datetime.fromisoformat(registro["emitido_em"])
        registro["reconhecido"] = {"profissional": profissional, "em": agora.isoformat(timespec="seconds"),
                                   "segundos_ate_ciencia": round((agora - emitido).total_seconds(), 1),
                                   "nota": nota[:300]}
        self.trilha.event(trace_id, "alerta_reconhecido", alerta_id=alerta_id,
                          paciente_id=registro["paciente_id"], **registro["reconhecido"])
        self.publicar("reconhecimento", {"id": alerta_id, **registro["reconhecido"]})
        return registro

    def listar(self, limite: int = 100, paciente_id: str | None = None) -> list[dict]:
        itens = [r for r in self.historico if not paciente_id or r["paciente_id"] == paciente_id]
        return itens[-limite:][::-1]
