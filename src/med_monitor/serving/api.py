"""API REST + painel da equipe médica em tempo real.

```
med-monitor serve --port 8000      # painel em http://localhost:8000
```

| Endpoint | O que faz |
|---|---|
| `GET /` | painel da equipe médica (leitos ao vivo + fila de alertas) |
| `GET /stream` | Server-Sent Events: `vitais`, `alerta`, `reconhecimento`, `simulacao` |
| `GET /health` | status do serviço e da simulação |
| `POST /simulacao/iniciar` · `POST /simulacao/parar` | replay acelerado dos leitos de UTI |
| `GET /pacientes` | estado atual de cada leito (sinais, NEWS2, episódio aberto) |
| `GET /alertas` | alertas emitidos, mais recentes primeiro |
| `POST /alertas/{id}/ciente` | reconhecimento do alerta por um profissional (auditado) |
| `POST /achados` | entrada de achados de outras modalidades (áudio, vídeo, prescrição) na fusão |
| `GET /auditoria` | trilha de auditoria |
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from importlib import resources
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..alertas.despacho import Despachante
from ..config import HOSPITAL
from ..domain import Achado, Modalidade
from ..fusao.motor import MotorFusao
from .simulador import Simulador


class IniciarSimulacao(BaseModel):
    velocidade: float = Field(120.0, ge=0, le=3600, description="minutos de UTI por minuto real × 60 (120 = 2 min/s)")


class Ciencia(BaseModel):
    profissional: str = Field(..., min_length=3, max_length=80, examples=["CRM-SP 111.222"])
    nota: str = Field("", max_length=300)


class AchadoEntrada(BaseModel):
    paciente_id: str = Field(..., examples=["PRT-2026-200102"])
    modalidade: Modalidade
    tipo: str = Field(..., examples=["fala_ofegante"])
    escore: float = Field(..., ge=0, le=1)
    fonte: str = Field(..., examples=["azure-speech+librosa"])
    evidencia: dict[str, Any] = Field(default_factory=dict)
    momento: datetime | None = Field(None, description="padrão: relógio atual do plantão")


def _sse(tipo: str, dados: Any) -> str:
    return f"event: {tipo}\ndata: {json.dumps(dados, ensure_ascii=False, default=str)}\n\n"


def criar_app(despachante: Despachante | None = None, motor: MotorFusao | None = None) -> FastAPI:
    despachante = despachante or Despachante()
    motor = motor or MotorFusao()
    simulador = Simulador(despachante, motor)

    @asynccontextmanager
    async def ciclo(_: FastAPI):
        yield
        simulador.parar()

    app = FastAPI(title=f"med-monitor — {HOSPITAL}", version=__version__, lifespan=ciclo,
                  description="Monitoramento multimodal de pacientes com alertas para a equipe médica.")
    app.state.despachante, app.state.motor, app.state.simulador = despachante, motor, simulador

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def painel() -> str:
        return resources.files(__package__).joinpath("painel.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "versao": __version__, "hospital": HOSPITAL, "simulacao": simulador.status()}

    # ------------------------------------------------------------ simulação
    @app.post("/simulacao/iniciar")
    async def iniciar(cfg: IniciarSimulacao | None = None) -> dict:
        if simulador.rodando:
            raise HTTPException(409, "simulação já está rodando")
        simulador.velocidade = (cfg or IniciarSimulacao()).velocidade
        simulador.iniciar()
        return simulador.status()

    @app.post("/simulacao/parar")
    async def parar() -> dict:
        simulador.parar()
        return simulador.status()

    @app.get("/simulacao")
    def simulacao() -> dict:
        return {**simulador.status(), "eventos_injetados": simulador.eventos_injetados}

    # ------------------------------------------------------------- consultas
    @app.get("/pacientes")
    def pacientes() -> list[dict]:
        return sorted(simulador.estado.values(), key=lambda e: e["leito"])

    @app.get("/alertas")
    def alertas(limite: int = 100, paciente_id: str | None = None) -> list[dict]:
        return despachante.listar(limite, paciente_id)

    @app.post("/alertas/{alerta_id}/ciente")
    async def ciente(alerta_id: str, corpo: Ciencia) -> dict:
        try:
            with despachante.trilha.trace("api", f"ciência do alerta {alerta_id}") as tid:
                return despachante.reconhecer(alerta_id, corpo.profissional, tid, corpo.nota)
        except KeyError:
            raise HTTPException(404, "alerta não encontrado") from None

    @app.post("/achados")
    async def receber_achado(entrada: AchadoEntrada) -> dict:
        """Ponto de entrada das outras modalidades: o achado entra na mesma fusão dos sinais vitais."""
        achado = Achado(paciente_id=entrada.paciente_id, modalidade=entrada.modalidade, tipo=entrada.tipo,
                        escore=entrada.escore, fonte=entrada.fonte, evidencia=entrada.evidencia,
                        momento=entrada.momento or simulador.momento_atual())
        estado = simulador.estado.get(entrada.paciente_id, {})
        with despachante.trilha.trace(entrada.fonte, f"achado {entrada.modalidade.value}/{entrada.tipo}",
                                      paciente_id=entrada.paciente_id) as tid:
            despachante.trilha.event(tid, "achado", **achado.as_dict())
            emitidos = [despachante.despachar(d, tid, contexto={"leito": estado.get("leito"),
                                                                "vitais": estado.get("vitais")})
                        for d in motor.processar(achado)]
        return {"achado": achado.as_dict(), "alertas": emitidos}

    @app.get("/auditoria")
    def auditoria(trace_id: str | None = None, limite: int = 100) -> list[dict]:
        return despachante.trilha.read(trace_id, limite)

    # ------------------------------------------------------------ streaming
    @app.get("/stream")
    async def stream(request: Request) -> StreamingResponse:
        fila = despachante.assinar()

        async def gerar():
            try:
                yield _sse("snapshot", {"simulacao": simulador.status(),
                                        "pacientes": pacientes(), "alertas": despachante.listar(50)})
                while not await request.is_disconnected():
                    try:
                        tipo, dados = await asyncio.wait_for(fila.get(), timeout=15)
                        yield _sse(tipo, dados)
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                despachante.cancelar(fila)

        return StreamingResponse(gerar(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = criar_app()
