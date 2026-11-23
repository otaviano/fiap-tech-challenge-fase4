"""Replay acelerado de leitos de UTI — o "tempo real" da demonstração (decisão D3).

Cada leito reproduz um registro do MIMIC-III minuto a minuto, numa velocidade
configurável (120× = 1 h de UTI em 30 s). A detecção é streaming de verdade:
o monitor vê um ponto por vez, sem olhar o futuro. Só a fonte é gravada.

Para a demonstração ser reproduzível, alguns leitos recebem eventos injetados
em minutos fixos (ver `LEITOS_PADRAO`); os demais mostram só o que o paciente
real teve. O relógio é comum a todos os leitos e começa às 07:00 do dia.

As primeiras `aquecimento` amostras (3 h) são processadas sem pausa e sem
emitir alertas: é o histórico que os detectores usam para aprender o basal de
cada paciente antes de o plantão começar a ser monitorado.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..anomalias.sinais_vitais import LIMITES_PLAUSIVEIS, MonitorSinaisVitais, pontuar_news2
from ..audit import get_logger
from ..dados import physionet, sintetico
from ..fusao.motor import MotorFusao
from ..alertas.despacho import Despachante

_log = get_logger("simulador")


@dataclass(frozen=True)
class Leito:
    leito: str
    paciente_id: str
    registro: str | None = None                       # None → série sintética
    eventos: tuple[tuple[str, int, int], ...] = ()    # (tipo, minuto de início, duração)
    semente: int = 0


# Pacientes no formato de prontuário da Fase 3 (PRT-AAAA-NNNNNN).
LEITOS_PADRAO: tuple[Leito, ...] = (
    Leito("UTI-01", "PRT-2026-200101", "p000020-2183-04-28-17-47n"),
    Leito("UTI-02", "PRT-2026-200102", "p000177-2125-11-29-12-35n", eventos=(("dessaturacao", 225, 25),)),
    Leito("UTI-03", "PRT-2026-200103", "p000217-2126-09-30-12-28n",
          eventos=(("deterioracao_combinada", 200, 70),)),
    Leito("UTI-04", "PRT-2026-200104", "p000263-2160-01-08-23-52n", eventos=(("taquicardia", 250, 30),)),
    Leito("UTI-05", "PRT-2026-200105", "p000123-2161-10-29-03-01n"),
    Leito("UTI-06", "PRT-2026-200106", "p000214-2188-10-24-16-23n"),
)

INICIO_PLANTAO = "07:00"


def _relogio_base() -> pd.Timestamp:
    hoje = datetime.now(timezone.utc).date()
    return pd.Timestamp(f"{hoje} {INICIO_PLANTAO}", tz="UTC")


def preparar_serie(leito: Leito, inicio: pd.Timestamp, aquecimento: int) -> tuple[pd.DataFrame, list]:
    """Carrega a série, injeta os eventos do leito e re-ancora no relógio do plantão."""
    if leito.registro:
        serie = physionet.carregar(leito.registro)
    else:
        serie = sintetico.gerar_basal(horas=24, semente=leito.semente)
    eventos = []
    for tipo, minuto, duracao in leito.eventos:
        serie, ev = sintetico.injetar(serie, tipo, minuto, duracao)
        eventos.append(ev)
    # o minuto `aquecimento` vira o início do plantão
    deslocamento = inicio - pd.Timestamp(serie.index[aquecimento]).tz_localize("UTC")
    serie = serie.copy()
    serie.index = serie.index.tz_localize("UTC") + deslocamento
    eventos = [sintetico.EventoInjetado(e.tipo, e.inicio.tz_localize("UTC") + deslocamento,
                                        e.fim.tz_localize("UTC") + deslocamento, e.sinais, e.clinico)
               for e in eventos]
    return serie, eventos


def _para_exibir(amostra: dict) -> dict:
    saida = {}
    for sinal, valor in amostra.items():
        lim = LIMITES_PLAUSIVEIS.get(sinal)
        ok = valor is not None and not np.isnan(valor) and (not lim or lim[0] <= valor <= lim[1])
        saida[sinal] = round(float(valor), 1) if ok else None
    return saida


class Simulador:
    def __init__(self, despachante: Despachante, motor: MotorFusao | None = None,
                 velocidade: float = 120.0, aquecimento: int = 180):
        self.despachante = despachante
        self.motor = motor or MotorFusao()
        self.velocidade = velocidade
        self.aquecimento = aquecimento
        self.estado: dict[str, dict] = {}
        self.eventos_injetados: list[dict] = []
        self.relogio: datetime | None = None
        self.inicio: datetime | None = None
        self.rodando = False
        self._parar = False
        self._tarefa: asyncio.Task | None = None

    # --------------------------------------------------------------- controle
    def iniciar(self, leitos: tuple[Leito, ...] = LEITOS_PADRAO) -> None:
        if self.rodando:
            return
        self._tarefa = asyncio.get_running_loop().create_task(self.executar(leitos))

    def parar(self) -> None:
        self._parar = True

    def status(self) -> dict:
        return {"rodando": self.rodando, "velocidade": self.velocidade,
                "relogio": self.relogio.isoformat() if self.relogio else None,
                "leitos": len(self.estado)}

    def momento_atual(self) -> datetime:
        """Relógio do plantão — usado para carimbar achados de outras modalidades."""
        return self.relogio or datetime.now(timezone.utc)

    # -------------------------------------------------------------- execução
    async def executar(self, leitos: tuple[Leito, ...] = LEITOS_PADRAO) -> None:
        self.rodando, self._parar = True, False
        self.estado.clear()
        self.eventos_injetados.clear()
        inicio = _relogio_base()
        self.inicio = inicio.to_pydatetime()
        trilha = self.despachante.trilha
        self.despachante.publicar("simulacao", {"rodando": True, "velocidade": self.velocidade})
        try:
            with trilha.trace("simulador", f"plantão com {len(leitos)} leitos a {self.velocidade:g}×") as tid:
                await asyncio.gather(*(self._leito(l, inicio, tid) for l in leitos))
        finally:
            self.rodando = False
            self.despachante.publicar("simulacao", {"rodando": False})

    async def _leito(self, leito: Leito, inicio: pd.Timestamp, trace_id: str) -> None:
        serie, eventos = await asyncio.to_thread(preparar_serie, leito, inicio, self.aquecimento)
        self.eventos_injetados += [{"leito": leito.leito, "paciente_id": leito.paciente_id, **e.as_dict()}
                                   for e in eventos]
        monitor = MonitorSinaisVitais(leito.paciente_id)
        pausa = 60.0 / self.velocidade if self.velocidade > 0 else 0.0
        self.estado[leito.paciente_id] = {"leito": leito.leito, "paciente_id": leito.paciente_id,
                                          "registro": leito.registro, "status": "aquecendo"}

        for i, (momento, linha) in enumerate(zip(serie.index, serie.to_dict("records"))):
            if self._parar:
                break
            achados = monitor.processar(momento, linha)
            if i < self.aquecimento:
                if i % 30 == 0:
                    await asyncio.sleep(0)          # não monopoliza o loop durante o aquecimento
                continue

            agora = momento.to_pydatetime()
            self.relogio = agora if self.relogio is None else max(self.relogio, agora)
            vitais = _para_exibir(linha)
            pontos = pontuar_news2({k: v for k, v in vitais.items() if v is not None})
            news2 = sum(pontos.values())
            estado = self.estado[leito.paciente_id]
            estado.update(status="monitorando", momento=agora.isoformat(), vitais=vitais, news2_parcial=news2, news2_pontos=pontos,
                          episodio=self.motor.episodio_aberto(leito.paciente_id, agora))
            for decisao in self.motor.processar_varios(achados):
                self.despachante.despachar(decisao, trace_id, contexto={"leito": leito.leito, "vitais": vitais})
            estado["episodio"] = self.motor.episodio_aberto(leito.paciente_id, agora)
            self.despachante.publicar("vitais", estado)
            await asyncio.sleep(pausa)

        self.estado[leito.paciente_id]["status"] = "fim do registro" if not self._parar else "pausado"
        self.despachante.publicar("vitais", self.estado[leito.paciente_id])
        _log.info("leito %s encerrado", leito.leito)
