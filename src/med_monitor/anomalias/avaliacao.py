"""Avaliação dos detectores de sinais vitais com anomalias injetadas.

Protocolo:

1. **Fundo real**: registros do MIMIC-III (PhysioNet), com ruído, deriva e
   artefatos de monitor de verdade.
2. **Injeção**: os seis padrões clínicos de `dados.sintetico` mais dois
   artefatos de sensor, em posições aleatórias (várias sementes), depois de 3 h
   de aquecimento.
3. **Métricas por detector e para o sistema (união)**:
   - *recall*: evento detectado se houver achado clínico entre o início e o
     fim + tolerância;
   - *latência*: minutos entre o início do evento e o primeiro achado;
   - *carga de alarme*: achados fora de qualquer evento, por 24 h. No fundo
     real isso **não é** igual a falso positivo — o paciente de UTI tem
     anomalias reais não rotuladas —, por isso também medimos a carga no
     registro sem injeção;
   - *alertas causados por artefato*: achados clínicos durante um artefato
     injetado que **não** aparecem no mesmo registro sem injeção (deveria ser
     zero). A comparação é necessária porque o paciente real pode ter um
     evento genuíno coincidindo com a janela do artefato.

Eventos cujo trecho original está sem sinal (sensor desconectado em > 30% da
janela) são descartados: não há o que detectar ali, e contá-los mediria o
sensor, não o detector.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import ANOMALIAS_RESULTS
from ..dados import physionet, sintetico
from .sinais_vitais import LIMITES_PLAUSIVEIS, MonitorSinaisVitais, detectores_padrao

TOLERANCIA_MIN = 10
FONTES = ("news2", "desvio-robusto", "isolation-forest")
SISTEMA = "sistema"


@dataclass
class ResultadoRegistro:
    registro: str
    semente: int
    horas: float
    eventos: list[dict] = field(default_factory=list)          # um por evento clínico avaliável
    descartados: int = 0
    fora_de_evento: dict[str, int] = field(default_factory=dict)
    em_artefato: dict[str, int] = field(default_factory=dict)


def _valido(serie: pd.Series, sinal: str) -> pd.Series:
    lo, hi = LIMITES_PLAUSIVEIS[sinal]
    return serie.between(lo, hi)


def _avaliavel(fundo: pd.DataFrame, ev: sintetico.EventoInjetado) -> bool:
    janela = fundo.loc[ev.inicio:ev.fim]
    return all(_valido(janela[s], s).mean() >= 0.7 for s in ev.sinais)


def _tipos_para(n_minutos: int, semente: int, aquecimento: int = 180, espaco: int = 90) -> list[str]:
    """Todos os tipos se couberem; senão, um subconjunto que gira com a semente."""
    tipos = list(sintetico.TIPOS_CLINICOS) + ["artefato", "artefato"]
    cabem = max(1, (n_minutos - aquecimento) // (espaco + 60))
    if cabem >= len(tipos):
        return tipos
    desloc = semente % len(tipos)
    return (tipos[desloc:] + tipos[:desloc])[:cabem]


def _clinicos(achados) -> list:
    return [a for a in achados if a.fonte in FONTES]


def _existe_no_basal(achado, basal, folga=pd.Timedelta(minutes=5)) -> bool:
    m = pd.Timestamp(achado.momento)
    return any(b.fonte == achado.fonte and b.tipo == achado.tipo
               and abs(pd.Timestamp(b.momento) - m) <= folga for b in basal)


def _monitor(nome: str, params_if: dict | None) -> MonitorSinaisVitais:
    return MonitorSinaisVitais(nome, detectores_padrao(params_if))


def avaliar_registro(fundo: pd.DataFrame, nome: str, semente: int,
                     basal: list | None = None, params_if: dict | None = None) -> ResultadoRegistro:
    serie, eventos = sintetico.injetar_varios(fundo, _tipos_para(len(fundo), semente), semente=semente)
    achados = _clinicos(_monitor(nome, params_if).processar_serie(serie))
    basal = basal if basal is not None else _clinicos(_monitor(nome, params_if).processar_serie(fundo))
    tol = pd.Timedelta(minutes=TOLERANCIA_MIN)

    res = ResultadoRegistro(nome, semente, round(len(fundo) / 60, 1))
    dentro = np.zeros(len(achados), dtype=bool)
    em_artefato = {f: 0 for f in (*FONTES, SISTEMA)}
    momentos = pd.to_datetime([a.momento for a in achados]) if achados else pd.DatetimeIndex([])

    for ev in eventos:
        na_janela = (momentos >= ev.inicio) & (momentos <= ev.fim + tol)
        dentro |= na_janela
        if not ev.clinico:
            for i in np.flatnonzero(na_janela):
                if _existe_no_basal(achados[i], basal):
                    continue
                em_artefato[achados[i].fonte] += 1
                em_artefato[SISTEMA] += 1
            continue
        if not _avaliavel(fundo, ev):
            res.descartados += 1
            continue
        latencias = {}
        for fonte in FONTES:
            idx = [i for i in np.flatnonzero(na_janela) if achados[i].fonte == fonte]
            if idx:
                latencias[fonte] = (momentos[idx[0]] - ev.inicio).total_seconds() / 60
        if latencias:
            latencias[SISTEMA] = min(latencias.values())
        res.eventos.append({"tipo": ev.tipo, "inicio": ev.inicio.isoformat(), "latencias": latencias})

    fora = {f: 0 for f in (*FONTES, SISTEMA)}
    for i in np.flatnonzero(~dentro):
        fora[achados[i].fonte] += 1
        fora[SISTEMA] += 1
    res.fora_de_evento, res.em_artefato = fora, em_artefato
    return res


def carga_basal(achados: list, fundo: pd.DataFrame) -> dict[str, float]:
    """Achados por 24 h no registro real, sem nada injetado."""
    dias = len(fundo) / 1440
    carga = {f: sum(a.fonte == f for a in achados) / dias for f in FONTES}
    carga[SISTEMA] = len(achados) / dias
    return carga


def _tarefa(args):
    nome, sementes, params_if = args
    fundo = physionet.carregar(nome)
    basal = _clinicos(_monitor(nome, params_if).processar_serie(fundo))
    return ([avaliar_registro(fundo, nome, s, basal, params_if) for s in sementes],
            carga_basal(basal, fundo), len(fundo))


def consolidar(resultados: list[ResultadoRegistro], cargas: dict[str, dict], minutos: dict[str, int]) -> dict:
    eventos = [e for r in resultados for e in r.eventos]
    horas_injetadas = sum(r.horas for r in resultados)
    por_detector = {}
    for fonte in (*FONTES, SISTEMA):
        lat = [e["latencias"][fonte] for e in eventos if fonte in e["latencias"]]
        por_tipo = {}
        for tipo in sintetico.TIPOS_CLINICOS:
            do_tipo = [e for e in eventos if e["tipo"] == tipo]
            if do_tipo:
                por_tipo[tipo] = round(sum(fonte in e["latencias"] for e in do_tipo) / len(do_tipo), 3)
        por_detector[fonte] = {
            "recall": round(len(lat) / len(eventos), 3) if eventos else None,
            "latencia_mediana_min": round(float(np.median(lat)), 1) if lat else None,
            "recall_por_tipo": por_tipo,
            "alertas_fora_de_evento_24h": round(
                sum(r.fora_de_evento[fonte] for r in resultados) / (horas_injetadas / 24), 1),
            "alertas_em_artefato": sum(r.em_artefato[fonte] for r in resultados),
            "carga_basal_24h": round(
                sum(cargas[n][fonte] * minutos[n] for n in cargas) / sum(minutos.values()), 1),
        }
    return {
        "registros": len(cargas),
        "horas_de_monitoramento": round(sum(minutos.values()) / 60, 1),
        "execucoes": len(resultados),
        "eventos_clinicos_avaliados": len(eventos),
        "eventos_descartados_sem_sinal": sum(r.descartados for r in resultados),
        "eventos_por_tipo": {t: sum(e["tipo"] == t for e in eventos) for t in sintetico.TIPOS_CLINICOS},
        "tolerancia_min": TOLERANCIA_MIN,
        "detectores": por_detector,
    }


def avaliar(sementes=(1, 2, 3), registros=None, processos: int | None = None,
            params_if: dict | None = None) -> dict:
    nomes = registros or [r.nome for r in physionet.REGISTROS]
    for nome in nomes:  # baixa antes de paralelizar (evita downloads concorrentes)
        physionet.carregar(nome)
    resultados, cargas, minutos = [], {}, {}
    with ProcessPoolExecutor(max_workers=processos) as pool:
        for nome, (res, carga, n) in zip(nomes, pool.map(_tarefa, [(n, sementes, params_if) for n in nomes])):
            resultados.extend(res)
            cargas[nome], minutos[nome] = carga, n
    resumo = consolidar(resultados, cargas, minutos)
    resumo["params_if"] = params_if or {}
    resumo["detalhe"] = [r.__dict__ for r in resultados]
    return resumo


def tabela_markdown(resumo: dict) -> str:
    nomes = {"news2": "NEWS2 (limiar clínico)", "desvio-robusto": "Desvio robusto (basal do paciente)",
             "isolation-forest": "Isolation Forest (multivariado)", SISTEMA: "**Sistema (união)**"}
    linhas = [
        f"Fundo real: {resumo['registros']} registros do MIMIC-III, "
        f"{resumo['horas_de_monitoramento']} h de monitoramento · {resumo['execucoes']} execuções · "
        f"{resumo['eventos_clinicos_avaliados']} eventos clínicos avaliados "
        f"({resumo['eventos_descartados_sem_sinal']} descartados por sensor sem sinal) · "
        f"tolerância {resumo['tolerancia_min']} min",
        "",
        "| Detector | Recall | Latência mediana (min) | Achados fora de evento / 24 h | Achados causados por artefato | Carga no registro sem injeção / 24 h |",
        "|---|---|---|---|---|---|",
    ]
    for f, d in resumo["detectores"].items():
        linhas.append(f"| {nomes[f]} | {d['recall']:.0%} | {d['latencia_mediana_min']} | "
                      f"{d['alertas_fora_de_evento_24h']} | {d['alertas_em_artefato']} | {d['carga_basal_24h']} |")
    tipos = list(sintetico.TIPOS_CLINICOS)
    linhas += ["", "Recall por tipo de evento:", "",
               "| Detector | " + " | ".join(tipos) + " |", "|---|" + "---|" * len(tipos)]
    for f, d in resumo["detectores"].items():
        linhas.append(f"| {nomes[f]} | " + " | ".join(
            f"{d['recall_por_tipo'].get(t, 0):.0%}" for t in tipos) + " |")
    linhas += ["", "n por tipo: " + ", ".join(f"{t}={n}" for t, n in resumo["eventos_por_tipo"].items())]
    return "\n".join(linhas) + "\n"


def salvar(resumo: dict, destino=ANOMALIAS_RESULTS) -> tuple:
    destino.mkdir(parents=True, exist_ok=True)
    js = destino / "avaliacao_sinais_vitais.json"
    md = destino / "avaliacao_sinais_vitais.md"
    js.write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md.write_text("# Avaliação — anomalias em sinais vitais\n\n" + tabela_markdown(resumo), encoding="utf-8")
    return js, md
