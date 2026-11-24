"""Anomalias na evolução de prescrições — "alterações inesperadas no tratamento".

Dois detectores complementares, como nos sinais vitais:

| Detector | Pergunta | Pega | Não pega |
|---|---|---|---|
| **Regras do formulário** | "Esta prescrição viola algo que o hospital definiu?" | dose acima da máxima, possível erro decimal (×10), duplicidade terapêutica, suspensão abrupta de quem exige desmame, mudança brusca de intervalo | mudança grande mas dentro dos limites |
| **Variação aprendida** | "Esta mudança de dose é atípica *para este medicamento*?" | dobrar a varfarina (que costuma variar ±15%), quando a insulina pode variar ±40% sem nada de errado | o que nunca mudou no histórico |

A variação aprendida é estimada no **histórico do hospital** (internações sem
anomalia): para cada medicamento, a distribuição do log da razão entre a dose
diária nova e a anterior nos ajustes. Uma mudança com |z robusto| ≥ 3,5 é
atípica.

Tudo é streaming: cada evento é avaliado contra o estado de prescrições ativas
do paciente, no momento em que é registrado — o alerta chega à farmácia
clínica antes da primeira dose.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..dados.prescricoes import CLASSES_SEM_DUPLICIDADE, FORMULARIO, EventoPrescricao
from ..domain import Achado, Modalidade

FATOR_ERRO_DECIMAL = 8.0      # ≥ 8× (ou ≤ 1/8) da dose anterior: provável vírgula no lugar errado
FATOR_SALTO = 2.0             # ≥ 2× na dose diária sem passar da máxima
FATOR_INTERVALO = 3.0         # intervalo encurtado ou alongado 3× ou mais
Z_ATIPICO = 3.5
JANELA_DESMAME = timedelta(hours=48)


def _achado(ev: EventoPrescricao, tipo: str, escore: float, fonte: str, **evidencia) -> Achado:
    m = ev.med
    return Achado(ev.paciente_id, Modalidade.PRESCRICAO, tipo, float(np.clip(escore, 0, 1)), fonte,
                  evidencia={"medicamento": ev.medicamento, "classe": m.classe,
                             "prescricao": ev.como_prontuario(), "alta_vigilancia": m.alta_vigilancia,
                             "prescritor": ev.prescritor, **evidencia},
                  momento=ev.momento)


@dataclass
class ModeloVariacao:
    """Distribuição robusta do log(dose diária nova / anterior) por medicamento."""

    por_medicamento: dict[str, tuple[float, float, int]] = field(default_factory=dict)   # mediana, escala, n
    escala_min: float = 0.15       # piso: ninguém titula com precisão melhor que ±15%

    @classmethod
    def ajustar(cls, internacoes: list[list[EventoPrescricao]], n_min: int = 20) -> "ModeloVariacao":
        razoes: dict[str, list[float]] = defaultdict(list)
        for eventos in internacoes:
            ultimo: dict[str, EventoPrescricao] = {}
            for ev in eventos:
                if ev.acao == "ajustar" and ev.medicamento in ultimo:
                    razoes[ev.medicamento].append(float(np.log(ev.dose_diaria / ultimo[ev.medicamento].dose_diaria)))
                if ev.acao in ("iniciar", "ajustar"):
                    ultimo[ev.medicamento] = ev
                else:
                    ultimo.pop(ev.medicamento, None)
        modelo = cls()
        for nome, r in razoes.items():
            if len(r) >= n_min:
                arr = np.array(r)
                med = float(np.median(arr))
                modelo.por_medicamento[nome] = (med, float(1.4826 * np.median(np.abs(arr - med))), len(r))
        return modelo

    def z(self, medicamento: str, razao: float) -> float | None:
        if medicamento not in self.por_medicamento or razao <= 0:
            return None
        med, escala, _ = self.por_medicamento[medicamento]
        return (np.log(razao) - med) / max(escala, self.escala_min)

    def salvar(self, caminho: Path) -> Path:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(self.por_medicamento, ensure_ascii=False, indent=2), encoding="utf-8")
        return caminho

    @classmethod
    def carregar(cls, caminho: Path) -> "ModeloVariacao":
        return cls({k: tuple(v) for k, v in json.loads(caminho.read_text(encoding="utf-8")).items()})


class MonitorPrescricoes:
    """Estado das prescrições ativas de um paciente e avaliação de cada novo evento."""

    def __init__(self, modelo: ModeloVariacao | None = None):
        self.modelo = modelo
        self.ativos: dict[str, EventoPrescricao] = {}
        self.historico: dict[str, list[EventoPrescricao]] = defaultdict(list)
        self._sinalizados: set[str] = set()      # medicamentos cuja prescrição ativa já gerou achado

    def processar(self, ev: EventoPrescricao) -> list[Achado]:
        achados = self._avaliar(ev)
        if achados:
            self._sinalizados.add(ev.medicamento)
        elif ev.acao in ("ajustar", "suspender", "alta"):
            self._sinalizados.discard(ev.medicamento)
        self.historico[ev.medicamento].append(ev)
        if ev.acao in ("suspender", "alta"):
            self.ativos.pop(ev.medicamento, None)
        else:
            self.ativos[ev.medicamento] = ev
        return achados

    def processar_varios(self, eventos: list[EventoPrescricao]) -> list[Achado]:
        return [a for ev in eventos for a in self.processar(ev)]

    # ----------------------------------------------------------------- regras
    def _avaliar(self, ev: EventoPrescricao) -> list[Achado]:
        if ev.acao == "alta":
            return []
        m = ev.med
        anterior = self.ativos.get(ev.medicamento)
        achados: list[Achado] = []
        regra = "regras-formulario"

        if ev.acao == "suspender":
            return self._suspensao(ev, anterior)

        # correção de uma prescrição que já foi sinalizada: é a resposta ao alerta, não um novo erro
        if anterior is not None and ev.medicamento in self._sinalizados and ev.dose_diaria < anterior.dose_diaria:
            return []

        # possível erro decimal: dose ~10× a anterior (ou a usual, ao iniciar)
        referencia = anterior.dose if anterior else float(np.mean(m.dose_usual))
        razao_dose = ev.dose / referencia if referencia else 1.0
        if razao_dose >= FATOR_ERRO_DECIMAL or razao_dose <= 1 / FATOR_ERRO_DECIMAL:
            achados.append(_achado(ev, "possivel_erro_decimal", 0.95, regra, razao=round(razao_dose, 2),
                                   dose_referencia=referencia))

        acima = []
        if ev.dose > m.dose_max:
            acima.append(f"dose {ev.dose:g} {m.unidade} > máxima {m.dose_max:g} {m.unidade}")
        if ev.dose_diaria > m.max_diario:
            acima.append(f"dose diária {ev.dose_diaria:g} {m.unidade} > máxima {m.max_diario:g} {m.unidade}")
        if acima and not achados:
            achados.append(_achado(ev, "dose_acima_maxima", 0.9 if m.alta_vigilancia else 0.8, regra,
                                   motivos=acima))

        if ev.acao == "iniciar" and m.classe in CLASSES_SEM_DUPLICIDADE:
            iguais = [a for nome, a in self.ativos.items() if nome != ev.medicamento and a.med.classe == m.classe]
            if iguais:
                achados.append(_achado(ev, "duplicidade_terapeutica", 0.75, regra,
                                       ja_ativo=[a.medicamento for a in iguais]))

        if anterior is not None and not achados:
            if ev.intervalo_h != anterior.intervalo_h:
                fator_int = max(ev.intervalo_h, anterior.intervalo_h) / min(ev.intervalo_h, anterior.intervalo_h)
                if fator_int >= FATOR_INTERVALO:
                    achados.append(_achado(ev, "intervalo_inesperado", 0.65, regra,
                                           intervalo_anterior_h=anterior.intervalo_h, intervalo_novo_h=ev.intervalo_h))
            razao_diaria = ev.dose_diaria / anterior.dose_diaria
            if not achados and (razao_diaria >= FATOR_SALTO or razao_diaria <= 1 / (FATOR_SALTO * 1.5)):
                achados.append(_achado(ev, "salto_de_dose", 0.6, regra, razao_dose_diaria=round(razao_diaria, 2),
                                       dose_diaria_anterior=anterior.dose_diaria))

        # variação aprendida: só para ajustes que as regras não explicaram
        if anterior is not None and not achados and self.modelo is not None:
            z = self.modelo.z(ev.medicamento, ev.dose_diaria / anterior.dose_diaria)
            if z is not None and abs(z) >= Z_ATIPICO:
                med, escala, n = self.modelo.por_medicamento[ev.medicamento]
                achados.append(_achado(ev, "alteracao_atipica", min(0.5 + abs(z) / 40, 0.7), "variacao-aprendida",
                                       z=round(float(z), 1),
                                       razao_dose_diaria=round(ev.dose_diaria / anterior.dose_diaria, 2),
                                       variacao_tipica=f"±{(np.exp(escala) - 1) * 100:.0f}%", n_historico=n))
        return achados

    def _suspensao(self, ev: EventoPrescricao, anterior: EventoPrescricao | None) -> list[Achado]:
        m = ev.med
        if anterior is None or m.desmame_acima_diario is None or anterior.dose_diaria <= m.desmame_acima_diario:
            return []
        recentes = [e for e in self.historico[ev.medicamento] if ev.momento - e.momento <= JANELA_DESMAME]
        houve_reducao = any(b.dose_diaria < a.dose_diaria for a, b in zip(recentes, recentes[1:]))
        if houve_reducao:
            return []
        return [_achado(ev, "suspensao_abrupta", 0.65, "regras-formulario",
                        dose_diaria_ao_suspender=anterior.dose_diaria,
                        desmame_necessario_acima_de=m.desmame_acima_diario)]


# Rótulo injetado → tipos de achado que contam como detecção correta
CORRESPONDENCIA = {
    "erro_decimal": {"possivel_erro_decimal", "dose_acima_maxima"},
    "dose_acima_maxima": {"dose_acima_maxima", "possivel_erro_decimal"},
    "salto_dose": {"salto_de_dose", "alteracao_atipica"},
    "duplicidade": {"duplicidade_terapeutica"},
    "suspensao_abrupta": {"suspensao_abrupta"},
    "frequencia_inesperada": {"intervalo_inesperado", "dose_acima_maxima", "salto_de_dose", "alteracao_atipica"},
    "ajuste_atipico": {"alteracao_atipica", "salto_de_dose"},
}


# --------------------------------------------------------------- avaliação

def avaliar(n_treino: int = 600, n_teste: int = 800, fracao_anomala: float = 0.6, semente: int = 7) -> dict:
    """Treina a variação no histórico (sem anomalias) e mede o detector em internações novas."""
    from ..dados import prescricoes as dados

    modelo = ModeloVariacao.ajustar(dados.gerar_coorte(n_treino, semente, 0.0, prefixo="PRT-2026-4"))
    teste = dados.gerar_coorte(n_teste, semente + 1, fracao_anomala)
    resultado: dict = {"internacoes_treino": n_treino, "internacoes_teste": n_teste,
                       "modelo_variacao": {k: {"variacao_tipica": f"±{(np.exp(max(v[1], 0.15)) - 1) * 100:.0f}%",
                                               "n": v[2]} for k, v in modelo.por_medicamento.items()},
                       "detectores": {}}
    for nome, mod in (("regras", None), ("regras + variação aprendida", modelo)):
        por_tipo: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        limpas = alertas_limpas = 0
        alertas_por_tipo: dict[str, int] = defaultdict(int)
        for internacao in teste:
            mon = MonitorPrescricoes(mod)
            sem_anomalia = not any(e.anomalia for e in internacao)
            for ev in internacao:
                achados = mon.processar(ev)
                if ev.anomalia:
                    por_tipo[ev.anomalia][1] += 1
                    por_tipo[ev.anomalia][0] += any(a.tipo in CORRESPONDENCIA[ev.anomalia] for a in achados)
                elif sem_anomalia and ev.acao != "alta":
                    limpas += 1
                    alertas_limpas += len(achados)
                    for a in achados:
                        alertas_por_tipo[a.tipo] += 1
        total = sum(n for _, n in por_tipo.values())
        resultado["detectores"][nome] = {
            "recall_por_tipo": {k: f"{d}/{n}" for k, (d, n) in sorted(por_tipo.items())},
            "recall": round(sum(d for d, _ in por_tipo.values()) / max(1, total), 3),
            "prescricoes_sem_anomalia": limpas,
            "alertas_sem_anomalia": alertas_limpas,
            "taxa_alerta_sem_anomalia": round(alertas_limpas / max(1, limpas), 4),
            "alertas_sem_anomalia_por_tipo": dict(alertas_por_tipo),
        }
    resultado["_modelo"] = modelo
    return resultado


def tabela_markdown(res: dict) -> str:
    tipos = list(next(iter(res["detectores"].values()))["recall_por_tipo"])
    linhas = [f"Treino: {res['internacoes_treino']} internações sem anomalia (histórico) · "
              f"teste: {res['internacoes_teste']} internações novas, 60% com uma anomalia injetada", "",
              "| Detector | " + " | ".join(tipos) + " | Recall | Alertas em internações sem anomalia |",
              "|---|" + "---|" * (len(tipos) + 2)]
    for nome, d in res["detectores"].items():
        linhas.append(f"| {nome} | " + " | ".join(d["recall_por_tipo"][t] for t in tipos)
                      + f" | {d['recall']:.0%} | {d['alertas_sem_anomalia']}/{d['prescricoes_sem_anomalia']} "
                        f"({d['taxa_alerta_sem_anomalia']:.1%}) |")
    linhas += ["", "Variação típica aprendida por medicamento: " + ", ".join(
        f"{k} {v['variacao_tipica']} (n={v['n']})" for k, v in res["modelo_variacao"].items())]
    return "\n".join(linhas) + "\n"


def salvar(res: dict, destino: Path) -> tuple[Path, Path, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    modelo = res.pop("_modelo")
    js = destino / "avaliacao_prescricoes.json"
    js.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = destino / "avaliacao_prescricoes.md"
    md.write_text("# Avaliação — anomalias em prescrições\n\n" + tabela_markdown(res), encoding="utf-8")
    res["_modelo"] = modelo
    return js, md, modelo.salvar(destino / "modelo_variacao_prescricoes.json")
