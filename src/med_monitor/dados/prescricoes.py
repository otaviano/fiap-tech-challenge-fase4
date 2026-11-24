"""Formulário do hospital e evolução sintética de prescrições, com anomalias rotuladas.

**Valores ilustrativos, não referência clínica.** As faixas de dose abaixo
existem para dar forma realista aos dados sintéticos do Hospital Santa Helena
(fictício) — não devem ser usadas para prescrever.

Cada internação é uma sequência de eventos de prescrição (`iniciar`,
`ajustar`, `suspender`, `alta`) no formato do prontuário da Fase 3
(`medicamento`, `dose`, `via`, `frequencia`), acrescido da dose numérica e do
intervalo em horas — o que permite calcular a dose diária.

A evolução "normal" imita a rotina: revisão diária às 08:00, insulina ajustada
com frequência, antibiótico com curso de 7–10 dias, corticoide em desmame,
analgesia reduzida com a melhora — e, raramente, **escalonamentos legítimos**
(profilaxia → dose terapêutica de anticoagulante), que existem na vida real e
deixam a avaliação honesta: são mudanças grandes que *não* são erro.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class Medicamento:
    nome: str
    classe: str
    unidade: str
    via: str
    dose_usual: tuple[float, float]
    dose_max: float                 # dose única máxima
    max_diario: float
    intervalos_h: tuple[int, ...]   # intervalos usuais
    alta_vigilancia: bool = False   # lista ISMP de medicamentos potencialmente perigosos
    desmame_acima_diario: float | None = None   # suspender acima desta dose diária exige desmame
    variacao_tipica: float = 0.25   # amplitude relativa de um ajuste rotineiro (0 = dose fixa, sem ajuste)
    passo: float = 1.0              # granularidade da dose (apresentação / prática de prescrição)


FORMULARIO: dict[str, Medicamento] = {m.nome: m for m in (
    Medicamento("enoxaparina", "anticoagulante", "mg", "SC", (40, 40), 150, 300, (24,), True,
                variacao_tipica=0, passo=20),
    Medicamento("heparina", "anticoagulante", "UI", "SC", (5000, 5000), 10000, 40000, (8, 12), True,
                variacao_tipica=0, passo=2500),
    Medicamento("varfarina", "anticoagulante", "mg", "VO", (2.5, 7.5), 15, 15, (24,), True,
                variacao_tipica=0.15, passo=1.25),
    Medicamento("insulina regular", "insulina", "UI", "SC", (2, 10), 20, 100, (6,), True, variacao_tipica=0.4),
    Medicamento("morfina", "opioide", "mg", "IV", (2, 6), 15, 90, (4, 6), True, passo=1),
    Medicamento("tramadol", "opioide", "mg", "IV", (50, 100), 100, 400, (6, 8), passo=50),
    Medicamento("dipirona", "analgesico", "mg", "IV", (1000, 1000), 2000, 4000, (6,), variacao_tipica=0, passo=500),
    Medicamento("paracetamol", "analgesico", "mg", "VO", (500, 1000), 1000, 4000, (6,), variacao_tipica=0, passo=250),
    Medicamento("ceftriaxona", "antibiotico", "mg", "IV", (1000, 2000), 2000, 4000, (12, 24), variacao_tipica=0,
                passo=1000),
    Medicamento("piperacilina-tazobactam", "antibiotico", "mg", "IV", (4500, 4500), 4500, 18000, (6, 8),
                variacao_tipica=0, passo=2250),
    Medicamento("vancomicina", "antibiotico", "mg", "IV", (1000, 1500), 2000, 4000, (12,), variacao_tipica=0.3,
                passo=250),
    Medicamento("furosemida", "diuretico", "mg", "IV", (20, 40), 80, 240, (12, 24), passo=10),
    Medicamento("metoprolol", "betabloqueador", "mg", "VO", (25, 50), 200, 400, (12,), desmame_acima_diario=50,
                passo=12.5),
    Medicamento("prednisona", "corticoide", "mg", "VO", (20, 60), 80, 80, (24,), desmame_acima_diario=20, passo=5),
    Medicamento("hidrocortisona", "corticoide", "mg", "IV", (50, 50), 100, 400, (6, 8), desmame_acima_diario=100,
                passo=25),
    Medicamento("clonazepam", "benzodiazepinico", "mg", "VO", (0.5, 1), 4, 8, (12, 24), desmame_acima_diario=1,
                passo=0.25),
    Medicamento("midazolam", "benzodiazepinico", "mg", "IV", (1, 3), 10, 60, (4, 6), True, desmame_acima_diario=10,
                passo=0.5),
    Medicamento("omeprazol", "ibp", "mg", "IV", (40, 40), 80, 160, (24,), variacao_tipica=0, passo=20),
    Medicamento("pantoprazol", "ibp", "mg", "IV", (40, 40), 80, 160, (24,), variacao_tipica=0, passo=20),
)}

# Classes em que dois itens ativos ao mesmo tempo são duplicidade terapêutica.
# (Antibióticos combinados e dipirona + paracetamol são prática comum: não entram.)
CLASSES_SEM_DUPLICIDADE = {"anticoagulante", "opioide", "benzodiazepinico", "corticoide", "betabloqueador", "ibp"}

# Combinações plausíveis de admissão (perfil clínico → medicamentos)
PERFIS = {
    "clinico": ["enoxaparina", "dipirona", "omeprazol"],
    "sepse": ["enoxaparina", "piperacilina-tazobactam", "vancomicina", "dipirona", "pantoprazol", "hidrocortisona"],
    "pos_operatorio": ["heparina", "morfina", "dipirona", "omeprazol", "ceftriaxona"],
    "cardiaco": ["enoxaparina", "metoprolol", "furosemida", "paracetamol", "omeprazol"],
    "diabetico": ["enoxaparina", "insulina regular", "dipirona", "omeprazol", "ceftriaxona"],
    "asma_grave": ["enoxaparina", "prednisona", "paracetamol", "omeprazol", "clonazepam"],
}

TIPOS_ANOMALIA = ("erro_decimal", "dose_acima_maxima", "salto_dose", "duplicidade", "suspensao_abrupta",
                  "frequencia_inesperada", "ajuste_atipico")


@dataclass(frozen=True)
class EventoPrescricao:
    paciente_id: str
    momento: datetime
    medicamento: str
    acao: str                  # iniciar | ajustar | suspender | alta
    dose: float
    intervalo_h: int
    prescritor: str = "CRM-SP 000.000"
    anomalia: str | None = None   # rótulo (só nos dados sintéticos)

    @property
    def med(self) -> Medicamento:
        return FORMULARIO[self.medicamento]

    @property
    def dose_diaria(self) -> float:
        return self.dose * 24 / self.intervalo_h

    def como_prontuario(self) -> dict:
        """Formato da tabela `medicacoes` da Fase 3."""
        m = self.med
        dose = f"{self.dose:g} {m.unidade}".replace(".", ",")
        return {"medicamento": self.medicamento, "dose": dose, "via": m.via,
                "frequencia": f"{self.intervalo_h}/{self.intervalo_h}h"}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["momento"] = self.momento.isoformat()
        return d


def _arredondar(dose: float, med: Medicamento) -> float:
    return max(med.passo, round(dose / med.passo) * med.passo)


def gerar_internacao(paciente_id: str, rng: np.random.Generator, inicio: datetime | None = None,
                     dias: int | None = None) -> list[EventoPrescricao]:
    """Evolução normal de uma internação (sem anomalias)."""
    inicio = inicio or datetime(2026, 3, 1, 10, 0)
    dias = dias or int(rng.integers(4, 11))
    perfil = PERFIS[rng.choice(list(PERFIS))]
    eventos: list[EventoPrescricao] = []
    ativos: dict[str, EventoPrescricao] = {}

    def emitir(nome, acao, dose, intervalo, quando):
        ev = EventoPrescricao(paciente_id, quando, nome, acao, dose, intervalo)
        eventos.append(ev)
        if acao in ("suspender", "alta"):
            ativos.pop(nome, None)
        else:
            ativos[nome] = ev

    for nome in perfil:
        m = FORMULARIO[nome]
        dose = _arredondar(rng.uniform(*m.dose_usual), m)
        emitir(nome, "iniciar", dose, int(rng.choice(m.intervalos_h)), inicio + timedelta(minutes=int(rng.integers(0, 90))))

    curso_atb = int(rng.integers(7, 11))
    for dia in range(1, dias):
        revisao = inicio.replace(hour=8, minute=0) + timedelta(days=dia, minutes=int(rng.integers(0, 120)))
        for nome, atual in list(ativos.items()):
            m = atual.med
            # fim de curso de antibiótico
            if m.classe == "antibiotico" and dia >= curso_atb:
                emitir(nome, "suspender", atual.dose, atual.intervalo_h, revisao)
                continue
            # desmame de corticoide/benzodiazepínico a partir do 3º dia
            if m.desmame_acima_diario is not None and m.classe in ("corticoide", "benzodiazepinico") and dia >= 3:
                if atual.dose_diaria <= m.desmame_acima_diario * 0.6:
                    emitir(nome, "suspender", atual.dose, atual.intervalo_h, revisao)
                elif rng.random() < 0.6:
                    nova = _arredondar(atual.dose * rng.uniform(0.5, 0.75), m)
                    if nova < atual.dose:
                        emitir(nome, "ajustar", nova, atual.intervalo_h, revisao)
                    else:                          # já na menor dose da apresentação: encerra o desmame
                        emitir(nome, "suspender", atual.dose, atual.intervalo_h, revisao)
                continue
            # analgesia opioide reduz e para com a melhora
            if m.classe == "opioide" and dia >= 3 and rng.random() < 0.35:
                emitir(nome, "suspender", atual.dose, atual.intervalo_h, revisao)
                continue
            # escalonamento terapêutico legítimo (raro): profilaxia → dose plena
            if nome == "enoxaparina" and atual.dose == 40 and rng.random() < 0.015:
                emitir(nome, "ajustar", 80.0, 12, revisao)
                continue
            # ajuste rotineiro (medicamentos de dose fixa não são titulados)
            chance = 0.45 if m.classe == "insulina" else 0.12
            if m.variacao_tipica > 0 and rng.random() < chance:
                fator = float(np.exp(rng.normal(0, m.variacao_tipica / 2)))
                nova = min(_arredondar(atual.dose * fator, m), m.dose_max)
                if nova != atual.dose:
                    emitir(nome, "ajustar", nova, atual.intervalo_h, revisao)
    alta = inicio.replace(hour=11, minute=0) + timedelta(days=dias)
    for nome, atual in list(ativos.items()):
        emitir(nome, "alta", atual.dose, atual.intervalo_h, alta)
    return sorted(eventos, key=lambda e: e.momento)


def injetar(eventos: list[EventoPrescricao], tipo: str, rng: np.random.Generator) -> list[EventoPrescricao] | None:
    """Insere uma anomalia rotulada numa internação; `None` se o tipo não cabe nela."""
    pid = eventos[0].paciente_id
    meio = [e for e in eventos if e.acao in ("iniciar", "ajustar") and e.momento < eventos[-1].momento - timedelta(days=1)]
    if not meio:
        return None
    novos = list(eventos)

    def apos(ev):
        return ev.momento + timedelta(hours=int(rng.integers(2, 20)))

    if tipo in ("erro_decimal", "dose_acima_maxima", "salto_dose", "frequencia_inesperada"):
        base = meio[int(rng.integers(len(meio)))]
        m = base.med
        if tipo == "erro_decimal":
            dose, intervalo = base.dose * 10, base.intervalo_h
        elif tipo == "dose_acima_maxima":
            dose, intervalo = _arredondar(m.dose_max * rng.uniform(1.3, 2.0), m), base.intervalo_h
        elif tipo == "salto_dose":
            dose, intervalo = _arredondar(base.dose * rng.uniform(2.2, 3.5), m), base.intervalo_h
            if dose > m.dose_max or dose * 24 / intervalo > m.max_diario:
                return None             # viraria "acima da máxima": deixa para o outro tipo
        else:
            menores = [h for h in (4, 6, 8, 12) if h <= base.intervalo_h / 3]
            if not menores:
                return None
            dose, intervalo = base.dose, max(menores)
        novos.append(replace(base, momento=apos(base), acao="ajustar", dose=dose, intervalo_h=intervalo, anomalia=tipo))
    elif tipo == "ajuste_atipico":
        # aumento de 1,6–1,9× — abaixo do limiar de "salto" das regras — num medicamento titulado com parcimônia
        candidatos = [e for e in meio if 0 < e.med.variacao_tipica < 0.35]
        if not candidatos:
            return None
        base = candidatos[int(rng.integers(len(candidatos)))]
        m = base.med
        dose = _arredondar(base.dose * rng.uniform(1.6, 1.9), m)
        razao = dose / base.dose
        if not 1.5 <= razao < 2.0 or dose > m.dose_max or dose * 24 / base.intervalo_h > m.max_diario:
            return None
        novos.append(replace(base, momento=apos(base), acao="ajustar", dose=dose, anomalia=tipo))
    elif tipo == "duplicidade":
        candidatos = [e for e in meio if e.med.classe in CLASSES_SEM_DUPLICIDADE]
        if not candidatos:
            return None
        base = candidatos[int(rng.integers(len(candidatos)))]
        irmaos = [m for m in FORMULARIO.values() if m.classe == base.med.classe and m.nome != base.medicamento]
        if not irmaos:
            return None
        outro = irmaos[int(rng.integers(len(irmaos)))]
        novos.append(EventoPrescricao(pid, apos(base), outro.nome, "iniciar",
                                      _arredondar(np.mean(outro.dose_usual), outro), outro.intervalos_h[0],
                                      anomalia=tipo))
    elif tipo == "suspensao_abrupta":
        candidatos = [e for e in meio if e.med.desmame_acima_diario is not None and e.acao == "iniciar"]
        if not candidatos:
            return None
        base = candidatos[0]
        m = base.med
        dose = max(base.dose, _arredondar(m.desmame_acima_diario * base.intervalo_h / 24 * 2, m))
        alto = replace(base, momento=base.momento + timedelta(minutes=5), acao="ajustar", dose=min(dose, m.dose_max))
        corte = replace(base, momento=base.momento + timedelta(hours=26), acao="suspender", dose=alto.dose,
                        anomalia=tipo)
        # o curso normal desse medicamento depois do corte é removido
        novos = [e for e in novos if not (e.medicamento == base.medicamento and e.momento > base.momento)]
        novos += [alto, corte]
    else:
        raise ValueError(tipo)
    return sorted(novos, key=lambda e: e.momento)


def gerar_coorte(n: int, semente: int, fracao_anomala: float = 0.0,
                 prefixo: str = "PRT-2026-3") -> list[list[EventoPrescricao]]:
    rng = np.random.default_rng(semente)
    coorte = []
    for i in range(n):
        pid = f"{prefixo}{i:05d}"
        inicio = datetime(2026, 3, 1, 10, 0) + timedelta(days=int(rng.integers(0, 60)))
        eventos = gerar_internacao(pid, rng, inicio)
        if rng.random() < fracao_anomala:
            for _ in range(6):                       # tenta tipos até um caber na internação
                comAnomalia = injetar(eventos, TIPOS_ANOMALIA[int(rng.integers(len(TIPOS_ANOMALIA)))], rng)
                if comAnomalia:
                    eventos = comAnomalia
                    break
        coorte.append(eventos)
    return coorte
