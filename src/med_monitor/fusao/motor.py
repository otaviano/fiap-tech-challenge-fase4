"""Motor de fusão: achados de qualquer modalidade → alertas para a equipe médica.

Fusão tardia (decisão D2): cada analisador entrega `Achado`s independentes; aqui
eles são agrupados por **episódio** de cada paciente e convertidos em
`Alerta`s por regras explícitas.

Como a severidade de um episódio é decidida:

1. **Severidade base** de cada achado — tabela `SEVERIDADE_BASE`. Para os
   sinais vitais, segue a resposta clínica do NEWS2: um parâmetro isolado com 3
   pontos pede avaliação urgente à beira do leito (*moderado*), escore ≥ 5
   pede resposta urgente (*alto*), ≥ 7 é emergência (*crítico*).
2. **Corroboração**: dois ou mais detectores *diferentes* apontando o mesmo
   paciente com achados **vigentes** (ver `VALIDADE`: 15 min para o monitor,
   1 h para consulta e vídeo, 6 h para prescrição) elevam um nível, **até no máximo
   alto**. Detectores da mesma modalidade leem o mesmo sensor: um oxímetro
   reposicionado engana os três ao mesmo tempo, então a concordância deles não
   é evidência independente o bastante para declarar emergência.
3. **Concordância entre canais**: achados de **canais físicos** diferentes
   (ex.: dessaturação no monitor + fala ofegante na consulta) elevam mais um
   nível, sem teto — aí sim são fontes independentes. O canal é a origem física
   do dado (`CANAL`): áudio e texto da mesma consulta saem da mesma gravação, e
   por isso se *corroboram* (teto alto) mas não contam como independentes.

*Crítico*, portanto, só sai de NEWS2 ≥ 7, de risco de autolesão relatado, ou de
canais físicos que concordam.

Um episódio começa no primeiro achado e termina quando o paciente passa
`janela_episodio` sem achados novos. Dentro do episódio, só sai alerta novo se
a severidade **subir** — escalonamento, não repetição. É o que impede o painel
de virar ruído que a equipe aprende a ignorar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..domain import Achado, Alerta, Modalidade, Severidade

S = Severidade

# (fonte, tipo) → severidade base. `None` no tipo vale para qualquer tipo da fonte.
SEVERIDADE_BASE: dict[tuple[str, str | None], Severidade] = {
    ("news2", "news2_alto"): S.CRITICO,
    ("news2", "news2_medio"): S.ALTO,
    ("news2", None): S.MODERADO,                 # parâmetro isolado com 3 pontos
    ("isolation-forest", None): S.MODERADO,      # deterioração multiparamétrica
    ("yolov8-pose", "queda"): S.ALTO,            # paciente no chão: avaliação médica imediata
}
# Teto da severidade base por fonte: desvio do basal, sozinho, pede olhar — não chamar o plantonista.
TETO_BASE = {"desvio-robusto": S.MODERADO}
TETO_CORROBORACAO = S.ALTO

# Por quanto tempo um achado continua valendo como evidência do estado do paciente.
# Um valor do monitor envelhece em minutos; o que o paciente relatou na consulta,
# em cerca de uma hora; uma prescrição perigosa é risco enquanto o medicamento age.
VALIDADE = {
    Modalidade.PRESCRICAO: timedelta(hours=6),
    Modalidade.AUDIO: timedelta(hours=1), Modalidade.TEXTO: timedelta(hours=1),
    Modalidade.VIDEO: timedelta(hours=1), Modalidade.MOVIMENTACAO: timedelta(hours=1),
}

# Canal físico de cada modalidade: é a independência *da origem* que importa.
CANAL = {
    Modalidade.SINAIS_VITAIS: "monitor",
    Modalidade.AUDIO: "consulta", Modalidade.TEXTO: "consulta",
    Modalidade.VIDEO: "camera", Modalidade.MOVIMENTACAO: "camera",
    Modalidade.PRESCRICAO: "prontuario",
}

PREFIXO = {
    Modalidade.SINAIS_VITAIS: "SV", Modalidade.AUDIO: "AU", Modalidade.TEXTO: "TX",
    Modalidade.VIDEO: "VD", Modalidade.MOVIMENTACAO: "MV", Modalidade.PRESCRICAO: "RX",
}

# Achados de reabilitação vão para a fisioterapia enquanto não houver risco clínico maior.
DESTINATARIO_POR_TIPO = {"desvio_execucao": "fisioterapia", "amplitude_reduzida": "fisioterapia"}

DESTINATARIO = {
    S.CRITICO: "time de resposta rápida",
    S.ALTO: "médico plantonista",
    S.MODERADO: "enfermagem do leito",
    S.INFORMATIVO: "enfermagem do leito",
}

ROTULO = {
    "taquicardia": "Taquicardia", "bradicardia": "Bradicardia",
    "hipotensao": "Hipotensão", "hipertensao": "Hipertensão",
    "taquipneia": "Taquipneia", "bradipneia": "Bradipneia", "dessaturacao": "Dessaturação",
    "news2_medio": "NEWS2 ≥ 5", "news2_alto": "NEWS2 ≥ 7",
    "padrao_multivariado": "Deterioração multiparamétrica",
    "sensor_sem_sinal": "Sensor sem sinal",
    # áudio e vídeo (entram pela API /achados até os analisadores ficarem prontos)
    "fala_ofegante": "Fala ofegante", "fadiga_vocal": "Fadiga vocal", "disartria": "Disartria",
    "termo_critico": "Termo crítico na consulta", "sentimento_negativo": "Sentimento negativo",
    "sintoma_relatado": "Sintoma relatado",
    "queda": "Queda", "desvio_postural": "Desvio postural", "amplitude_reduzida": "Amplitude de movimento reduzida",
    "desvio_execucao": "Execução incorreta na fisioterapia",
    "possivel_erro_decimal": "Possível erro decimal na prescrição", "dose_acima_maxima": "Dose acima da máxima",
    "duplicidade_terapeutica": "Duplicidade terapêutica", "suspensao_abrupta": "Suspensão abrupta",
    "intervalo_inesperado": "Intervalo de dose inesperado", "salto_de_dose": "Salto de dose",
    "alteracao_atipica": "Ajuste de dose atípico para o medicamento",
}
UNIDADE = {"fc": "bpm", "pas": "mmHg", "pad": "mmHg", "pam": "mmHg", "fr": "irpm", "spo2": "%"}
NOME_SINAL = {"fc": "FC", "pas": "PAS", "pad": "PAD", "pam": "PAM", "fr": "FR", "spo2": "SpO2"}


def _destinatario(principal: Achado, severidade: Severidade) -> str:
    if principal.modalidade is Modalidade.PRESCRICAO and severidade is not S.CRITICO:
        # a prescrição ainda não foi administrada: quem age é a farmácia e, se grave, quem prescreveu
        return "farmácia clínica" if severidade.peso <= S.MODERADO.peso else "médico prescritor e farmácia clínica"
    if severidade.peso <= S.MODERADO.peso and principal.tipo in DESTINATARIO_POR_TIPO:
        return DESTINATARIO_POR_TIPO[principal.tipo]
    return DESTINATARIO[severidade]


def severidade_base(achado: Achado) -> Severidade:
    # relato de risco de autolesão não espera corroboração
    if achado.evidencia.get("categoria") == "risco_autolesao":
        return S.CRITICO
    for chave in ((achado.fonte, achado.tipo), (achado.fonte, None)):
        if chave in SEVERIDADE_BASE:
            return SEVERIDADE_BASE[chave]
    # padrão para fontes sem tabela própria (desvio robusto, áudio, vídeo...): pelo escore
    if achado.escore >= 0.8:
        sev = S.ALTO
    elif achado.escore >= 0.5:
        sev = S.MODERADO
    else:
        sev = S.INFORMATIVO
    teto = TETO_BASE.get(achado.fonte)
    return teto if teto and sev.peso > teto.peso else sev


def _num(valor) -> str:
    return f"{valor:g}".replace(".", ",")


def _medida(sinal: str, valor) -> str:
    unidade = UNIDADE.get(sinal, "")
    return f"{_num(valor)}{unidade}" if unidade == "%" else f"{_num(valor)} {unidade}".strip()


def descrever(achado: Achado) -> str:
    """Frase curta e concreta para o painel: o que, quanto e comparado a quê."""
    ev = achado.evidencia
    sinal = ev.get("sinal")
    if achado.tipo.endswith("_do_basal") and sinal:
        direcao = "acima" if "_acima_" in achado.tipo else "abaixo"
        return (f"{NOME_SINAL.get(sinal, sinal)} {_medida(sinal, ev['valor'])} "
                f"{direcao} do basal do paciente ({_num(ev['basal'])})")
    rotulo = ROTULO.get(achado.tipo, achado.tipo.replace("_", " ").capitalize())
    if achado.tipo == "padrao_multivariado":
        desvios = ", ".join(f"{NOME_SINAL.get(s, s)} {_num(round(z, 1))}σ" for s, z in ev.get("principais_desvios", {}).items())
        return f"{rotulo} ({desvios})"
    if achado.tipo == "termo_critico":
        termos = ", ".join(f"“{t}”" for t in ev.get("termos", []))
        return f"Paciente relata {termos} ({ev.get('categoria', '').replace('_', ' ')})"
    if achado.tipo == "sintoma_relatado":
        return f"Sintomas citados na consulta: {', '.join(ev.get('sintomas', []))}"
    if achado.tipo in ("fala_ofegante", "fadiga_vocal"):
        criterios = "; ".join(ev.get("criterios_atendidos", []))
        return f"{rotulo} na consulta ({criterios})" if criterios else rotulo
    if achado.tipo == "sentimento_negativo":
        return f"{rotulo} na consulta (confiança {_num(round(ev.get('confianca', 0), 2))})"
    if achado.modalidade is Modalidade.PRESCRICAO and "prescricao" in ev:
        p = ev["prescricao"]
        base = f"{rotulo}: {p['medicamento']} {p['dose']} {p['via']} {p['frequencia']}"
        if achado.tipo == "possivel_erro_decimal":
            return base + f" ({_num(ev['razao'])}× a dose de referência de {_num(ev['dose_referencia'])})"
        if achado.tipo == "dose_acima_maxima":
            return base + f" ({'; '.join(ev.get('motivos', []))})"
        if achado.tipo == "duplicidade_terapeutica":
            return base + f" — já ativo: {', '.join(ev.get('ja_ativo', []))}"
        if achado.tipo == "suspensao_abrupta":
            return base + f" sem desmame (dose diária {_num(ev['dose_diaria_ao_suspender'])})"
        if achado.tipo == "intervalo_inesperado":
            return base + f" (antes {ev['intervalo_anterior_h']}/{ev['intervalo_anterior_h']}h)"
        if achado.tipo in ("salto_de_dose", "alteracao_atipica"):
            extra = f", variação típica {ev['variacao_tipica']}" if "variacao_tipica" in ev else ""
            return base + f" ({_num(ev['razao_dose_diaria'])}× a dose diária anterior{extra})"
        return base
    if achado.tipo == "desvio_execucao":
        return (f"{rotulo} ({ev.get('exercicio')}): {len(ev.get('repeticoes_incorretas', []))} de "
                f"{ev.get('total_repeticoes')} repetições" + (f" — {'; '.join(ev['desvios'][:2])}" if ev.get("desvios") else ""))
    if achado.tipo == "amplitude_reduzida":
        return f"{rotulo}: {_num(ev.get('atual'))}° vs {_num(ev.get('anterior'))}° na sessão anterior"
    if achado.tipo == "sensor_sem_sinal":
        return f"{rotulo}: {NOME_SINAL.get(sinal, sinal)} há {ev.get('minutos')} min"
    if sinal and "valor" in ev:
        return f"{rotulo} — {NOME_SINAL.get(sinal, sinal)} {_medida(sinal, ev['valor'])}"
    if "news2_parcial" in ev:
        return f"{rotulo} (NEWS2 parcial {ev['news2_parcial']})"
    return rotulo


def _com_fuso(achado: Achado) -> Achado:
    """Achados sem fuso (séries offline) são tratados como UTC, para comparar com os de outras fontes."""
    if achado.momento.tzinfo is not None:
        return achado
    from dataclasses import replace

    return replace(achado, momento=achado.momento.replace(tzinfo=timezone.utc))


@dataclass
class _Episodio:
    inicio: datetime
    ultimo: datetime
    achados: list[Achado] = field(default_factory=list)
    severidade_emitida: Severidade | None = None


@dataclass(frozen=True)
class Decisao:
    """Alerta emitido + achados que o fundamentam (para auditoria e painel)."""

    alerta: Alerta
    achados: tuple[Achado, ...]


class MotorFusao:
    def __init__(self, janela_episodio: timedelta = timedelta(minutes=30),
                 janela_corroboracao: timedelta = timedelta(minutes=15),
                 severidade_minima: Severidade = S.MODERADO):
        self.janela_episodio = janela_episodio
        self.janela_corroboracao = janela_corroboracao
        self.severidade_minima = severidade_minima
        self._episodios: dict[str, _Episodio] = {}

    def episodio_aberto(self, paciente_id: str, agora: datetime | None = None) -> dict | None:
        """Episódio em curso; com `agora`, um episódio sem achados há mais de `janela_episodio` já acabou."""
        ep = self._episodios.get(paciente_id)
        if not ep:
            return None
        if agora is not None:
            agora = agora if agora.tzinfo else agora.replace(tzinfo=timezone.utc)
            if agora - ep.ultimo > self.janela_episodio and not any(self._vigente(a, agora) for a in ep.achados):
                return None
        return {"inicio": ep.inicio.isoformat(), "ultimo": ep.ultimo.isoformat(),
                "achados": len(ep.achados),
                "severidade": ep.severidade_emitida.value if ep.severidade_emitida else None}

    def _vigente(self, a: Achado, agora: datetime) -> bool:
        """O achado ainda vale como evidência em `agora` (validade da modalidade, no mínimo a janela de corroboração)."""
        idade = agora - a.momento
        return timedelta(0) <= idade <= max(self.janela_corroboracao, VALIDADE.get(a.modalidade, timedelta(0)))

    def processar(self, achado: Achado) -> list[Decisao]:
        achado = _com_fuso(achado)
        if achado.evidencia.get("tecnico"):
            return [self._alerta_tecnico(achado)]

        ep = self._episodios.get(achado.paciente_id)
        continua = ep is not None and (achado.momento - ep.ultimo <= self.janela_episodio
                                       or any(self._vigente(a, achado.momento) for a in ep.achados))
        if not continua:
            ep = self._episodios[achado.paciente_id] = _Episodio(achado.momento, achado.momento)
        ep.achados.append(achado)
        ep.ultimo = max(ep.ultimo, achado.momento)

        recentes = [a for a in ep.achados if self._vigente(a, achado.momento)]
        principal = max(recentes, key=lambda a: (severidade_base(a).peso, a.escore))
        severidade = max((severidade_base(a) for a in ep.achados), key=lambda s: s.peso)
        fontes = {a.fonte for a in recentes}
        modalidades = {a.modalidade for a in recentes}
        canais = {CANAL.get(a.modalidade, a.modalidade.value) for a in recentes}
        motivos = []
        if len(fontes) >= 2 and severidade.peso < TETO_CORROBORACAO.peso:
            severidade = severidade.elevar()
            motivos.append(f"corroborado por {len(fontes)} detectores")
        if len(canais) >= 2:
            severidade = severidade.elevar()
            motivos.append("concordância entre " + " e ".join(sorted(canais)))

        if severidade.peso < self.severidade_minima.peso:
            return []
        if ep.severidade_emitida is not None and severidade.peso <= ep.severidade_emitida.peso:
            return []
        escalonou = ep.severidade_emitida is not None
        ep.severidade_emitida = severidade

        demais = [descrever(a) for a in recentes if a is not principal]
        partes = [descrever(principal)]
        if demais:
            partes.append("Também: " + "; ".join(dict.fromkeys(demais)))
        if motivos:
            partes.append("Severidade elevada: " + ", ".join(motivos))
        if escalonou:
            partes.insert(0, "ESCALONAMENTO")
        mods = tuple(sorted(modalidades, key=lambda m: m.value))
        prefixo = "MM" if len(canais) > 1 else PREFIXO.get(principal.modalidade, "XX")
        alerta = Alerta(
            paciente_id=achado.paciente_id,
            codigo=f"{prefixo}-{principal.tipo.upper()}",
            severidade=severidade,
            mensagem=" · ".join(partes),
            achados=tuple(a.id for a in recentes),
            modalidades=mods,
            destinatario=_destinatario(principal, severidade),
            momento=achado.momento,
        )
        return [Decisao(alerta, tuple(recentes))]

    def processar_varios(self, achados: list[Achado]) -> list[Decisao]:
        return [d for a in achados for d in self.processar(a)]

    @staticmethod
    def _alerta_tecnico(achado: Achado) -> Decisao:
        alerta = Alerta(
            paciente_id=achado.paciente_id, codigo="TEC-SENSOR_SEM_SINAL",
            severidade=S.INFORMATIVO, mensagem=descrever(achado) + " — monitoramento cego",
            achados=(achado.id,), modalidades=(achado.modalidade,),
            destinatario="enfermagem do leito", momento=achado.momento)
        return Decisao(alerta, (achado,))
