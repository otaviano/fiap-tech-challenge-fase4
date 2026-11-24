"""Roteiro da demonstração integrada: o plantão com todas as modalidades.

O simulador reproduz os leitos de UTI (monitor) e, em horários fixos do
plantão, dispara eventos dos outros canais — exatamente o que o vídeo de
entrega precisa mostrar, de forma reproduzível:

| Horário | Leito | Canal | Evento | Alerta esperado |
|---|---|---|---|---|
| 07:20 | UTI-03 | monitor | deterioração combinada (injetada) | moderado → alto |
| 07:45 | UTI-02 | monitor | dessaturação (injetada) | moderado → alto |
| 07:52 | UTI-02 | consulta | áudio: fala ofegante, "falta de ar" (Azure Speech + Language) | **crítico** — consulta + monitor |
| 08:00 | UTI-05 | prontuário | morfina 4 mg → **40 mg** (erro decimal) | alto — prescritor e farmácia |
| 08:10 | UTI-04 | monitor | taquicardia (injetada) | moderado → alto |
| 08:25 | UTI-05 | monitor | depressão respiratória: FR 7, SpO2 88 (injetada) | **crítico** — monitor + prontuário |
| 08:45 | UTI-03 | consulta | áudio: fadiga vocal, "cansada", "febre" | alto |
| 09:00 | UTI-06 | consulta | áudio: "estou bem, não tenho falta de ar…" (negações) | nenhum achado |
| 09:20 | UTI-04 | câmera | sessão de fisioterapia (REHAB24-6 PM_016) | moderado — fisioterapia |

A UTI-05 foi escolhida para o erro de morfina porque o paciente real desse
leito não tem eventos na janela: o crítico das 08:25 só pode vir da
combinação prescrição + depressão respiratória. Na UTI-06, o paciente diz que
está bem às 09:00 e o monitor mostra NEWS2 ≥ 5 às 09:13 — o relato subjetivo
não substitui o dado objetivo.

Enquanto um evento é analisado (o áudio chama o Azure e leva segundos de
verdade), o plantão **pausa**: os achados recebem o horário do disparo e se
alinham com o que o monitor mostrava naquele minuto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from ..config import ANOMALIAS_RESULTS, VIDEO_RESULTS
from ..domain import Achado, Modalidade
from .simulador import Leito

Produtor = Callable[[str, datetime], tuple[list[Achado], dict]]


@dataclass(frozen=True)
class EventoDemo:
    horario: str          # "HH:MM" no relógio do plantão
    leito: str
    paciente_id: str
    canal: str            # consulta | prontuario | camera
    titulo: str
    produtor: Produtor = field(compare=False)

    def minutos_apos_inicio(self, inicio_plantao: str = "07:00") -> int:
        h, m = map(int, self.horario.split(":"))
        h0, m0 = map(int, inicio_plantao.split(":"))
        return (h * 60 + m) - (h0 * 60 + m0)


# Mesmos pacientes do plantão padrão; a UTI-05 ganha a depressão respiratória que
# conversa com o erro de prescrição de morfina.
LEITOS_DEMO: tuple[Leito, ...] = (
    Leito("UTI-01", "PRT-2026-200101", "p000020-2183-04-28-17-47n"),
    Leito("UTI-02", "PRT-2026-200102", "p000177-2125-11-29-12-35n", eventos=(("dessaturacao", 225, 25),)),
    Leito("UTI-03", "PRT-2026-200103", "p000217-2126-09-30-12-28n", eventos=(("deterioracao_combinada", 200, 70),)),
    Leito("UTI-04", "PRT-2026-200104", "p000263-2160-01-08-23-52n", eventos=(("taquicardia", 250, 30),)),
    Leito("UTI-05", "PRT-2026-200105", "p000123-2161-10-29-03-01n", eventos=(("depressao_respiratoria", 265, 40),)),
    Leito("UTI-06", "PRT-2026-200106", "p000214-2188-10-24-16-23n"),
)


# ------------------------------------------------------------------ produtores

def consulta(paciente_arquivo: str) -> Produtor:
    """Analisa a consulta gravada do paciente (gera as amostras se ainda não existirem)."""

    def produzir(paciente_id: str, momento: datetime) -> tuple[list[Achado], dict]:
        from ..audio import amostras
        from ..audio.consulta import analisar_consulta

        atual = amostras.DESTINO_PADRAO / f"{paciente_arquivo}_atual.wav"
        basal = amostras.DESTINO_PADRAO / f"{paciente_arquivo}_basal.wav"
        if not (atual.exists() and basal.exists()):
            amostras.gerar_padrao()
        rel = analisar_consulta(atual, paciente_id, basal=basal, momento=momento)
        texto = (rel.transcricao or {}).get("texto", "")
        return rel.achados, {"servicos": rel.servicos, "transcricao": texto[:240],
                             "negados": sorted({t["termo"] for t in rel.termos if t["negado"]}), "erros": rel.erros}

    return produzir


def prescricao_morfina(paciente_id: str, momento: datetime) -> tuple[list[Achado], dict]:
    """Morfina 4 mg de 4/4h desde as 06:00; às 08:00, alguém prescreve 40 mg."""
    from ..anomalias.prescricoes import ModeloVariacao, MonitorPrescricoes
    from ..dados.prescricoes import EventoPrescricao

    caminho = ANOMALIAS_RESULTS / "modelo_variacao_prescricoes.json"
    monitor = MonitorPrescricoes(ModeloVariacao.carregar(caminho) if caminho.exists() else None)
    inicio = EventoPrescricao(paciente_id, momento - timedelta(hours=2), "morfina", "iniciar", 4, 4, "CRM-SP 222.333")
    erro = EventoPrescricao(paciente_id, momento, "morfina", "ajustar", 40, 4, "CRM-SP 222.333")
    monitor.processar(inicio)
    achados = monitor.processar(erro)
    return achados, {"prescricao": erro.como_prontuario(), "anterior": inicio.como_prontuario(),
                     "prescritor": erro.prescritor}


def sessao_fisioterapia(sessao: str) -> Produtor:
    """Usa o relatório já gerado da sessão (`med-monitor video relatorio`) e reatribui ao paciente."""

    def produzir(paciente_id: str, momento: datetime) -> tuple[list[Achado], dict]:
        caminho = VIDEO_RESULTS / f"sessao_{sessao}.json"
        if not caminho.exists():
            return [], {"erros": [f"relatório {caminho.name} ausente: rode `med-monitor video relatorio {sessao}`"]}
        rel = json.loads(caminho.read_text(encoding="utf-8"))
        achados = [Achado(paciente_id, Modalidade(a["modalidade"]), a["tipo"], a["escore"], a["fonte"],
                          evidencia=a["evidencia"], momento=momento) for a in rel.get("achados", [])]
        incorretas = [r["repeticao"] for r in rel["repeticoes"] if r["prob_incorreta"] >= 0.5]
        return achados, {"sessao": sessao, "repeticoes": len(rel["repeticoes"]), "previstas_incorretas": incorretas}

    return produzir


ROTEIRO_DEMO: tuple[EventoDemo, ...] = (
    EventoDemo("07:52", "UTI-02", "PRT-2026-200102", "consulta", "Consulta à beira do leito (áudio)",
               consulta("PRT-2026-200102")),
    EventoDemo("08:00", "UTI-05", "PRT-2026-200105", "prontuario", "Nova prescrição registrada",
               prescricao_morfina),
    EventoDemo("08:45", "UTI-03", "PRT-2026-200103", "consulta", "Consulta à beira do leito (áudio)",
               consulta("PRT-2026-200103")),
    # a gravação "estável" foi gerada para o PRT-2026-200105; aqui é o relato do paciente da UTI-06
    EventoDemo("09:00", "UTI-06", "PRT-2026-200106", "consulta", "Consulta à beira do leito (áudio)",
               consulta("PRT-2026-200105")),
    EventoDemo("09:20", "UTI-04", "PRT-2026-200104", "camera", "Sessão de fisioterapia gravada (vídeo)",
               sessao_fisioterapia("PM_016")),
)
