from datetime import datetime, timedelta

import numpy as np
import pytest

from med_monitor.anomalias.prescricoes import CORRESPONDENCIA, ModeloVariacao, MonitorPrescricoes, avaliar
from med_monitor.dados import prescricoes as dados
from med_monitor.dados.prescricoes import FORMULARIO, EventoPrescricao
from med_monitor.domain import Achado, Modalidade, Severidade
from med_monitor.fusao.motor import MotorFusao, descrever

T0 = datetime(2026, 3, 1, 10, 0)


def ev(med, acao, dose, intervalo, horas=0.0, pid="P1"):
    return EventoPrescricao(pid, T0 + timedelta(hours=horas), med, acao, dose, intervalo)


def tipos(monitor, *eventos):
    return [[a.tipo for a in monitor.processar(e)] for e in eventos]


def test_formato_do_prontuario_da_fase3():
    p = ev("clonazepam", "iniciar", 0.5, 12).como_prontuario()
    assert p == {"medicamento": "clonazepam", "dose": "0,5 mg", "via": "VO", "frequencia": "12/12h"}


def test_erro_decimal_e_dose_acima_da_maxima():
    m = MonitorPrescricoes()
    r = tipos(m, ev("morfina", "iniciar", 4, 4), ev("morfina", "ajustar", 40, 4, 6))
    assert r == [[], ["possivel_erro_decimal"]]
    assert tipos(MonitorPrescricoes(), ev("tramadol", "iniciar", 200, 6)) == [["dose_acima_maxima"]]


def test_dose_diaria_acima_da_maxima_pelo_intervalo():
    # 1000 mg de 4/4h = 6000 mg/dia > 4000 mg/dia de dipirona
    assert tipos(MonitorPrescricoes(), ev("dipirona", "iniciar", 1000, 4)) == [["dose_acima_maxima"]]


def test_duplicidade_so_nas_classes_de_risco():
    m = MonitorPrescricoes()
    assert tipos(m, ev("enoxaparina", "iniciar", 40, 24), ev("heparina", "iniciar", 5000, 12, 2)) == \
        [[], ["duplicidade_terapeutica"]]
    m2 = MonitorPrescricoes()        # dois antibióticos juntos é prática comum
    assert tipos(m2, ev("vancomicina", "iniciar", 1000, 12), ev("piperacilina-tazobactam", "iniciar", 4500, 6, 1)) == \
        [[], []]


def test_suspensao_abrupta_exige_desmame_recente():
    m = MonitorPrescricoes()
    assert tipos(m, ev("prednisona", "iniciar", 40, 24), ev("prednisona", "suspender", 40, 24, 30))[1] == \
        ["suspensao_abrupta"]
    desmame = MonitorPrescricoes()
    r = tipos(desmame, ev("prednisona", "iniciar", 40, 24), ev("prednisona", "ajustar", 20, 24, 24),
              ev("prednisona", "suspender", 20, 24, 40))
    assert r[2] == []                                       # houve redução nas últimas 48 h


def test_intervalo_encurtado():
    m = MonitorPrescricoes()
    assert tipos(m, ev("ceftriaxona", "iniciar", 1000, 24), ev("ceftriaxona", "ajustar", 1000, 6, 3))[1] == \
        ["intervalo_inesperado"]


def test_correcao_apos_alerta_nao_gera_novo_alerta():
    m = MonitorPrescricoes()
    r = tipos(m, ev("omeprazol", "iniciar", 40, 24), ev("omeprazol", "ajustar", 400, 24, 2),
              ev("omeprazol", "ajustar", 40, 24, 5))
    assert r == [[], ["possivel_erro_decimal"], []]


def test_variacao_aprendida_pega_o_que_a_regra_nao_pega():
    historico = []
    rng = np.random.default_rng(0)
    for i in range(40):     # clonazepam no histórico: só desmames suaves
        d = 1.0
        evs = [ev("clonazepam", "iniciar", d, 12, pid=f"H{i}")]
        for k in range(1, 3):
            d = d * rng.uniform(0.7, 0.85)
            evs.append(ev("clonazepam", "ajustar", d, 12, 24 * k, pid=f"H{i}"))
        historico.append(evs)
    modelo = ModeloVariacao.ajustar(historico)
    eventos = (ev("clonazepam", "iniciar", 0.5, 12), ev("clonazepam", "ajustar", 0.85, 12, 24))
    assert tipos(MonitorPrescricoes(None), *eventos)[1] == []                         # 1,7× < limiar da regra
    assert tipos(MonitorPrescricoes(modelo), *eventos)[1] == ["alteracao_atipica"]


def test_alta_nao_e_avaliada():
    m = MonitorPrescricoes()
    assert tipos(m, ev("prednisona", "iniciar", 60, 24), ev("prednisona", "alta", 60, 24, 10)) == [[], []]


def test_coorte_normal_quase_sem_alertas_e_anomalias_rotuladas():
    coorte = dados.gerar_coorte(80, 11, 0.0)
    m_alertas = sum(len(MonitorPrescricoes().processar_varios(i)) for i in coorte)
    n = sum(len(i) for i in coorte)
    assert m_alertas / n < 0.03
    anomalas = dados.gerar_coorte(80, 12, 1.0)
    assert sum(any(e.anomalia for e in i) for i in anomalas) >= 70
    for internacao in anomalas:
        assert internacao == sorted(internacao, key=lambda e: e.momento)
        for e in internacao:
            assert e.medicamento in FORMULARIO and e.intervalo_h > 0


def test_avaliacao_pequena_tem_recall_alto():
    res = avaliar(n_treino=120, n_teste=120)
    det = res["detectores"]["regras + variação aprendida"]
    assert det["recall"] > 0.8 and det["taxa_alerta_sem_anomalia"] < 0.05
    assert set(CORRESPONDENCIA) == set(dados.TIPOS_ANOMALIA)


# ------------------------------------------------------------------ fusão

def _presc(tipo, escore, **ev_):
    return Achado("P1", Modalidade.PRESCRICAO, tipo, escore, "regras-formulario", evidencia=ev_, momento=T0)


def test_prescricao_vai_para_a_farmacia_e_ao_prescritor_quando_grave():
    p = {"medicamento": "morfina", "dose": "40 mg", "via": "IV", "frequencia": "4/4h"}
    [d] = MotorFusao().processar(_presc("possivel_erro_decimal", 0.95, prescricao=p, razao=10.0, dose_referencia=4.0))
    assert d.alerta.severidade is Severidade.ALTO
    assert d.alerta.destinatario == "médico prescritor e farmácia clínica"
    assert d.alerta.mensagem.startswith("Possível erro decimal na prescrição: morfina 40 mg IV 4/4h (10× ")
    [d2] = MotorFusao().processar(_presc("duplicidade_terapeutica", 0.75, prescricao=p, ja_ativo=["tramadol"]))
    assert d2.alerta.severidade is Severidade.MODERADO and d2.alerta.destinatario == "farmácia clínica"


def test_erro_de_opioide_mais_bradipneia_no_monitor_e_critico():
    p = {"medicamento": "morfina", "dose": "40 mg", "via": "IV", "frequencia": "4/4h"}
    m = MotorFusao()
    m.processar(_presc("possivel_erro_decimal", 0.95, prescricao=p, razao=10.0, dose_referencia=4.0))
    brad = Achado("P1", Modalidade.SINAIS_VITAIS, "bradipneia", 0.62, "news2",
                  evidencia={"sinal": "fr", "valor": 6.0}, momento=T0 + timedelta(minutes=40))
    [d] = m.processar(brad)
    assert d.alerta.severidade is Severidade.CRITICO
    assert "concordância entre monitor e prontuario" in d.alerta.mensagem
    assert d.alerta.destinatario == "time de resposta rápida"
