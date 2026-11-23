import math

import numpy as np
import pandas as pd
import pytest

from med_monitor.anomalias.sinais_vitais import (
    DetectorDesvioRobusto, FiltroArtefato, MonitorSinaisVitais, _Gatilho, pontuar_news2)
from med_monitor.dados import sintetico
from med_monitor.domain import Modalidade

FONTES_CLINICAS = {"news2", "desvio-robusto", "isolation-forest"}


@pytest.fixture(scope="module")
def serie_com_eventos():
    basal = sintetico.gerar_basal(horas=24, semente=7)
    serie, eventos = sintetico.injetar_varios(
        basal, list(sintetico.TIPOS_CLINICOS) + ["artefato", "artefato"], semente=7)
    achados = MonitorSinaisVitais("PRT-TESTE").processar_serie(serie)
    return serie, eventos, achados


# ------------------------------------------------------------------ NEWS2

@pytest.mark.parametrize("amostra, esperado", [
    ({"fr": 8}, {"fr": 3}), ({"fr": 9}, {"fr": 1}), ({"fr": 20}, {"fr": 0}),
    ({"fr": 24}, {"fr": 2}), ({"fr": 25}, {"fr": 3}),
    ({"spo2": 91}, {"spo2": 3}), ({"spo2": 93}, {"spo2": 2}), ({"spo2": 95}, {"spo2": 1}), ({"spo2": 96}, {"spo2": 0}),
    ({"pas": 90}, {"pas": 3}), ({"pas": 100}, {"pas": 2}), ({"pas": 110}, {"pas": 1}),
    ({"pas": 219}, {"pas": 0}), ({"pas": 220}, {"pas": 3}),
    ({"fc": 40}, {"fc": 3}), ({"fc": 50}, {"fc": 1}), ({"fc": 90}, {"fc": 0}),
    ({"fc": 110}, {"fc": 1}), ({"fc": 130}, {"fc": 2}), ({"fc": 131}, {"fc": 3}),
])
def test_news2_faixas(amostra, esperado):
    assert pontuar_news2(amostra) == esperado


def test_news2_ignora_sinal_ausente():
    assert pontuar_news2({"fc": math.nan, "fr": 16}) == {"fr": 0}


# ---------------------------------------------------------------- gatilho

def test_gatilho_exige_persistencia_e_dispara_uma_vez():
    g = _Gatilho(persistencia=3, rearme=2)
    disparos = [g.atualizar(v) for v in [True, True, True, True, True]]
    assert disparos == [False, False, True, False, False]


def test_gatilho_rearma_so_apos_periodo_sem_condicao():
    g = _Gatilho(persistencia=1, rearme=2)
    assert g.atualizar(True)
    assert not g.atualizar(False)
    assert not g.atualizar(True)        # só 1 amostra sem condição: ainda no mesmo episódio
    g.atualizar(False), g.atualizar(False)
    assert g.atualizar(True)            # novo episódio


def test_gatilho_amostra_sem_dado_nao_conta():
    g = _Gatilho(persistencia=2, rearme=1)
    assert not g.atualizar(True)
    assert not g.atualizar(None)
    assert g.atualizar(True)


# --------------------------------------------------------------- artefato

def test_filtro_troca_implausivel_por_nan():
    f = FiltroArtefato()
    limpa = f.limpar({"spo2": 0.0, "fc": 80.0, "pas": 400.0})
    assert math.isnan(limpa["spo2"]) and math.isnan(limpa["pas"]) and limpa["fc"] == 80.0
    assert f.descartes == 2


def test_filtro_avisa_sensor_mudo_uma_vez():
    f = FiltroArtefato(sem_sinal_min=3, sinais=("spo2",))
    achados = [a for _ in range(10) for a in f.atualizar("P", pd.Timestamp("2026-01-01"), {"spo2": math.nan})]
    assert len(achados) == 1
    assert achados[0].tipo == "sensor_sem_sinal" and achados[0].evidencia["tecnico"] is True


# ---------------------------------------------------------- desvio robusto

def test_desvio_robusto_nao_dispara_antes_do_basal():
    d = DetectorDesvioRobusto(min_amostras=60)
    t0 = pd.Timestamp("2026-01-01")
    achados = [a for i in range(50) for a in d.atualizar("P", t0 + pd.Timedelta(minutes=i), {"fc": 150.0})]
    assert achados == []


def test_desvio_robusto_ignora_variacao_pequena():
    rng = np.random.default_rng(0)
    d = DetectorDesvioRobusto()
    t0 = pd.Timestamp("2026-01-01")
    achados = []
    for i in range(400):   # FC estável em 70 ± 0,5 e depois +8 bpm: z alto, mas abaixo do DELTA_MIN
        fc = 70 + rng.normal(0, 0.5) + (8 if i > 300 else 0)
        achados += d.atualizar("P", t0 + pd.Timedelta(minutes=i), {"fc": fc})
    assert achados == []


# ---------------------------------------------------------------- monitor

def test_monitor_detecta_todos_os_eventos_clinicos(serie_com_eventos):
    _, eventos, achados = serie_com_eventos
    tol = pd.Timedelta(minutes=10)
    clinicos = [a for a in achados if a.fonte in FONTES_CLINICAS]
    for ev in (e for e in eventos if e.clinico):
        assert any(ev.inicio <= pd.Timestamp(a.momento) <= ev.fim + tol for a in clinicos), ev.tipo


def test_monitor_nao_alerta_artefato_nem_fora_de_evento(serie_com_eventos):
    _, eventos, achados = serie_com_eventos
    tol = pd.Timedelta(minutes=10)
    clinicos_janelas = [(e.inicio, e.fim + tol) for e in eventos if e.clinico]
    for a in achados:
        if a.fonte not in FONTES_CLINICAS:
            continue
        m = pd.Timestamp(a.momento)
        assert any(i <= m <= f for i, f in clinicos_janelas), f"alerta espúrio: {a.tipo} em {m}"


def test_achados_sao_de_sinais_vitais_e_explicaveis(serie_com_eventos):
    _, _, achados = serie_com_eventos
    assert achados
    for a in achados:
        assert a.modalidade is Modalidade.SINAIS_VITAIS
        assert 0 <= a.escore <= 1
        assert a.evidencia, "todo achado precisa de evidência"
    multivariados = [a for a in achados if a.fonte == "isolation-forest"]
    assert all("principais_desvios" in a.evidencia for a in multivariados)


def test_injetar_respeita_aquecimento_e_nao_sobrepoe():
    basal = sintetico.gerar_basal(horas=24, semente=1)
    _, eventos = sintetico.injetar_varios(basal, list(sintetico.TIPOS_CLINICOS), semente=1, aquecimento=180)
    assert eventos[0].inicio >= basal.index[180]
    for a, b in zip(eventos, eventos[1:]):
        assert a.fim < b.inicio


def test_injetar_falha_em_serie_curta():
    with pytest.raises(ValueError):
        sintetico.injetar_varios(sintetico.gerar_basal(horas=4), list(sintetico.TIPOS_CLINICOS))
