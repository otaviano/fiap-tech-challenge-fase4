import pandas as pd

from med_monitor.anomalias import avaliacao
from med_monitor.dados import physionet, sintetico


def test_avaliar_registro_em_fundo_sintetico():
    fundo = sintetico.gerar_basal(horas=24, semente=3)
    res = avaliacao.avaliar_registro(fundo, "sint", semente=3)
    assert len(res.eventos) == len(sintetico.TIPOS_CLINICOS)
    assert all(avaliacao.SISTEMA in e["latencias"] for e in res.eventos)
    assert res.em_artefato[avaliacao.SISTEMA] == 0


def test_tipos_para_serie_curta_gira_com_a_semente():
    a = avaliacao._tipos_para(700, semente=0)
    b = avaliacao._tipos_para(700, semente=1)
    assert 1 <= len(a) < len(sintetico.TIPOS_CLINICOS) and a != b


def test_tabela_markdown_tem_todos_os_detectores():
    fundo = sintetico.gerar_basal(horas=24, semente=3)
    res = avaliacao.avaliar_registro(fundo, "sint", semente=3)
    resumo = avaliacao.consolidar([res], {"sint": {f: 0.0 for f in (*avaliacao.FONTES, avaliacao.SISTEMA)}},
                                  {"sint": len(fundo)})
    md = avaliacao.tabela_markdown(resumo)
    for nome in ("NEWS2", "Desvio robusto", "Isolation Forest", "Sistema"):
        assert nome in md


def test_mapeamento_canonico_usa_nbp_quando_falta_pa_invasiva():
    bruto = pd.DataFrame({"HR": [80.0], "NBPSys": [118.0], "NBPDias": [70.0], "SpO2": [97.0]})
    df = physionet._para_canonico(bruto)
    assert list(df.columns) == list(physionet.SINAIS)
    assert df.loc[0, "pas"] == 118.0 and df.loc[0, "fc"] == 80.0
    assert pd.isna(df.loc[0, "fr"])
