from datetime import datetime, timedelta, timezone

from med_monitor.domain import Achado, Modalidade, Severidade
from med_monitor.fusao.motor import MotorFusao, descrever, severidade_base

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
SV = Modalidade.SINAIS_VITAIS


def achado(tipo, fonte, minuto=0, escore=0.6, modalidade=SV, paciente="P1", **ev):
    return Achado(paciente, modalidade, tipo, escore, fonte, evidencia=ev, momento=T0 + timedelta(minutes=minuto))


def test_severidade_base_segue_resposta_do_news2():
    assert severidade_base(achado("dessaturacao", "news2")) is Severidade.MODERADO
    assert severidade_base(achado("news2_medio", "news2")) is Severidade.ALTO
    assert severidade_base(achado("news2_alto", "news2")) is Severidade.CRITICO
    assert severidade_base(achado("padrao_multivariado", "isolation-forest")) is Severidade.MODERADO


def test_desvio_do_basal_sozinho_nao_passa_de_moderado():
    assert severidade_base(achado("fc_acima_do_basal", "desvio-robusto", escore=1.0)) is Severidade.MODERADO
    assert severidade_base(achado("fc_acima_do_basal", "desvio-robusto", escore=0.3)) is Severidade.INFORMATIVO


def test_parametro_vermelho_gera_alerta_moderado_para_enfermagem():
    [d] = MotorFusao().processar(achado("dessaturacao", "news2", sinal="spo2", valor=88.0))
    assert d.alerta.severidade is Severidade.MODERADO
    assert d.alerta.destinatario == "enfermagem do leito"
    assert d.alerta.codigo == "SV-DESSATURACAO"
    assert "SpO2 88%" in d.alerta.mensagem


def test_corroboracao_escalona_ate_alto_e_nao_repete():
    m = MotorFusao()
    m.processar(achado("dessaturacao", "news2"))
    [d] = m.processar(achado("spo2_abaixo_do_basal", "desvio-robusto", minuto=1, sinal="spo2", valor=85.0, basal=96.0))
    assert d.alerta.severidade is Severidade.ALTO
    assert d.alerta.mensagem.startswith("ESCALONAMENTO")
    assert len(d.alerta.achados) == 2
    # terceiro detector do mesmo sensor: corroboração intramodal não passa de alto
    assert m.processar(achado("padrao_multivariado", "isolation-forest", minuto=2)) == []


def test_concordancia_multimodal_chega_a_critico():
    m = MotorFusao()
    m.processar(achado("dessaturacao", "news2"))
    m.processar(achado("spo2_abaixo_do_basal", "desvio-robusto", minuto=1, sinal="spo2", valor=85.0, basal=96.0))
    [d] = m.processar(achado("fala_ofegante", "azure-speech", minuto=4, escore=0.7, modalidade=Modalidade.AUDIO))
    assert d.alerta.severidade is Severidade.CRITICO
    assert d.alerta.multimodal and d.alerta.codigo.startswith("MM-")
    assert d.alerta.destinatario == "time de resposta rápida"


def test_news2_alto_e_critico_mesmo_sozinho():
    [d] = MotorFusao().processar(achado("news2_alto", "news2", news2_parcial=8))
    assert d.alerta.severidade is Severidade.CRITICO


def test_novo_episodio_apos_janela_sem_achados():
    m = MotorFusao(janela_episodio=timedelta(minutes=30))
    assert m.processar(achado("taquipneia", "news2"))
    assert m.processar(achado("taquipneia", "news2", minuto=10)) == []      # mesmo episódio, mesma severidade
    assert m.processar(achado("taquipneia", "news2", minuto=45))            # passou a janela: episódio novo


def test_episodio_expira_pelo_relogio():
    m = MotorFusao(janela_episodio=timedelta(minutes=30))
    m.processar(achado("taquipneia", "news2"))
    assert m.episodio_aberto("P1", T0 + timedelta(minutes=20))["severidade"] == "moderado"
    assert m.episodio_aberto("P1", T0 + timedelta(minutes=31)) is None


def test_corroboracao_considera_so_a_janela_recente():
    m = MotorFusao(janela_corroboracao=timedelta(minutes=15))
    m.processar(achado("taquipneia", "news2"))
    # outro detector, mas 20 min depois: não corrobora
    assert m.processar(achado("fr_acima_do_basal", "desvio-robusto", minuto=20, sinal="fr", valor=30.0, basal=16.0)) == []


def test_pacientes_sao_independentes():
    m = MotorFusao()
    m.processar(achado("dessaturacao", "news2", paciente="P1"))
    [d] = m.processar(achado("spo2_abaixo_do_basal", "desvio-robusto", paciente="P2", escore=0.6,
                             sinal="spo2", valor=90.0, basal=97.0))
    assert d.alerta.paciente_id == "P2" and d.alerta.severidade is Severidade.MODERADO


def test_aviso_tecnico_vai_para_enfermagem_como_informativo():
    [d] = MotorFusao().processar(achado("sensor_sem_sinal", "filtro-artefato", escore=0.2,
                                        sinal="spo2", minutos=10, tecnico=True))
    assert d.alerta.severidade is Severidade.INFORMATIVO and d.alerta.codigo == "TEC-SENSOR_SEM_SINAL"


def test_informativo_nao_vira_alerta_clinico():
    assert MotorFusao().processar(achado("fc_acima_do_basal", "desvio-robusto", escore=0.3,
                                         sinal="fc", valor=95.0, basal=80.0)) == []


def test_achado_sem_fuso_convive_com_achado_com_fuso():
    m = MotorFusao()
    ingenuo = Achado("P1", SV, "dessaturacao", 0.6, "news2", momento=datetime(2026, 1, 1, 8, 0))
    m.processar(ingenuo)
    assert m.processar(achado("fala_ofegante", "azure-speech", minuto=2, escore=0.7, modalidade=Modalidade.AUDIO))


def test_descrever_em_ptbr():
    assert descrever(achado("hipotensao", "news2", sinal="pas", valor=86.9)) == "Hipotensão — PAS 86,9 mmHg"
    assert "abaixo do basal do paciente (96,1)" in descrever(
        achado("spo2_abaixo_do_basal", "desvio-robusto", sinal="spo2", valor=80.7, basal=96.1))


def test_audio_e_texto_da_mesma_consulta_corroboram_mas_nao_sao_canais_independentes():
    m = MotorFusao()
    m.processar(achado("fala_ofegante", "librosa", escore=0.75, modalidade=Modalidade.AUDIO))
    [d] = m.processar(achado("termo_critico", "lexico-hospital", minuto=0, escore=0.8,
                             modalidade=Modalidade.TEXTO, categoria="respiratorio", termos=["falta de ar"]))
    assert d.alerta.severidade is Severidade.ALTO          # não crítico: mesma gravação
    assert not d.alerta.codigo.startswith("MM-")


def test_consulta_mais_monitor_chega_a_critico():
    m = MotorFusao()
    m.processar(achado("taquipneia", "news2", sinal="fr", valor=27.0))
    [d] = m.processar(achado("fala_ofegante", "librosa", minuto=3, escore=0.75, modalidade=Modalidade.AUDIO))
    assert d.alerta.severidade is Severidade.CRITICO      # monitor e consulta: canais independentes
    assert d.alerta.codigo.startswith("MM-")
    assert "concordância entre consulta e monitor" in d.alerta.mensagem
    # o texto da mesma consulta não gera um segundo alerta: já está no teto do episódio
    assert m.processar(achado("termo_critico", "lexico-hospital", minuto=3, escore=0.8,
                              modalidade=Modalidade.TEXTO, categoria="respiratorio", termos=["falta de ar"])) == []


def test_risco_de_autolesao_e_critico_sozinho():
    [d] = MotorFusao().processar(achado("termo_critico", "lexico-hospital", escore=0.95,
                                        modalidade=Modalidade.TEXTO, categoria="risco_autolesao",
                                        termos=["vontade de morrer"]))
    assert d.alerta.severidade is Severidade.CRITICO
