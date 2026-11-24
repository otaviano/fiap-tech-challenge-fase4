from types import SimpleNamespace

import numpy as np
import pytest

from med_monitor.audio import acustica, consulta, termos
from med_monitor.audio.acustica import SR, FeaturesVoz
from med_monitor.azure.language import AnaliseTexto, EntidadeSaude
from med_monitor.azure.speech import Transcricao, transcricao_de_referencia
from med_monitor.domain import Modalidade


# ------------------------------------------------------------- áudio sintético (sem espeak)

def _vogal(dur_s, f0=120.0, amp=0.3):
    t = np.arange(int(dur_s * SR)) / SR
    sinal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    silabas = 0.55 + 0.45 * np.sin(2 * np.pi * 5 * t) ** 2        # ~5 sílabas/s de modulação
    return amp * sinal / 2 * silabas


def _fala(grupos_s, pausa_s, f0=120.0, sopro=False, seed=0):
    rng = np.random.default_rng(seed)
    partes = [np.zeros(int(0.4 * SR))]
    for g in grupos_s:
        partes.append(_vogal(g, f0))
        n = int(pausa_s * SR)
        silencio = np.zeros(n)
        if sopro and n > int(0.3 * SR):   # respiração: ruído grave e fraco no começo da pausa
            from scipy.signal import butter, sosfilt
            k = int(0.3 * SR)
            ruido = sosfilt(butter(4, [300, 2000], btype="band", fs=SR, output="sos"), rng.normal(0, 1, k))
            silencio[:k] = ruido / np.max(np.abs(ruido)) * 0.02 * np.hanning(k)
        partes.append(silencio)
    y = np.concatenate(partes)
    return y + rng.normal(0, 10 ** (-55 / 20), len(y))


@pytest.fixture(scope="module")
def basal():
    return acustica.extrair(_fala([3.0] * 6, 0.5))


def test_extrai_grupos_e_pausas(basal):
    assert 2.5 <= basal.grupo_fala_medio_s <= 3.5
    assert basal.f0_mediana_hz == pytest.approx(120, rel=0.1)
    assert basal.inspiracoes_por_min == 0


def test_fala_entrecortada_com_respiracao_e_dispneia(basal):
    atual = acustica.extrair(_fala([0.9] * 18, 0.6, sopro=True))
    assert atual.inspiracoes_por_min > 4
    tipos = [a.tipo for a in acustica.detectar(atual, basal)]
    assert tipos == ["fala_ofegante"]


def test_mesma_fala_nao_gera_alteracao(basal):
    assert acustica.detectar(acustica.extrair(_fala([3.0] * 6, 0.5, seed=3)), basal) == []


def test_sem_basal_usa_referencia_e_declara(basal):
    atual = acustica.extrair(_fala([0.9] * 18, 0.6, sopro=True))
    [alt] = acustica.detectar(atual, None)
    assert alt.comparacao == "referência populacional" and alt.escore == 0.5


def test_audio_sem_fala_falha_explicitamente():
    with pytest.raises(ValueError):
        acustica.extrair(np.random.default_rng(0).normal(0, 1e-4, SR * 2))


def _feats(**kw):
    base = dict(duracao_s=20, tempo_fala_s=17, razao_pausa=0.1, pausas_por_min=12, pausa_media_s=0.4,
                grupo_fala_medio_s=3.0, taxa_articulacao=5.0, f0_mediana_hz=120.0, f0_variab_st=1.5,
                intensidade_db=-25.0, planicidade_espectral=0.01, inspiracoes_por_min=0.0)
    return FeaturesVoz(**{**base, **kw})


def test_fadiga_exige_dois_criterios():
    b = _feats()
    assert acustica.detectar(_feats(taxa_articulacao=3.5), b) == []                       # só lenta
    [a] = acustica.detectar(_feats(taxa_articulacao=3.5, intensidade_db=-31.0), b)        # lenta + fraca
    assert a.tipo == "fadiga_vocal"


def test_pausas_com_fala_lenta_sem_respiracao_sao_fadiga_nao_dispneia():
    b = _feats()
    atual = _feats(taxa_articulacao=3.5, intensidade_db=-31.0, grupo_fala_medio_s=1.5, pausas_por_min=25)
    assert [a.tipo for a in acustica.detectar(atual, b)] == ["fadiga_vocal"]
    com_sopro = _feats(taxa_articulacao=3.5, intensidade_db=-31.0, grupo_fala_medio_s=1.5, pausas_por_min=25,
                       inspiracoes_por_min=8.0)
    assert sorted(a.tipo for a in acustica.detectar(com_sopro, b)) == ["fadiga_vocal", "fala_ofegante"]


def test_fala_fragmentada_sem_sinal_respiratorio_nao_e_dispneia():
    """Frases curtas e pausas na pontuação mudam o estilo, não a respiração."""
    b = _feats()
    so_fragmentada = _feats(grupo_fala_medio_s=1.5, pausas_por_min=25)
    assert acustica.detectar(so_fragmentada, b) == []
    com_respiracao = _feats(grupo_fala_medio_s=1.5, pausas_por_min=25, inspiracoes_por_min=8.0)
    assert [a.tipo for a in acustica.detectar(com_respiracao, b)] == ["fala_ofegante"]
    soprosa = _feats(grupo_fala_medio_s=1.5, pausas_por_min=25, planicidade_espectral=0.02)
    assert [a.tipo for a in acustica.detectar(soprosa, b)] == ["fala_ofegante"]


# ------------------------------------------------------------------- termos

@pytest.mark.parametrize("texto, esperado", [
    ("Estou com falta de ar desde cedo.", {("falta de ar", False)}),
    ("Não tenho falta de ar.", {("falta de ar", True)}),
    ("Não consigo respirar.", {("nao consigo respirar", False)}),
    ("Sem febre, mas com tontura.", {("febre", True), ("tontura", False)}),
    ("Nunca desmaiei. Ontem tive palpitação.", {("desmaiei", True), ("palpitacao", False)}),
    ("DOR NO PEITO forte", {("dor no peito", False)}),
])
def test_termos_e_negacao(texto, esperado):
    assert {(t.termo, t.negado) for t in termos.encontrar(texto)} == esperado


def test_termo_mais_longo_tem_prioridade():
    [t] = termos.encontrar("Não consigo respirar")
    assert t.termo == "nao consigo respirar" and t.categoria == "respiratorio"


# ------------------------------------------------------------------ pipeline

class _SpeechFalso:
    def __init__(self, disponivel=True, falha=False):
        self.disponivel, self.falha = disponivel, falha

    def transcrever(self, caminho):
        if self.falha:
            raise RuntimeError("timeout")
        return Transcricao("Estou com falta de ar e muito cansaço, não tenho febre.", "azure-speech")


class _LanguageFalso:
    disponivel = True

    def analisar(self, texto):
        return AnaliseTexto(sentimento="negative", confianca_sentimento={"negative": 0.9},
                            entidades=[EntidadeSaude("falta de ar", "SymptomOrSign", 0.95),
                                       EntidadeSaude("tosse seca", "SymptomOrSign", 0.9),
                                       EntidadeSaude("febre", "SymptomOrSign", 0.9, negada=True)])


@pytest.fixture
def wav(tmp_path):
    import soundfile as sf

    caminho = tmp_path / "PRT-1_atual.wav"
    sf.write(caminho, _fala([3.0] * 4, 0.5), SR)
    return caminho


def test_pipeline_com_azure_falso_gera_achados_de_texto(wav):
    rel = consulta.analisar_consulta(wav, "PRT-1", speech=_SpeechFalso(), language=_LanguageFalso())
    assert rel.servicos == {"speech": "azure-speech", "language": "azure-language"}
    tipos = {(a.modalidade, a.tipo, a.fonte) for a in rel.achados}
    assert (Modalidade.TEXTO, "termo_critico", "lexico-hospital+azure-ta4h") in tipos   # corroborado
    assert (Modalidade.TEXTO, "sintoma_relatado", "azure-ta4h") in tipos                # "tosse seca"
    assert (Modalidade.TEXTO, "sentimento_negativo", "azure-language") in tipos
    extra = next(a for a in rel.achados if a.tipo == "sintoma_relatado")
    assert extra.evidencia["sintomas"] == ["tosse seca"]                                 # febre negada fica fora


def test_pipeline_sem_azure_declara_e_usa_roteiro(wav):
    wav.with_suffix(".txt").write_text("Estou com falta de ar.", encoding="utf-8")
    rel = consulta.analisar_consulta(wav, "PRT-1", speech=_SpeechFalso(False), language=SimpleNamespace(disponivel=False))
    assert rel.servicos == {"speech": "referencia (roteiro .txt)", "language": "indisponível"}
    assert [a.tipo for a in rel.achados] == ["termo_critico"]


def test_falha_do_azure_nao_derruba_a_analise(wav):
    rel = consulta.analisar_consulta(wav, "PRT-1", speech=_SpeechFalso(falha=True),
                                     language=SimpleNamespace(disponivel=False))
    assert rel.erros and rel.erros[0].startswith("speech:")
    assert rel.servicos["speech"] == "indisponível" and rel.features["grupo_fala_medio_s"] > 0


def test_transcricao_de_referencia_ausente(tmp_path):
    assert transcricao_de_referencia(tmp_path / "x.wav") is None


# ------------------------------------------------------------ síntese neural

def test_ssml_azure_tem_voz_prosodia_pausas_e_escapa_texto():
    from med_monitor.audio.amostras import ssml_azure

    s = ssml_azure("Dor <forte> & falta de ar agora mesmo", "dispneia", "pt-BR-AntonioNeural")
    assert '<voice name="pt-BR-AntonioNeural">' in s and 'xml:lang="pt-BR"' in s
    assert "&lt;forte&gt;" in s and "&amp;" in s
    assert s.count('<break time="550ms"/>') == 2               # 8 palavras, pausa a cada 3
    assert 'rate="-30%"' in ssml_azure("ok", "fadiga", "pt-BR-FranciscaNeural")


def test_sintese_azure_usa_cache_e_cai_no_espeak_sem_chave(tmp_path, monkeypatch):
    import soundfile as sf

    from med_monitor.audio import amostras
    from med_monitor.config import AzureSettings

    monkeypatch.setattr(amostras, "CACHE_TTS", tmp_path)
    chamadas = []

    def tts_falso(texto, condicao, voz, settings):
        chamadas.append(voz)
        return np.sin(np.linspace(0, 2000, SR)) * 0.5, SR

    monkeypatch.setattr(amostras, "_tts_azure", tts_falso)
    monkeypatch.setattr(amostras, "_tts_espeak", lambda t, c, v: (chamadas.append(v) or np.zeros(SR) + 0.1, SR))
    com_chave = AzureSettings(speech_key="x", speech_region="brazilsouth")
    sem_chave = AzureSettings(speech_key="", speech_region="")
    amostras.sintetizar("teste", "normal", "pt-BR-AntonioNeural", settings=com_chave)
    amostras.sintetizar("teste", "normal", "pt-BR-AntonioNeural", settings=sem_chave)
    assert chamadas == ["pt-BR-AntonioNeural", "pt-br+m3"]     # sem chave: variante equivalente do espeak


def test_qualificador_positivo_nao_vira_sintoma(tmp_path):
    import soundfile as sf

    class Lang:
        disponivel = True

        def analisar(self, texto):
            return AnaliseTexto(sentimento="mixed", confianca_sentimento={"negative": 0.4},
                                entidades=[EntidadeSaude("Dormi bem", "SymptomOrSign", 0.9),
                                           EntidadeSaude("Estou bem", "SymptomOrSign", 0.8),
                                           EntidadeSaude("tosse", "SymptomOrSign", 0.9)])

    caminho = tmp_path / "P_atual.wav"
    sf.write(caminho, _fala([3.0] * 3, 0.5), SR)
    caminho.with_suffix(".txt").write_text("Dormi bem, estou bem, só uma tosse.", encoding="utf-8")
    rel = consulta.analisar_consulta(caminho, "P", speech=_SpeechFalso(False), language=Lang())
    [extra] = [a for a in rel.achados if a.tipo == "sintoma_relatado"]
    assert extra.evidencia["sintomas"] == ["tosse"]
