from datetime import timezone

import pytest

from med_monitor.domain import Achado, Alerta, Modalidade, Severidade


def test_achado_rejeita_escore_fora_do_intervalo():
    with pytest.raises(ValueError):
        Achado("PRT-1", Modalidade.SINAIS_VITAIS, "taquicardia", 1.2, "zscore")


def test_achado_serializa_enum_e_data():
    a = Achado("PRT-1", Modalidade.AUDIO, "disartria", 0.7, "librosa")
    d = a.as_dict()
    assert d["modalidade"] == "audio"
    assert a.momento.tzinfo == timezone.utc
    assert isinstance(d["momento"], str)


def test_severidade_elevar_satura_no_critico():
    assert Severidade.MODERADO.elevar() is Severidade.ALTO
    assert Severidade.CRITICO.elevar() is Severidade.CRITICO


def test_alerta_multimodal_quando_ha_modalidades_distintas():
    a1 = Achado("PRT-1", Modalidade.SINAIS_VITAIS, "dessaturacao", 0.9, "limiar")
    a2 = Achado("PRT-1", Modalidade.AUDIO, "fala_ofegante", 0.6, "librosa")
    alerta = Alerta("PRT-1", "RESP-001", Severidade.ALTO, "Possível desconforto respiratório",
                    achados=(a1.id, a2.id), modalidades=(a1.modalidade, a2.modalidade))
    assert alerta.multimodal
    assert alerta.as_dict()["modalidades"] == ["sinais_vitais", "audio"]


def test_alerta_unimodal():
    alerta = Alerta("PRT-1", "SV-001", Severidade.MODERADO, "x", achados=("a",),
                    modalidades=(Modalidade.SINAIS_VITAIS, Modalidade.SINAIS_VITAIS))
    assert not alerta.multimodal
