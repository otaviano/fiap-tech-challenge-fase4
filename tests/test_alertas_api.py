import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from med_monitor.alertas.despacho import CanalWebhook, Despachante
from med_monitor.audit import AuditTrail
from med_monitor.domain import Achado, Modalidade
from med_monitor.fusao.motor import MotorFusao
from med_monitor.serving.api import _sse, criar_app
from med_monitor.serving.simulador import Leito, Simulador

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def despachante(tmp_path):
    return Despachante(AuditTrail(tmp_path / "aud.jsonl"))


def _decisao(minuto=0):
    a = Achado("P1", Modalidade.SINAIS_VITAIS, "news2_alto", 0.7, "news2",
               evidencia={"news2_parcial": 8}, momento=T0 + timedelta(minutes=minuto))
    return MotorFusao().processar(a)[0]


def test_despacho_audita_publica_e_guarda(despachante):
    async def caso():
        fila = despachante.assinar()
        reg = despachante.despachar(_decisao(), "trace-1", contexto={"leito": "UTI-01"})
        tipo, dados = fila.get_nowait()
        return reg, tipo, dados

    reg, tipo, dados = asyncio.run(caso())
    assert tipo == "alerta" and dados["id"] == reg["id"]
    assert reg["achados_detalhe"][0]["tipo"] == "news2_alto"
    assert despachante.listar()[0]["id"] == reg["id"]
    assert [e["evento"] for e in despachante.trilha.read("trace-1")] == ["alerta_emitido"]


def test_reconhecimento_registra_quem_e_quando(despachante):
    reg = despachante.despachar(_decisao(), "t")
    out = despachante.reconhecer(reg["id"], "CRM-SP 111.222", "t2", "avaliado à beira do leito")
    assert out["reconhecido"]["profissional"] == "CRM-SP 111.222"
    assert out["reconhecido"]["segundos_ate_ciencia"] >= 0
    assert despachante.trilha.read("t2")[0]["evento"] == "alerta_reconhecido"
    # segunda ciência não sobrescreve a primeira
    assert despachante.reconhecer(reg["id"], "outro", "t3")["reconhecido"]["profissional"] == "CRM-SP 111.222"


def test_canal_externo_que_falha_nao_derruba_despacho(tmp_path):
    class Quebrado:
        nome = "quebrado"

        def enviar(self, registro):
            raise ConnectionError("fora do ar")

    d = Despachante(AuditTrail(tmp_path / "a.jsonl"), canais=(Quebrado(),))
    d.despachar(_decisao(), "t")
    assert [e["evento"] for e in d.trilha.read("t")] == ["alerta_emitido", "canal_falhou"]


def test_webhook_respeita_severidade_minima(monkeypatch):
    enviados = []
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: enviados.append(req) or _Fecha())
    canal = CanalWebhook("http://exemplo.invalid/hook", minima="alto")
    canal.enviar({"severidade": "moderado", "paciente_id": "P", "mensagem": "m"})
    canal.enviar({"severidade": "critico", "paciente_id": "P", "mensagem": "m"})
    assert len(enviados) == 1


class _Fecha:
    def close(self):
        pass


def test_sse_formato():
    assert _sse("alerta", {"a": "é"}) == 'event: alerta\ndata: {"a": "é"}\n\n'


def test_simulador_detecta_evento_injetado_em_serie_sintetica(despachante):
    sim = Simulador(despachante, velocidade=0, aquecimento=180)
    leito = Leito("UTI-99", "PRT-TESTE", registro=None, eventos=(("dessaturacao", 300, 25),), semente=11)
    asyncio.run(sim.executar((leito,)))
    alertas = despachante.listar(500)
    ev = sim.eventos_injetados[0]
    inicio_ev, fim_ev = datetime.fromisoformat(ev["inicio"]), datetime.fromisoformat(ev["fim"])
    no_evento = [a for a in alertas if inicio_ev <= datetime.fromisoformat(a["momento"]) <= fim_ev]
    assert no_evento, "a dessaturação injetada deveria gerar alerta"
    assert any("SpO2" in a["mensagem"] for a in no_evento)
    assert sim.estado["PRT-TESTE"]["status"] == "fim do registro"
    # o aquecimento (3 h antes do plantão) treina os detectores mas não emite alerta
    assert all(datetime.fromisoformat(a["momento"]) >= sim.inicio for a in alertas)


def test_api_achado_externo_entra_na_fusao(tmp_path):
    d = Despachante(AuditTrail(tmp_path / "a.jsonl"))
    cliente = TestClient(criar_app(despachante=d))
    assert cliente.get("/health").json()["status"] == "ok"
    corpo = {"paciente_id": "P9", "modalidade": "audio", "tipo": "fala_ofegante", "escore": 0.9,
             "fonte": "azure-speech", "momento": "2026-01-01T08:00:00Z"}
    r = cliente.post("/achados", json=corpo).json()
    assert r["alertas"][0]["severidade"] == "alto" and r["alertas"][0]["codigo"] == "AU-FALA_OFEGANTE"
    alerta_id = r["alertas"][0]["id"]
    assert cliente.get("/alertas").json()[0]["id"] == alerta_id
    assert cliente.post(f"/alertas/{alerta_id}/ciente", json={"profissional": "CRM-SP 1"}).status_code == 200
    assert cliente.post("/alertas/naoexiste/ciente", json={"profissional": "CRM-SP 1"}).status_code == 404
    assert cliente.post("/achados", json={**corpo, "escore": 1.5}).status_code == 422
    assert "Painel UTI" in cliente.get("/").text
    eventos = [e["evento"] for e in cliente.get("/auditoria", params={"limite": 50}).json()]
    assert {"achado", "alerta_emitido", "alerta_reconhecido"} <= set(eventos)
