import asyncio
from datetime import datetime

import pytest

from med_monitor.alertas.despacho import Despachante
from med_monitor.audit import AuditTrail
from med_monitor.domain import Achado, Modalidade
from med_monitor.serving import demo
from med_monitor.serving.demo import EventoDemo
from med_monitor.serving.simulador import Leito, Simulador


def _rodar(sim, leitos, roteiro, ate="07:45"):
    h, m = map(int, ate.split(":"))

    async def caso():
        tarefa = asyncio.create_task(sim.executar(leitos, roteiro))
        while not tarefa.done():
            await asyncio.sleep(0.01)
            if sim.relogio and sim.relogio.hour * 60 + sim.relogio.minute >= h * 60 + m:
                sim.parar()
        await tarefa

    asyncio.run(caso())


@pytest.fixture
def despachante(tmp_path):
    return Despachante(AuditTrail(tmp_path / "a.jsonl"))


def test_evento_de_outro_canal_entra_na_fusao_no_horario(despachante):
    chamadas = []

    def produtor(pid, momento):
        chamadas.append((pid, momento))
        return [Achado(pid, Modalidade.TEXTO, "termo_critico", 0.8, "lexico-hospital",
                       evidencia={"categoria": "respiratorio", "termos": ["falta de ar"]}, momento=momento)], {}

    leitos = (Leito("UTI-91", "P-91", None, semente=1), Leito("UTI-92", "P-92", None, semente=2))
    roteiro = (EventoDemo("07:10", "UTI-91", "P-91", "consulta", "Consulta", produtor),)
    sim = Simulador(despachante, velocidade=0)
    _rodar(sim, leitos, roteiro, ate="07:30")

    [(pid, momento)] = chamadas
    assert pid == "P-91" and momento.hour == 7 and momento.minute == 10
    [evento] = sim.linha_do_tempo
    assert evento["status"] == "concluido" and evento["canal"] == "consulta"
    assert evento["achados"] == ["Paciente relata “falta de ar” (respiratorio)"]
    assert any(a["codigo"] == "TX-TERMO_CRITICO" and a["contexto"]["canal"] == "consulta"
               for a in despachante.listar(100))


def test_leitos_andam_juntos(despachante):
    vistos = {}
    sim = Simulador(despachante, velocidade=0)
    original = despachante.publicar

    def espiar(tipo, dados):
        if tipo == "vitais" and dados.get("momento") and dados["status"] == "monitorando":
            vistos.setdefault(dados["leito"], []).append(dados["momento"])
        original(tipo, dados)

    despachante.publicar = espiar
    leitos = tuple(Leito(f"UTI-{i}", f"P-{i}", None, semente=i) for i in range(3))
    _rodar(sim, leitos, (), ate="07:20")
    # a cada momento, nenhum leito está mais de 1 minuto à frente dos outros
    n = min(len(v) for v in vistos.values()) - 1      # o último minuto pode ter sido interrompido pelo "parar"
    for k in range(n):
        minutos = {datetime.fromisoformat(v[k]).minute for v in vistos.values()}
        assert len(minutos) == 1


def test_produtor_que_falha_nao_derruba_o_plantao(despachante):
    def quebra(pid, momento):
        raise ConnectionError("Azure fora do ar")

    sim = Simulador(despachante, velocidade=0)
    _rodar(sim, (Leito("UTI-91", "P-91", None, semente=1),),
           (EventoDemo("07:05", "UTI-91", "P-91", "consulta", "Consulta", quebra),), ate="07:15")
    [evento] = sim.linha_do_tempo
    assert evento["status"] == "erro" and "Azure fora do ar" in evento["detalhes"]["erro"]
    assert sim.relogio.minute >= 15                     # o plantão seguiu


def test_prescricao_da_demo_e_erro_decimal_de_morfina():
    achados, detalhes = demo.prescricao_morfina("P-1", datetime(2026, 1, 1, 8, 0))
    assert [a.tipo for a in achados] == ["possivel_erro_decimal"]
    assert detalhes["prescricao"]["dose"] == "40 mg" and detalhes["anterior"]["dose"] == "4 mg"


def test_sessao_sem_relatorio_explica_o_que_falta(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "VIDEO_RESULTS", tmp_path)
    achados, detalhes = demo.sessao_fisioterapia("PM_999")("P-1", datetime(2026, 1, 1, 9, 20))
    assert achados == [] and "video relatorio PM_999" in detalhes["erros"][0]


def test_roteiro_da_demo_esta_em_ordem_e_dentro_do_plantao():
    minutos = [e.minutos_apos_inicio() for e in demo.ROTEIRO_DEMO]
    assert minutos == sorted(minutos) and minutos[0] > 0
    leitos = {l.leito for l in demo.LEITOS_DEMO}
    assert all(e.leito in leitos for e in demo.ROTEIRO_DEMO)
    pacientes = {l.leito: l.paciente_id for l in demo.LEITOS_DEMO}
    assert all(pacientes[e.leito] == e.paciente_id for e in demo.ROTEIRO_DEMO)
