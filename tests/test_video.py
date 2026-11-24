from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from med_monitor.dados import rehab24
from med_monitor.domain import Achado, Modalidade, Severidade
from med_monitor.fusao.motor import MotorFusao
from med_monitor.video import avaliacao, metricas, objetos, relatorio
from med_monitor.video.pose import K, SeriePose, _escolher


def _serie(T=60, braco_graus=None, deitado=None, pessoas=None):
    """Pessoa de pé, de frente; `braco_graus` (T,) abre o braço direito; `deitado` (T,) bool."""
    xy = np.zeros((T, 17, 2))
    ombro_d, ombro_e = np.array([520.0, 300.0]), np.array([440.0, 300.0])
    quadril_d, quadril_e = np.array([520.0, 500.0]), np.array([440.0, 500.0])
    for t in range(T):
        pts = {"ombro_d": ombro_d, "ombro_e": ombro_e, "quadril_d": quadril_d, "quadril_e": quadril_e,
               "joelho_d": np.array([510.0, 650]), "joelho_e": np.array([450.0, 650]),
               "tornozelo_d": np.array([510.0, 800]), "tornozelo_e": np.array([450.0, 800]),
               "cotovelo_e": np.array([440.0, 420]), "punho_e": np.array([440.0, 520]), "nariz": np.array([480.0, 220])}
        g = np.radians(braco_graus[t] if braco_graus is not None else 0)
        direcao = np.array([np.sin(g), np.cos(g)])          # 0° = braço para baixo, 90° = horizontal
        pts["cotovelo_d"] = ombro_d + 120 * direcao
        pts["punho_d"] = ombro_d + 220 * direcao
        for nome, p in pts.items():
            xy[t, K[nome]] = p
        if deitado is not None and deitado[t]:
            for nome in ("quadril_d", "quadril_e", "tornozelo_d", "tornozelo_e", "ombro_d", "ombro_e"):
                xy[t, K[nome], 1] = 900 + (5 if "quadril" in nome else 0)
    caixa = np.tile([400.0, 200, 560, 820], (T, 1))
    if deitado is not None:
        caixa[deitado] = [200, 850, 900, 950]
    return SeriePose("teste", 15.0, np.arange(T) * 2, xy, np.ones((T, 17)), caixa,
                     np.array(pessoas if pessoas is not None else [1] * T), 1920, 1080)


# ---------------------------------------------------------------- geometria

def test_angulo_reto():
    a, b, c = np.array([[1.0, 0]]), np.array([[0.0, 0]]), np.array([[0.0, 1]])
    assert metricas.angulo(a, b, c)[0] == pytest.approx(90)


@pytest.mark.parametrize("graus", [0, 45, 90, 150])
def test_abducao_do_ombro_acompanha_o_braco(graus):
    s = _serie(T=5, braco_graus=np.full(5, graus))
    assert metricas.angulos(s)["abducao_ombro_d"][0] == pytest.approx(graus, abs=1)


def test_cotovelo_estendido_e_tronco_reto():
    a = metricas.angulos(_serie(T=3, braco_graus=np.full(3, 90)))
    assert a["cotovelo_d"][0] == pytest.approx(180, abs=1)
    assert a["inclinacao_tronco"][0] == pytest.approx(0, abs=1)


def test_mudancas_de_direcao_contam_hesitacoes():
    t = np.linspace(0, 1, 120)
    suave = 80 * np.sin(np.pi * t)                       # sobe e desce: 1 inversão
    hesitante = suave + 12 * np.sin(2 * np.pi * 6 * t)   # degraus no caminho
    assert metricas.mudancas_de_direcao(suave) == 1
    assert metricas.mudancas_de_direcao(hesitante) > 3


def test_metricas_da_repeticao():
    graus = np.concatenate([np.linspace(0, 160, 30), np.linspace(160, 0, 30)])
    s = _serie(T=60, braco_graus=graus)
    m = metricas.metricas_repeticao(s, metricas.angulos(s), 0, 118, exercicio=1)
    assert m["abducao_ombro_d_max"] == pytest.approx(152, abs=10)
    assert m["abducao_ombro_d_amp"] > 120
    assert m["duracao_s"] == pytest.approx(119 / 30, abs=0.05)
    assert m["mudancas_direcao"] == 1 and m["pose_valida"] == 1.0


def test_confianca_baixa_vira_nan():
    s = _serie(T=4)
    s.conf[:, K["cotovelo_d"]] = 0.1
    assert np.isnan(metricas.angulos(s)["abducao_ombro_d"]).all()


# ------------------------------------------------------------ áreas críticas

def test_rastreia_o_paciente_por_proximidade():
    caixas = np.array([[0, 0, 300, 600], [800, 100, 900, 400]], dtype=float)
    assert _escolher(caixas, None) == 0                                    # maior no início
    assert _escolher(caixas, np.array([790, 90, 910, 410.0])) == 1        # depois, o mais próximo


def test_pessoa_extra_em_intervalos():
    s = _serie(T=60, pessoas=[1] * 20 + [2] * 30 + [1] * 10)
    [e] = objetos.eventos_pessoas_extras(s)
    assert e.tipo == "pessoa_na_area" and e.inicio_s == pytest.approx(40 / 30, abs=0.1)
    assert objetos.eventos_pessoas_extras(_serie(T=60)) == []


def test_queda_detectada_so_quando_sustentada():
    deitado = np.zeros(90, bool); deitado[40:80] = True
    [q] = objetos.queda(_serie(T=90, deitado=deitado))
    assert q.tipo == "queda"
    breve = np.zeros(90, bool); breve[40:44] = True                    # < 1 s: não é queda
    assert objetos.queda(_serie(T=90, deitado=breve)) == []
    assert objetos.queda(_serie(T=90)) == []


def test_area_do_paciente_tem_margem():
    a = objetos.area_do_paciente(_serie(T=10))
    assert a[0] < 400 and a[2] > 560


# --------------------------------------------------------------- avaliação

def _tabela(n_pessoas=5, por_pessoa=10, seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for p in range(1, n_pessoas + 1):
        for i in range(por_pessoa):
            correta = i % 2 == 0
            linhas.append({"video": f"V{p}", "camera": 17, "pessoa": p, "exercicio": 1, "repeticao": i + 1,
                           "inicio": i * 200, "fim": i * 200 + 150, "correta": correta, "pose_valida": 1.0,
                           "abducao_ombro_d_max": rng.normal(160 if correta else 115, 8),
                           "abducao_ombro_d_amp": rng.normal(140 if correta else 95, 8),
                           "cotovelo_d_min": rng.normal(165, 5), "inclinacao_tronco_max": rng.normal(4 if correta else 11, 2),
                           "desnivel_ombros_max": rng.normal(10, 3), "abducao_ombro_e_max": rng.normal(15, 4),
                           "duracao_s": rng.normal(7, 1), "mudancas_direcao": 1.0,
                           "fracao_pessoa_visivel": 0.0, "fracao_pessoa_na_area": 0.0, "pessoa_extra_rotulo": 0})
    return pd.DataFrame(linhas)


def test_classificador_validado_por_pessoa_separa_padroes_claros():
    r = avaliacao.avaliar_exercicio(_tabela(), 1)
    assert r["pessoas"] == 5 and r["repeticoes"] == 50
    assert r["auc"] > 0.95 and r["sensibilidade"] > 0.85
    assert list(r["auc_por_metrica"])[0] in ("abducao_ombro_d_max", "abducao_ombro_d_amp")


def test_explicacao_aponta_a_metrica_desviante():
    df = _tabela()
    ref = avaliacao.referencia_correta(df, 1, excluir_pessoa=1)
    linha = df[(df.pessoa == 1) & ~df.correta].iloc[0]
    frases = avaliacao.explicar(linha, ref)
    assert frases and "abaixo do padrão correto" in frases[0]


def test_pessoas_extras_contra_rotulo():
    df = pd.DataFrame({"fracao_pessoa_visivel": [0.0, 0.9, 0.1, 0.1], "fracao_pessoa_na_area": [0, 0.5, 0, 0],
                       "pessoa_extra_rotulo": [0, 3, 2, 1]})
    r = avaliacao.avaliar_pessoas_extras(df)
    assert r["pessoa_inteira"]["sensibilidade"] == 1.0 and r["pessoa_inteira"]["especificidade"] == 1.0
    assert r["parte_perceptivel_ou_mais"]["sensibilidade"] == 0.5          # o braço solto não é detectado
    assert r["repeticoes_com_pessoa_na_area"] == 1


# ---------------------------------------------------------------- achados

def _rel(probs, comparacao=None, eventos=()):
    reps = [{"repeticao": i + 1, "inicio_s": i * 7.0, "fim_s": i * 7.0 + 6, "prob_incorreta": p,
             "rotulo_fisioterapeuta": "incorreta" if p >= 0.5 else "correta",
             "desvios": ["amplitude da abdução 92° — abaixo do padrão correto (141°, z=-4,1)"] if p >= 0.5 else []}
            for i, p in enumerate(probs)]
    return relatorio.RelatorioSessao("V1", 17, 1, 1, reps, list(eventos), {}, comparacao)


def test_achado_de_desvio_tem_teto_moderado_e_vai_para_fisioterapia():
    rel = _rel([0.9, 0.95, 0.9, 0.1])
    [a] = relatorio._achados(rel, "P1", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert a.tipo == "desvio_execucao" and a.escore < 0.8
    assert a.evidencia["desvios"] == ["amplitude da abdução abaixo do padrão correto"]
    [d] = MotorFusao().processar(a)
    assert d.alerta.severidade is Severidade.MODERADO and d.alerta.destinatario == "fisioterapia"


def test_sessao_toda_correta_nao_gera_achado():
    assert relatorio._achados(_rel([0.1, 0.2]), "P1", datetime.now(timezone.utc)) == []


def test_amplitude_reduzida_e_queda():
    comp = {"sessao_anterior": "V0", "metrica": "abducao_ombro_d_amp", "atual": 90.0, "anterior": 140.0,
            "variacao": -0.357}
    queda = {"tipo": "queda", "inicio_s": 3.0, "fim_s": 5.0, "detalhe": "paciente no chão"}
    achados = relatorio._achados(_rel([0.1], comp, [queda]), "P1", datetime.now(timezone.utc))
    tipos = {(a.modalidade, a.tipo) for a in achados}
    assert tipos == {(Modalidade.MOVIMENTACAO, "amplitude_reduzida"), (Modalidade.VIDEO, "queda")}
    q = next(a for a in achados if a.tipo == "queda")
    [d] = MotorFusao().processar(q)
    assert d.alerta.severidade is Severidade.ALTO


# ------------------------------------------------------------------ dataset

def test_camera_ideal_segue_o_plano_do_movimento():
    R = rehab24.Repeticao
    braco = [R("A", 1, 1, 1, 0, 10, "front", "right arm", True)]
    agacha = [R("B", 1, 6, 1, 0, 10, "front", "", True)]
    assert rehab24.camera_ideal("A", braco) == 17      # abdução: câmera de frente
    assert rehab24.camera_ideal("B", agacha) == 18     # agachamento: câmera de lado


def test_baixar_videos_so_pede_o_que_falta(tmp_path, monkeypatch):
    monkeypatch.setattr(rehab24, "DESTINO", tmp_path)

    class ZipFalso:
        extraidos = []

        def nomes(self):
            return ["Ex1/PM_000-Camera17-30fps.mp4", "Ex1/PM_000-Camera18-30fps-transposed.mp4"]

        def extrair(self, nome, destino):
            self.extraidos.append(nome)
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"x")
            return destino

    z = ZipFalso()
    rehab24.baixar_videos(["PM_000"], 18, z)
    rehab24.baixar_videos(["PM_000"], 18, z)          # já existe: não baixa de novo
    assert z.extraidos == ["Ex1/PM_000-Camera18-30fps-transposed.mp4"]


def test_area_critica_ignora_pessoa_longe_ou_incerta():
    T = 40
    s = _serie(T=T, pessoas=[2] * T)
    outras = np.full((T, 4, 5), np.nan)
    outras[:10, 0] = [1700, 200, 1919, 1070, 0.5]     # vulto na borda, confiança baixa: ignora
    outras[10:20, 0] = [1500, 200, 1700, 900, 0.9]    # pessoa nítida, mas longe do paciente: ignora
    outras[20:40, 0] = [500, 250, 700, 850, 0.9]      # pessoa dentro da área do paciente: conta
    s.outras = outras
    mascara = objetos.pessoa_na_area(s)
    assert not mascara[:20].any() and mascara[20:].all()
    [e] = objetos.eventos_pessoas_extras(s)
    assert e.detalhe == "outra pessoa na área do paciente"
