"""Features acústicas da fala e detecção de alterações vocais.

O Azure transcreve **o que** o paciente disse; aqui medimos **como** ele disse.
Cansaço e falta de ar mudam a fala de formas mensuráveis, mesmo quando o
paciente diz "estou bem":

| Feature | Como é medida | O que sinaliza |
|---|---|---|
| `grupo_fala_medio_s` | duração média dos trechos de fala entre pausas | **dispneia**: o paciente para para respirar mais cedo |
| `pausas_por_min` | pausas ≥ 200 ms por minuto de consulta | dispneia, fadiga |
| `razao_pausa` | fração do tempo em silêncio (entre a 1ª e a última fala) | fadiga, dispneia |
| `taxa_articulacao` | núcleos silábicos por segundo de fala (picos de intensidade sonora, à la de Jong & Wempe 2009) | **fadiga**: fala mais lenta |
| `f0_mediana_hz`, `f0_variab_st` | frequência fundamental (pYIN) e sua variabilidade em semitons | fadiga: voz mais grave e **monótona** |
| `intensidade_db` | RMS médio dos trechos de fala | fadiga: voz mais fraca (só comparável com o mesmo microfone) |
| `planicidade_espectral` | planicidade do espectro nos trechos vozeados | **soprosidade** (ar escapando) — proxy simples de HNR |
| `inspiracoes_por_min` | trechos **sem vozeamento** (≥ 150 ms), mais fracos que a fala e com energia em frequências baixas | **dispneia**: respiração audível entre as frases |

**Fala é energia *com* vozeamento.** Um detector de voz só por energia confunde
respiração audível com fala — e justamente no paciente dispneico, em que a
inspiração ruidosa preenche a pausa, ele enxergaria *menos* pausas. Por isso
um quadro só conta como fala se tiver energia **e** vozeamento (pYIN) a até
60 ms. Do que sobra com energia e sem voz, só conta como inspiração o trecho
que é ≥ 6 dB mais fraco que a fala e tem centroide espectral < 3 kHz — o que
separa a respiração das fricativas ("s", "f", "ch"), que também não têm voz mas
concentram energia acima de 4 kHz e se alongam na fala lenta.

**Comparação com o basal do próprio paciente.** Valores absolutos dependem de
microfone, sala, idioma e de quem fala. O que é clínico é a *mudança*: a mesma
pessoa, na consulta anterior, falava em grupos de 3 s e agora fala em grupos de
1,2 s. Sem basal, usamos faixas de referência conservadoras e o relatório diz
que a comparação foi populacional.

Limitação declarada: disartria (fala arrastada por lesão neurológica) exige
medidas de articulação que um proxy de energia não captura com confiança —
fica fora do detector automático e aparece apenas pelo texto (termos como
"fala enrolada").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SR = 16_000
JANELA = 400        # 25 ms
PASSO = 160         # 10 ms
PAUSA_MIN_S = 0.20
FALA_MIN_S = 0.10
DILATACAO_VOZ = 6      # quadros (60 ms): consoantes surdas coladas a vogais continuam sendo fala
INSPIRACAO_MIN_S = 0.15
INSPIRACAO_CENTROIDE_MAX_HZ = 3000
INSPIRACAO_ATENUACAO_DB = 6.0


@dataclass(frozen=True)
class FeaturesVoz:
    duracao_s: float
    tempo_fala_s: float
    razao_pausa: float
    pausas_por_min: float
    pausa_media_s: float
    grupo_fala_medio_s: float
    taxa_articulacao: float
    f0_mediana_hz: float | None
    f0_variab_st: float | None
    intensidade_db: float
    planicidade_espectral: float
    inspiracoes_por_min: float

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def carregar(caminho: Path | str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(caminho), sr=SR, mono=True)
    return y


def _com_energia(rms_db: np.ndarray) -> np.ndarray:
    """Quadros acima de um limiar adaptativo entre o piso de ruído e o pico."""
    piso, pico = np.percentile(rms_db, 10), np.percentile(rms_db, 95)
    return rms_db > piso + 0.30 * (pico - piso)


def _acima_do_ruido(rms_db: np.ndarray, margem_db: float = 6.0) -> np.ndarray:
    """Quadros audíveis acima do ruído da sala — onde a respiração pode estar (mais fraca que a fala)."""
    return rms_db > np.percentile(rms_db, 10) + margem_db


def _suavizar(mascara: np.ndarray) -> np.ndarray:
    """Fecha buracos curtos (< PAUSA_MIN) e remove ilhas curtas (< FALA_MIN)."""
    return _remover_curtos(~_remover_curtos(~mascara, PAUSA_MIN_S), FALA_MIN_S)


def _segmentos_fala(rms_db: np.ndarray, vozeado: np.ndarray | None = None) -> np.ndarray:
    """Máscara de fala: energia e, quando há pYIN, vozeamento próximo."""
    energia = _com_energia(rms_db)
    if vozeado is None:
        return _suavizar(energia)
    perto_de_voz = np.convolve(vozeado.astype(float), np.ones(2 * DILATACAO_VOZ + 1), mode="same") > 0
    return _suavizar(energia & perto_de_voz)


def _remover_curtos(mascara: np.ndarray, minimo_s: float) -> np.ndarray:
    saida = mascara.copy()
    minimo = int(round(minimo_s * SR / PASSO))
    i = 0
    while i < len(saida):
        if saida[i]:
            j = i
            while j < len(saida) and saida[j]:
                j += 1
            if j - i < minimo:
                saida[i:j] = False
            i = j
        else:
            i += 1
    return saida


def _corridas(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Intervalos [início, fim) onde a máscara é verdadeira."""
    bordas = np.diff(np.concatenate([[0], mascara.astype(int), [0]]))
    return list(zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)))


def extrair(y: np.ndarray) -> FeaturesVoz:
    import librosa
    from scipy.signal import find_peaks

    duracao = len(y) / SR
    rms = librosa.feature.rms(y=y, frame_length=JANELA, hop_length=PASSO)[0]
    f0, vozeado, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=PASSO)
    n = min(len(rms), len(vozeado))
    rms, f0, vozeado = rms[:n], f0[:n], vozeado[:n]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=1.0)
    fala = _segmentos_fala(rms_db, vozeado)
    grupos = _corridas(fala)
    if not grupos:
        raise ValueError("nenhum trecho de fala detectado no áudio")

    ini, fim = grupos[0][0], grupos[-1][1]
    janela_util = (fim - ini) * PASSO / SR
    tempo_fala = fala.sum() * PASSO / SR
    pausas = _corridas(~fala[ini:fim])
    dur_pausas = [(b - a) * PASSO / SR for a, b in pausas]
    dur_grupos = [(b - a) * PASSO / SR for a, b in grupos]

    # núcleos silábicos: picos do envelope de intensidade, dentro da fala
    envelope = np.convolve(rms_db, np.ones(5) / 5, mode="same")
    picos, _ = find_peaks(envelope, prominence=2.0, distance=int(0.10 * SR / PASSO))
    silabas = int(np.sum(fala[picos])) if len(picos) else 0

    # respiração audível: energia sem voz, longa, mais fraca que a fala e grave (não fricativa)
    centroide = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=JANELA * 2, hop_length=PASSO)[0][:n]
    nivel_fala = float(np.median(rms_db[fala]))
    sopro = _acima_do_ruido(rms_db) & ~fala
    inspiracoes = 0
    for a, b in _corridas(sopro):
        if not (ini <= a < fim) or (b - a) * PASSO / SR < INSPIRACAO_MIN_S:
            continue
        if (np.median(centroide[a:b]) < INSPIRACAO_CENTROIDE_MAX_HZ
                and np.median(rms_db[a:b]) <= nivel_fala - INSPIRACAO_ATENUACAO_DB):
            inspiracoes += 1

    f0_validos = f0[vozeado & ~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0_validos) >= 10:
        f0_med = float(np.median(f0_validos))
        semitons = 12 * np.log2(f0_validos / f0_med)
        f0_var = float(np.std(semitons))
    else:
        f0_med = f0_var = None

    planicidade = librosa.feature.spectral_flatness(y=y, n_fft=JANELA * 2, hop_length=PASSO)[0][:n]
    voz = fala & vozeado
    plan = float(np.mean(planicidade[voz])) if np.any(voz) else float(np.mean(planicidade[fala]))

    return FeaturesVoz(
        duracao_s=duracao,
        tempo_fala_s=tempo_fala,
        razao_pausa=1 - tempo_fala / janela_util if janela_util > 0 else 0.0,
        pausas_por_min=len(dur_pausas) / (janela_util / 60) if janela_util > 0 else 0.0,
        pausa_media_s=float(np.mean(dur_pausas)) if dur_pausas else 0.0,
        grupo_fala_medio_s=float(np.mean(dur_grupos)),
        taxa_articulacao=silabas / tempo_fala if tempo_fala > 0 else 0.0,
        f0_mediana_hz=f0_med,
        f0_variab_st=f0_var,
        intensidade_db=float(np.mean(rms_db[fala])),
        planicidade_espectral=plan,
        inspiracoes_por_min=inspiracoes / (janela_util / 60) if janela_util > 0 else 0.0,
    )


# ------------------------------------------------------------- detecção

@dataclass(frozen=True)
class Criterio:
    feature: str
    descricao: str
    atendido: bool
    atual: float | None
    referencia: float | None
    variacao: float | None      # relativa (ex.: -0.35 = 35% menor) ou absoluta em dB/st


def _rel(atual, ref):
    return None if atual is None or not ref else (atual - ref) / ref


# Sem basal: faixas conservadoras de fala conversacional adulta. Só disparam
# em alterações grosseiras — a comparação com o basal é o caminho principal.
REFERENCIA = {"grupo_fala_medio_s": 1.5, "pausas_por_min": 30.0, "taxa_articulacao": 3.0, "razao_pausa": 0.40,
              "inspiracoes_por_min": 6.0}
INSPIRACOES_DELTA = 4.0   # inspirações audíveis/min a mais que o basal


def criterios_dispneia(atual: FeaturesVoz, basal: FeaturesVoz | None) -> list[Criterio]:
    if basal is None:
        return [
            Criterio("grupo_fala_medio_s", "grupos de fala curtos (< 1,5 s)",
                     atual.grupo_fala_medio_s < REFERENCIA["grupo_fala_medio_s"], atual.grupo_fala_medio_s,
                     REFERENCIA["grupo_fala_medio_s"], None),
            Criterio("pausas_por_min", "muitas pausas (> 30/min)",
                     atual.pausas_por_min > REFERENCIA["pausas_por_min"], atual.pausas_por_min,
                     REFERENCIA["pausas_por_min"], None),
            Criterio("inspiracoes_por_min", "respiração audível frequente (> 6/min)",
                     atual.inspiracoes_por_min > REFERENCIA["inspiracoes_por_min"], atual.inspiracoes_por_min,
                     REFERENCIA["inspiracoes_por_min"], None),
        ]
    dg = _rel(atual.grupo_fala_medio_s, basal.grupo_fala_medio_s)
    dp = _rel(atual.pausas_por_min, basal.pausas_por_min)
    dpl = _rel(atual.planicidade_espectral, basal.planicidade_espectral)
    return [
        Criterio("grupo_fala_medio_s", "grupos de fala ≥ 35% mais curtos que o basal",
                 dg is not None and dg <= -0.35, atual.grupo_fala_medio_s, basal.grupo_fala_medio_s, dg),
        Criterio("pausas_por_min", "pausas ≥ 50% mais frequentes que o basal",
                 dp is not None and dp >= 0.50, atual.pausas_por_min, basal.pausas_por_min, dp),
        Criterio("planicidade_espectral", "voz mais soprosa (planicidade ≥ 30% maior)",
                 dpl is not None and dpl >= 0.30, atual.planicidade_espectral, basal.planicidade_espectral, dpl),
        Criterio("inspiracoes_por_min", "respiração audível ≥ 4/min acima do basal",
                 atual.inspiracoes_por_min - basal.inspiracoes_por_min >= INSPIRACOES_DELTA,
                 atual.inspiracoes_por_min, basal.inspiracoes_por_min,
                 atual.inspiracoes_por_min - basal.inspiracoes_por_min),
    ]


def criterios_fadiga(atual: FeaturesVoz, basal: FeaturesVoz | None) -> list[Criterio]:
    if basal is None:
        return [
            Criterio("taxa_articulacao", "fala lenta (< 3 sílabas/s)",
                     atual.taxa_articulacao < REFERENCIA["taxa_articulacao"], atual.taxa_articulacao,
                     REFERENCIA["taxa_articulacao"], None),
            Criterio("razao_pausa", "mais de 40% do tempo em silêncio",
                     atual.razao_pausa > REFERENCIA["razao_pausa"], atual.razao_pausa,
                     REFERENCIA["razao_pausa"], None),
        ]
    dt = _rel(atual.taxa_articulacao, basal.taxa_articulacao)
    dv = _rel(atual.f0_variab_st, basal.f0_variab_st)
    df0 = _rel(atual.f0_mediana_hz, basal.f0_mediana_hz)
    di = atual.intensidade_db - basal.intensidade_db
    return [
        Criterio("taxa_articulacao", "fala ≥ 20% mais lenta que o basal",
                 dt is not None and dt <= -0.20, atual.taxa_articulacao, basal.taxa_articulacao, dt),
        Criterio("f0_variab_st", "entonação ≥ 25% mais monótona",
                 dv is not None and dv <= -0.25, atual.f0_variab_st, basal.f0_variab_st, dv),
        Criterio("f0_mediana_hz", "voz ≥ 8% mais grave",
                 df0 is not None and df0 <= -0.08, atual.f0_mediana_hz, basal.f0_mediana_hz, df0),
        Criterio("intensidade_db", "voz ≥ 4 dB mais fraca",
                 di <= -4.0, atual.intensidade_db, basal.intensidade_db, di),
    ]


@dataclass(frozen=True)
class AlteracaoVocal:
    tipo: str               # "fala_ofegante" | "fadiga_vocal"
    escore: float
    criterios: list[Criterio]
    comparacao: str         # "basal do paciente" | "referência populacional"

    def evidencia(self) -> dict:
        return {"comparacao": self.comparacao,
                "criterios_atendidos": [c.descricao for c in self.criterios if c.atendido],
                "medidas": {c.feature: {"atual": None if c.atual is None else round(c.atual, 3),
                                        "referencia": None if c.referencia is None else round(c.referencia, 3)}
                            for c in self.criterios}}


def _pausas_explicadas_pela_lentidao(dispneia: list[Criterio], fadiga: list[Criterio]) -> bool:
    """Pausas a mais com fala lenta e sem respiração audível são fadiga, não falta de ar.

    Na dispneia a fala é interrompida para respirar, mas *dentro* de cada grupo
    a articulação segue no ritmo normal. Na fadiga, a fala inteira desacelera e
    as pausas aumentam junto. Só atribuímos à dispneia quando há respiração
    audível ou quando a articulação está preservada.
    """
    atendidos = {c.feature for c in dispneia if c.atendido}
    lenta = any(c.feature == "taxa_articulacao" and c.atendido for c in fadiga)
    return lenta and "inspiracoes_por_min" not in atendidos


SINAIS_RESPIRATORIOS = frozenset({"inspiracoes_por_min", "planicidade_espectral"})


def _so_fragmentacao(dispneia: list[Criterio]) -> bool:
    """Fala fragmentada sem nenhum sinal respiratório não é falta de ar.

    Grupos de fala mais curtos e pausas mais frequentes são o mesmo fenômeno
    medido duas vezes: contá-los como dois critérios independentes faz qualquer
    mudança no *estilo* de fala (frases curtas, pausas na pontuação) parecer
    dispneia. Exigimos também respiração audível ou voz soprosa.
    """
    return not any(c.atendido and c.feature in SINAIS_RESPIRATORIOS for c in dispneia)


def detectar(atual: FeaturesVoz, basal: FeaturesVoz | None = None) -> list[AlteracaoVocal]:
    """Uma alteração é declarada quando ≥ 2 critérios concordam.

    Um critério isolado é ruído plausível (o paciente estava distraído, falou
    mais baixo, a sala estava barulhenta); dois juntos são padrão. Na fala
    ofegante, um dos dois precisa ser respiratório.
    """
    comparacao = "basal do paciente" if basal is not None else "referência populacional"
    dispneia, fadiga = criterios_dispneia(atual, basal), criterios_fadiga(atual, basal)
    saida = []
    for tipo, criterios in (("fala_ofegante", dispneia), ("fadiga_vocal", fadiga)):
        n = sum(c.atendido for c in criterios)
        if tipo == "fala_ofegante" and n >= 2 and (_so_fragmentacao(dispneia)
                                                   or _pausas_explicadas_pela_lentidao(dispneia, fadiga)):
            continue
        if n >= 2:
            escore = round(min(1.0, 0.45 + 0.15 * n), 2) if basal is not None else 0.5
            saida.append(AlteracaoVocal(tipo, escore, criterios, comparacao))
    return saida
