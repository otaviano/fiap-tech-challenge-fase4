"""Detecção de anomalias em sinais vitais, em modo streaming.

Cada amostra (um minuto do monitor) passa por um filtro de artefato e por três
detectores complementares. Cada detector cobre um ponto cego dos outros:

| Detector | Pergunta que responde | Pega | Não pega |
|---|---|---|---|
| `DetectorNEWS2` | "Este valor é perigoso para *qualquer* adulto?" | valores absolutos fora da faixa (SpO2 85%, PAS 85) | piora dentro da faixa normal |
| `DetectorDesvioRobusto` | "Este valor é estranho *para este paciente*?" | mudança brusca em relação ao próprio basal (FC 70→110) | deriva lenta, que vira o novo basal |
| `DetectorMultivariado` | "Esta *combinação* de sinais é estranha?" | piora sutil e simultânea em vários sinais (padrão de sepse) | alteração isolada e pequena num só sinal |

Todos são *stateful* e processam um ponto por vez — o mesmo código roda no
replay acelerado da demonstração e num feed de monitor de verdade. Nenhum
detector dispara a cada minuto enquanto a condição dura: um `_Gatilho` exige
persistência para disparar e um período sem a condição para rearmar. É a
diferença entre um alerta e o ruído de alarme que faz a equipe desligar o
monitor.

O NEWS2 (Royal College of Physicians, 2017) usa sete parâmetros; o monitor
numérico tem quatro (FR, SpO2, PAS, FC). Faltam consciência, temperatura e uso
de O2 suplementar — o escore aqui é um **NEWS2 parcial**, sempre ≤ o real.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

from ..config import SEED
from ..domain import Achado, Modalidade

# Fora destas faixas o valor é fisiologicamente implausível: é artefato
# (sensor solto, cabo, movimento), não paciente. Zero cai sempre aqui.
LIMITES_PLAUSIVEIS = {
    "fc": (20, 250), "pas": (40, 260), "pad": (15, 180), "pam": (25, 200),
    "fr": (3, 60), "spo2": (50, 100),
}


def _valido(valor) -> bool:
    return valor is not None and not (isinstance(valor, float) and np.isnan(valor))


def _achado(paciente_id: str, momento, tipo: str, escore: float, fonte: str, **evidencia) -> Achado:
    return Achado(
        paciente_id=paciente_id,
        modalidade=Modalidade.SINAIS_VITAIS,
        tipo=tipo,
        escore=float(np.clip(escore, 0.0, 1.0)),
        fonte=fonte,
        evidencia=evidencia,
        momento=pd.Timestamp(momento).to_pydatetime(),
    )


class _Gatilho:
    """Dispara uma vez por episódio.

    Dispara quando a condição se sustenta por `persistencia` amostras seguidas e
    só rearma depois de `rearme` amostras sem a condição. Amostra sem dado
    (`None`) não conta para nenhum dos lados.
    """

    def __init__(self, persistencia: int = 3, rearme: int = 15):
        self.persistencia = persistencia
        self.rearme = rearme
        self._seguidas = 0
        self._inativas = 0
        self.disparado = False

    def atualizar(self, ativo: bool | None) -> bool:
        if ativo is None:
            return False
        if ativo:
            self._seguidas += 1
            self._inativas = 0
            if not self.disparado and self._seguidas >= self.persistencia:
                self.disparado = True
                return True
            return False
        self._seguidas = 0
        self._inativas += 1
        if self._inativas >= self.rearme:
            self.disparado = False
        return False


class Detector(Protocol):
    nome: str

    def atualizar(self, paciente_id: str, momento: datetime, amostra: dict) -> list[Achado]: ...


# ----------------------------------------------------------------- artefato

class FiltroArtefato:
    """Troca valores implausíveis por NaN e avisa quando um sensor fica mudo.

    O aviso `sensor_sem_sinal` é **técnico**, não clínico: diz à equipe que o
    monitoramento daquele sinal está cego — o que, para um sistema de alerta,
    também é um risco.
    """

    nome = "filtro-artefato"

    def __init__(self, sem_sinal_min: int = 10, sinais: Iterable[str] = ("spo2", "fc")):
        self.sem_sinal_min = sem_sinal_min
        self._mudo = {s: _Gatilho(persistencia=sem_sinal_min, rearme=3) for s in sinais}
        self.descartes = 0

    def limpar(self, amostra: dict) -> dict:
        limpa = {}
        for sinal, valor in amostra.items():
            lim = LIMITES_PLAUSIVEIS.get(sinal)
            if lim and _valido(valor) and not (lim[0] <= valor <= lim[1]):
                self.descartes += 1
                limpa[sinal] = np.nan
            else:
                limpa[sinal] = valor if _valido(valor) else np.nan
        return limpa

    def atualizar(self, paciente_id: str, momento, amostra: dict) -> list[Achado]:
        achados = []
        for sinal, gatilho in self._mudo.items():
            if gatilho.atualizar(not _valido(amostra.get(sinal))):
                achados.append(_achado(paciente_id, momento, "sensor_sem_sinal", 0.2, self.nome,
                                       sinal=sinal, minutos=self.sem_sinal_min, tecnico=True))
        return achados


# -------------------------------------------------------------------- NEWS2

def pontuar_news2(amostra: dict) -> dict[str, int]:
    """Pontos NEWS2 por parâmetro disponível (escala 1 de SpO2)."""
    pontos: dict[str, int] = {}
    fr, spo2, pas, fc = (amostra.get(k) for k in ("fr", "spo2", "pas", "fc"))
    if _valido(fr):
        pontos["fr"] = 3 if fr <= 8 else 1 if fr <= 11 else 0 if fr <= 20 else 2 if fr <= 24 else 3
    if _valido(spo2):
        pontos["spo2"] = 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    if _valido(pas):
        pontos["pas"] = 3 if pas <= 90 else 2 if pas <= 100 else 1 if pas <= 110 else 0 if pas <= 219 else 3
    if _valido(fc):
        pontos["fc"] = 3 if fc <= 40 else 1 if fc <= 50 else 0 if fc <= 90 else 1 if fc <= 110 else 2 if fc <= 130 else 3
    return pontos


# Parâmetro com 3 pontos → nome clínico do achado, conforme o lado da faixa.
_NOME_CRITICO = {
    ("fc", True): "taquicardia", ("fc", False): "bradicardia",
    ("pas", True): "hipertensao", ("pas", False): "hipotensao",
    ("fr", True): "taquipneia", ("fr", False): "bradipneia",
    ("spo2", False): "dessaturacao",
}
_REFERENCIA_ALTA = {"fc": 90, "pas": 150, "fr": 20}


class DetectorNEWS2:
    """Limiares clínicos absolutos, pela tabela do NEWS2.

    Dois gatilhos, como no protocolo: qualquer parâmetro isolado com 3 pontos
    ("red score"), e o escore agregado ≥ 5 (resposta urgente) ou ≥ 7
    (emergência).
    """

    nome = "news2"

    def __init__(self, persistencia: int = 3, rearme: int = 15):
        self._param = {s: _Gatilho(persistencia, rearme) for s in ("fr", "spo2", "pas", "fc")}
        self._agregado = {5: _Gatilho(persistencia, rearme), 7: _Gatilho(persistencia, rearme)}

    def atualizar(self, paciente_id: str, momento, amostra: dict) -> list[Achado]:
        pontos = pontuar_news2(amostra)
        if not pontos:
            return []
        total = sum(pontos.values())
        achados = []
        for sinal, gatilho in self._param.items():
            ativo = None if sinal not in pontos else pontos[sinal] == 3
            if gatilho.atualizar(ativo):
                alto = sinal in _REFERENCIA_ALTA and amostra[sinal] > _REFERENCIA_ALTA[sinal]
                tipo = _NOME_CRITICO[(sinal, alto)]
                achados.append(_achado(paciente_id, momento, tipo, 0.5 + total / 24, self.nome,
                                       sinal=sinal, valor=round(float(amostra[sinal]), 1),
                                       pontos=pontos, news2_parcial=total))
        for corte, gatilho in self._agregado.items():
            if gatilho.atualizar(total >= corte):
                tipo = "news2_alto" if corte >= 7 else "news2_medio"
                achados.append(_achado(paciente_id, momento, tipo, total / 12, self.nome,
                                       pontos=pontos, news2_parcial=total, corte=corte))
        return achados


# -------------------------------------------------------- desvio robusto

# Piso da escala (evita z enorme quando o sinal está "parado") e mudança
# absoluta mínima para valer como achado (evita alertar 2 bpm de variação).
ESCALA_MIN = {"fc": 3.0, "pas": 5.0, "pam": 4.0, "fr": 2.0, "spo2": 1.0}
DELTA_MIN = {"fc": 15.0, "pas": 20.0, "pam": 15.0, "fr": 6.0, "spo2": 4.0}
# Direções clinicamente relevantes: SpO2 só preocupa caindo.
DIRECOES = {"fc": (+1, -1), "pas": (+1, -1), "pam": (+1, -1), "fr": (+1, -1), "spo2": (-1,)}


class DetectorDesvioRobusto:
    """Desvio em relação ao basal do próprio paciente (mediana e MAD em janela móvel).

    O basal é a janela `[t - janela - atraso, t - atraso]`: o `atraso` impede
    que o início do evento contamine a própria referência. Mediana e MAD em vez
    de média e desvio-padrão porque um artefato isolado não desloca a mediana.
    """

    nome = "desvio-robusto"

    def __init__(self, janela: int = 120, atraso: int = 10, min_amostras: int = 60,
                 z_limiar: float = 4.0, persistencia: int = 4, rearme: int = 20,
                 sinais: Iterable[str] = ("fc", "pas", "fr", "spo2")):
        self.janela, self.atraso, self.min_amostras, self.z_limiar = janela, atraso, min_amostras, z_limiar
        self.sinais = tuple(sinais)
        self._hist = {s: deque(maxlen=janela + atraso) for s in self.sinais}
        self._gat = {(s, d): _Gatilho(persistencia, rearme) for s in self.sinais for d in DIRECOES[s]}

    def _basal(self, sinal: str) -> tuple[float, float] | None:
        valores = [v for v in list(self._hist[sinal])[: -self.atraso or None] if _valido(v)]
        if len(valores) < self.min_amostras:
            return None
        arr = np.asarray(valores)
        med = float(np.median(arr))
        escala = max(1.4826 * float(np.median(np.abs(arr - med))), ESCALA_MIN[sinal])
        return med, escala

    def atualizar(self, paciente_id: str, momento, amostra: dict) -> list[Achado]:
        achados = []
        for sinal in self.sinais:
            valor = amostra.get(sinal)
            basal = self._basal(sinal)
            self._hist[sinal].append(valor if _valido(valor) else np.nan)
            if basal is None or not _valido(valor):
                for d in DIRECOES[sinal]:
                    self._gat[(sinal, d)].atualizar(None)
                continue
            med, escala = basal
            z = (valor - med) / escala
            for d in DIRECOES[sinal]:
                ativo = d * z >= self.z_limiar and abs(valor - med) >= DELTA_MIN[sinal]
                if self._gat[(sinal, d)].atualizar(ativo):
                    tipo = f"{sinal}_{'acima' if d > 0 else 'abaixo'}_do_basal"
                    achados.append(_achado(paciente_id, momento, tipo, abs(z) / 10, self.nome,
                                           sinal=sinal, valor=round(float(valor), 1),
                                           basal=round(med, 1), z=round(float(z), 1)))
        return achados


# ------------------------------------------------------- multivariado (IF)

class DetectorMultivariado:
    """Isolation Forest sobre o vetor de sinais, treinado no basal do próprio paciente.

    Aprende o "normal" deste paciente nas primeiras `aquecimento` amostras e é
    re-treinado a cada `reajuste` amostras sobre a janela recente, para
    acompanhar mudanças lentas e esperadas (sedação, sono, titulação de droga).
    O limiar é um quantil baixo dos escores de treino: uma amostra precisa ser
    mais isolável que 99,5% do basal recente.

    Explicabilidade: o achado lista os sinais que mais se afastaram do basal
    (z robusto por sinal), para que a equipe saiba *por que* o vetor é anômalo.
    """

    nome = "isolation-forest"
    SINAIS = ("fc", "pas", "pad", "fr", "spo2")

    def __init__(self, aquecimento: int = 180, janela: int = 240, reajuste: int = 60,
                 quantil: float = 0.005, persistencia: int = 5, rearme: int = 30,
                 semente: int = SEED):
        self.aquecimento, self.reajuste, self.quantil = aquecimento, reajuste, quantil
        self.semente = semente
        self._buffer: deque[np.ndarray] = deque(maxlen=janela)
        self._ultimo = {s: np.nan for s in self.SINAIS}
        self._modelo = None
        self._limiar = None
        self._ref = None
        self._desde_ajuste = 0
        self._gatilho = _Gatilho(persistencia, rearme)

    def _vetor(self, amostra: dict) -> np.ndarray | None:
        # carry-forward curto: um minuto sem PA não deve cegar o detector inteiro
        for s in self.SINAIS:
            if _valido(amostra.get(s)):
                self._ultimo[s] = float(amostra[s])
        vetor = np.array([self._ultimo[s] for s in self.SINAIS])
        return None if np.isnan(vetor).any() else vetor

    def _ajustar(self) -> None:
        from sklearn.ensemble import IsolationForest

        X = np.vstack(self._buffer)
        self._modelo = IsolationForest(n_estimators=100, random_state=self.semente).fit(X)
        self._limiar = float(np.quantile(self._modelo.score_samples(X), self.quantil))
        med = np.median(X, axis=0)
        self._ref = (med, np.maximum(1.4826 * np.median(np.abs(X - med), axis=0),
                                     [ESCALA_MIN.get(s, 2.0) for s in self.SINAIS]))
        self._desde_ajuste = 0

    def atualizar(self, paciente_id: str, momento, amostra: dict) -> list[Achado]:
        vetor = self._vetor(amostra)
        if vetor is None:
            self._gatilho.atualizar(None)
            return []
        achados = []
        if self._modelo is not None:
            escore = float(self._modelo.score_samples(vetor.reshape(1, -1))[0])
            if self._gatilho.atualizar(escore < self._limiar):
                med, esc = self._ref
                z = (vetor - med) / esc
                ordem = np.argsort(-np.abs(z))[:3]
                contrib = {self.SINAIS[i]: round(float(z[i]), 1) for i in ordem}
                achados.append(_achado(
                    paciente_id, momento, "padrao_multivariado",
                    0.5 + (self._limiar - escore) * 5, self.nome,
                    escore_if=round(escore, 3), limiar=round(self._limiar, 3),
                    principais_desvios=contrib,
                    valores={s: round(float(v), 1) for s, v in zip(self.SINAIS, vetor)}))
        self._buffer.append(vetor)
        self._desde_ajuste += 1
        pronto = len(self._buffer) >= min(self.aquecimento, self._buffer.maxlen)
        if pronto and (self._modelo is None or self._desde_ajuste >= self.reajuste):
            self._ajustar()
        return achados


# ------------------------------------------------------------------ monitor

def detectores_padrao(params_if: dict | None = None) -> list[Detector]:
    return [DetectorNEWS2(), DetectorDesvioRobusto(), DetectorMultivariado(**(params_if or {}))]


class MonitorSinaisVitais:
    """Orquestra filtro + detectores para um paciente."""

    def __init__(self, paciente_id: str, detectores: list[Detector] | None = None):
        self.paciente_id = paciente_id
        self.filtro = FiltroArtefato()
        self.detectores = detectores if detectores is not None else detectores_padrao()

    def processar(self, momento, amostra: dict) -> list[Achado]:
        limpa = self.filtro.limpar(amostra)
        achados = self.filtro.atualizar(self.paciente_id, momento, limpa)
        for detector in self.detectores:
            achados.extend(detector.atualizar(self.paciente_id, momento, limpa))
        return achados

    def processar_serie(self, df: pd.DataFrame) -> list[Achado]:
        achados: list[Achado] = []
        for momento, linha in zip(df.index, df.to_dict("records")):
            achados.extend(self.processar(momento, linha))
        return achados


def achados_para_df(achados: list[Achado]) -> pd.DataFrame:
    if not achados:
        return pd.DataFrame(columns=["momento", "tipo", "fonte", "escore", "evidencia"])
    return pd.DataFrame([{"momento": pd.Timestamp(a.momento), "tipo": a.tipo, "fonte": a.fonte,
                          "escore": round(a.escore, 2), "evidencia": a.evidencia} for a in achados])
