"""Séries de sinais vitais sintéticas e injeção de anomalias rotuladas.

Dados reais de UTI não vêm com rótulo de "aqui houve uma anomalia". Para medir
recall, precisão e latência dos detectores, injetamos eventos conhecidos — com
início, fim e tipo — sobre um fundo, que pode ser:

- **real** (registros do PhysioNet): avalia o detector no ruído, nas derivas e
  nos artefatos de um monitor de verdade;
- **sintético** (`gerar_basal`): fundo controlado, usado nos testes e quando
  não há rede para baixar o PhysioNet.

Os eventos imitam padrões clínicos de deterioração e um artefato de sensor (que
o sistema deve *não* tratar como evento clínico).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .physionet import SINAIS

# Basal de um adulto estável e o desvio-padrão do ruído minuto a minuto.
BASAL = {"fc": 78.0, "pas": 122.0, "pad": 68.0, "pam": 86.0, "fr": 16.0, "spo2": 97.0}
RUIDO = {"fc": 2.0, "pas": 3.0, "pad": 2.0, "pam": 2.0, "fr": 1.0, "spo2": 0.6}


@dataclass(frozen=True)
class EventoInjetado:
    tipo: str
    inicio: pd.Timestamp
    fim: pd.Timestamp
    sinais: tuple[str, ...]
    clinico: bool = True        # False = artefato: não deveria virar alerta clínico

    def as_dict(self) -> dict:
        d = asdict(self)
        d["inicio"], d["fim"] = self.inicio.isoformat(), self.fim.isoformat()
        return d


# tipo → (deltas por sinal no platô, duração em min (mín, máx), rampa em min)
# Deltas são relativos ao valor local, exceto quando o sinal tem alvo absoluto.
PADROES: dict[str, dict] = {
    "taquicardia":    {"delta": {"fc": +38}, "dur": (20, 45), "rampa": 3},
    "bradicardia":    {"delta": {"fc": -32}, "dur": (15, 30), "rampa": 2},
    "hipotensao":     {"delta": {"pas": -38, "pad": -20, "pam": -26}, "dur": (20, 40), "rampa": 8},
    "dessaturacao":   {"alvo": {"spo2": 85.0}, "dur": (15, 30), "rampa": 10},
    "taquipneia":     {"delta": {"fr": +13}, "dur": (20, 40), "rampa": 5},
    # deterioração sutil e simultânea: cada sinal isolado fica perto do normal,
    # o conjunto é que é anômalo (padrão de sepse incipiente)
    "deterioracao_combinada": {"delta": {"fc": +20, "pas": -20, "pad": -10, "pam": -14,
                                         "fr": +6, "spo2": -3}, "dur": (40, 80), "rampa": 30},
}
TIPOS_CLINICOS = tuple(PADROES)

# Padrões usados só na demonstração (fora da avaliação): depressão respiratória por
# excesso de opioide — a frequência respiratória cai e a saturação acompanha.
PADROES_EXTRA: dict[str, dict] = {
    "depressao_respiratoria": {"alvo": {"fr": 7.0, "spo2": 88.0}, "delta": {"fc": -10}, "dur": (30, 45), "rampa": 8},
}


def gerar_basal(horas: float = 12, inicio: str = "2026-01-01 00:00", semente: int = 42,
                basal: dict | None = None) -> pd.DataFrame:
    """Série estável, minuto a minuto: deriva lenta (passeio aleatório amortecido) + ruído."""
    rng = np.random.default_rng(semente)
    n = int(horas * 60)
    indice = pd.date_range(inicio, periods=n, freq="1min", name="momento")
    base = {**BASAL, **(basal or {})}
    dados = {}
    for s in SINAIS:
        deriva = np.zeros(n)
        for i in range(1, n):
            deriva[i] = 0.98 * deriva[i - 1] + rng.normal(0, RUIDO[s] * 0.3)
        dados[s] = base[s] + deriva + rng.normal(0, RUIDO[s], n)
    df = pd.DataFrame(dados, index=indice)
    df["spo2"] = df["spo2"].clip(upper=100)
    return df


def _perfil(n: int, rampa: int) -> np.ndarray:
    """0 → 1 em `rampa` min, platô, 1 → 0 em `rampa` min."""
    perfil = np.ones(n)
    r = max(1, min(rampa, n // 2))
    perfil[:r] = np.linspace(0, 1, r, endpoint=False)
    perfil[-r:] = np.linspace(1, 0, r)
    return perfil


def injetar(df: pd.DataFrame, tipo: str, inicio_idx: int, duracao: int,
            rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, EventoInjetado]:
    """Aplica um evento ao DataFrame (cópia) e devolve o evento rotulado."""
    df = df.copy()
    if tipo == "artefato":
        # sensor de SpO2 solto: cai a zero por 1–3 min, junto com a FC de pulso
        n = duracao
        fatia = df.index[inicio_idx:inicio_idx + n]
        df.loc[fatia, "spo2"] = 0.0
        ev = EventoInjetado("artefato", fatia[0], fatia[-1], ("spo2",), clinico=False)
        return df, ev

    padrao = PADROES[tipo] if tipo in PADROES else PADROES_EXTRA[tipo]
    fatia = df.index[inicio_idx:inicio_idx + duracao]
    perfil = _perfil(len(fatia), padrao["rampa"])
    for sinal, delta in padrao.get("delta", {}).items():
        df.loc[fatia, sinal] = df.loc[fatia, sinal] + delta * perfil
    for sinal, alvo in padrao.get("alvo", {}).items():
        atual = df.loc[fatia, sinal].ffill().fillna(alvo)
        df.loc[fatia, sinal] = atual + (alvo - atual) * perfil
    df["spo2"] = df["spo2"].clip(upper=100)
    sinais = tuple(padrao.get("delta", {})) + tuple(padrao.get("alvo", {}))
    return df, EventoInjetado(tipo, fatia[0], fatia[-1], sinais)


def injetar_varios(df: pd.DataFrame, tipos: list[str], semente: int = 42,
                   aquecimento: int = 180, espaco: int = 90) -> tuple[pd.DataFrame, list[EventoInjetado]]:
    """Injeta os eventos em posições aleatórias, sem sobreposição e após o aquecimento.

    `aquecimento` reserva o início da série para os detectores aprenderem o
    basal do paciente; `espaco` é a distância mínima entre eventos, para que um
    não contamine a linha de base do seguinte.
    """
    rng = np.random.default_rng(semente)
    eventos: list[EventoInjetado] = []
    cursor = aquecimento
    livre = len(df) - cursor
    duracoes = [int(rng.integers(*PADROES[t]["dur"])) if t in PADROES else int(rng.integers(1, 4))
                for t in tipos]
    folga = livre - sum(duracoes) - espaco * len(tipos)
    if folga < 0:
        raise ValueError(f"série curta demais para {len(tipos)} eventos ({len(df)} min)")
    # distribui a folga aleatoriamente entre os intervalos
    cortes = np.sort(rng.integers(0, folga + 1, size=len(tipos)))
    gaps = np.diff(np.concatenate([[0], cortes]))
    for tipo, dur, gap in zip(tipos, duracoes, gaps):
        cursor += int(gap) + espaco
        df, ev = injetar(df, tipo, cursor, dur, rng)
        eventos.append(ev)
        cursor += dur
    return df, eventos
