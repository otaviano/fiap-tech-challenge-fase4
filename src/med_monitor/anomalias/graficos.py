"""Gráfico da série de sinais vitais com os achados e os eventos injetados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..domain import Achado

PAINEIS = (("fc", "FC (bpm)"), ("pas", "PAS (mmHg)"), ("fr", "FR (irpm)"), ("spo2", "SpO₂ (%)"))
CORES = {"news2": "#d62728", "desvio-robusto": "#1f77b4", "isolation-forest": "#9467bd",
         "filtro-artefato": "#7f7f7f"}


def plotar(serie: pd.DataFrame, achados: list[Achado], destino: Path, titulo: str,
           eventos: list | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .sinais_vitais import LIMITES_PLAUSIVEIS

    fig, eixos = plt.subplots(len(PAINEIS), 1, figsize=(14, 9), sharex=True)
    for ax, (sinal, rotulo) in zip(eixos, PAINEIS):
        lo, hi = LIMITES_PLAUSIVEIS[sinal]
        valores = serie[sinal].where(serie[sinal].between(lo, hi))   # artefato não é plotado
        ax.plot(serie.index, valores, lw=0.8, color="#333333")
        ax.set_ylabel(rotulo)
        ax.grid(alpha=0.3)
        for ev in eventos or []:
            cor = "#2ca02c" if ev.clinico else "#bbbbbb"
            ax.axvspan(ev.inicio, ev.fim, color=cor, alpha=0.15)
        for a in achados:
            sinal_achado = a.evidencia.get("sinal")
            if sinal_achado not in (None, sinal):
                continue
            ax.axvline(pd.Timestamp(a.momento), color=CORES.get(a.fonte, "black"), lw=1, alpha=0.8)

    eixos[0].set_title(titulo)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legenda = [Line2D([0], [0], color=c, lw=2, label=f) for f, c in CORES.items() if f != "filtro-artefato"]
    if eventos:
        legenda += [Patch(color="#2ca02c", alpha=0.3, label="evento injetado"),
                    Patch(color="#bbbbbb", alpha=0.5, label="artefato injetado")]
    eixos[0].legend(handles=legenda, loc="upper right", fontsize=8, ncol=len(legenda))
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=110)
    plt.close(fig)
    return destino
