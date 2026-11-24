"""Relatório automático de uma sessão de fisioterapia gravada.

Para um vídeo: pose → repetições → probabilidade de execução incorreta (modelo
treinado **em outras pessoas**) → explicação de cada desvio → eventos de área
crítica e queda → objetos em cena → comparação com a sessão anterior do mesmo
paciente. Sai em Markdown + gráfico, e como `Achado`s para a fusão.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import VIDEO_RESULTS
from ..dados import rehab24
from ..domain import Achado, Modalidade
from . import avaliacao, metricas, objetos, pose

LIMIAR_INCORRETA = 0.5
QUEDA_AMPLITUDE = 0.20     # amplitude 20% menor que a da sessão anterior → achado de movimentação


@dataclass
class RelatorioSessao:
    video: str
    camera: int
    pessoa: int
    exercicio: int
    repeticoes: list[dict]
    eventos: list[dict]
    objetos: dict[str, float]
    comparacao_anterior: dict | None
    achados: list[Achado] = field(default_factory=list)

    @property
    def incorretas(self) -> list[dict]:
        return [r for r in self.repeticoes if r["prob_incorreta"] >= LIMIAR_INCORRETA]

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "achados"}
        d["achados"] = [a.as_dict() for a in self.achados]
        return d


def _modelo_sem(df: pd.DataFrame, exercicio: int, pessoa: int):
    d = df[(df.exercicio == exercicio) & (df.pessoa != pessoa) & (df.pose_valida >= 0.5)]
    feats = avaliacao.FEATURES[exercicio]
    return avaliacao._pipeline().fit(d[feats].to_numpy(float), (~d.correta).to_numpy(int))


def _sessao_anterior(df: pd.DataFrame, video: str, exercicio: int, pessoa: int) -> pd.DataFrame | None:
    da_pessoa = sorted(df[(df.pessoa == pessoa) & (df.exercicio == exercicio)].video.unique())
    if video not in da_pessoa or da_pessoa.index(video) == 0:
        return None
    return df[df.video == da_pessoa[da_pessoa.index(video) - 1]]


def gerar(video_id: str, df: pd.DataFrame, paciente_id: str | None = None, com_objetos: bool = True,
          momento: datetime | None = None) -> RelatorioSessao:
    """`df` é a tabela de métricas de todas as repetições (`avaliacao.tabela_repeticoes`)."""
    sessao = df[df.video == video_id].sort_values("repeticao")
    if sessao.empty:
        raise KeyError(f"{video_id} sem métricas (rode `med-monitor video avaliar` antes)")
    camera, pessoa, exercicio = int(sessao.camera.iloc[0]), int(sessao.pessoa.iloc[0]), int(sessao.exercicio.iloc[0])
    paciente_id = paciente_id or f"REHAB-P{pessoa:02d}"
    momento = momento or datetime.now(timezone.utc)
    feats = avaliacao.FEATURES[exercicio]

    modelo = _modelo_sem(df, exercicio, pessoa)
    ref = avaliacao.referencia_correta(df, exercicio, excluir_pessoa=pessoa)
    probs = modelo.predict_proba(sessao[feats].to_numpy(float))[:, 1]
    repeticoes = []
    for (_, linha), p in zip(sessao.iterrows(), probs):
        repeticoes.append({"repeticao": int(linha.repeticao), "inicio_s": round(linha.inicio / 30, 1),
                           "fim_s": round(linha.fim / 30, 1), "prob_incorreta": round(float(p), 3),
                           "rotulo_fisioterapeuta": "correta" if linha.correta else "incorreta",
                           "desvios": avaliacao.explicar(linha, ref) if p >= LIMIAR_INCORRETA else []})

    arquivo = rehab24.caminho_video(video_id, camera)
    serie = pose.extrair_com_cache(arquivo)
    eventos = [e.as_dict() for e in objetos.eventos_pessoas_extras(serie) + objetos.queda(serie)]
    cena = objetos.objetos_em_cena(arquivo) if com_objetos and arquivo.exists() else {}

    comparacao = None
    anterior = _sessao_anterior(df, video_id, exercicio, pessoa)
    principal = {1: "abducao_ombro_d_amp", 6: "joelho_media_amp"}[exercicio]
    if anterior is not None:
        atual_amp, ant_amp = float(sessao[principal].median()), float(anterior[principal].median())
        comparacao = {"sessao_anterior": str(anterior.video.iloc[0]), "metrica": principal,
                      "atual": round(atual_amp, 1), "anterior": round(ant_amp, 1),
                      "variacao": round((atual_amp - ant_amp) / ant_amp, 3) if ant_amp else None}

    rel = RelatorioSessao(video_id, camera, pessoa, exercicio, repeticoes, eventos, cena, comparacao)
    rel.achados = _achados(rel, paciente_id, momento)
    return rel


def _achados(rel: RelatorioSessao, paciente_id: str, momento: datetime) -> list[Achado]:
    saida = []
    incorretas = rel.incorretas
    if incorretas:
        fracao = len(incorretas) / len(rel.repeticoes)
        escore = float(np.clip(fracao * np.mean([r["prob_incorreta"] for r in incorretas]) + 0.3, 0, 0.79))
        # "amplitude da abdução 92° — abaixo do padrão…" → "amplitude da abdução abaixo do padrão"
        desvios = sorted({re.sub(r" -?\d[\d,]*\S* — ", " ", d).split(" (")[0]
                          for r in incorretas for d in r["desvios"]})
        saida.append(Achado(paciente_id, Modalidade.VIDEO, "desvio_execucao", escore, "yolov8-pose",
                            evidencia={"exercicio": rehab24.EXERCICIOS[rel.exercicio],
                                       "repeticoes_incorretas": [r["repeticao"] for r in incorretas],
                                       "total_repeticoes": len(rel.repeticoes), "desvios": desvios},
                            momento=momento))
    for e in rel.eventos:
        if e["tipo"] == "queda":
            saida.append(Achado(paciente_id, Modalidade.VIDEO, "queda", 0.95, "yolov8-pose",
                                evidencia=e, momento=momento))
    c = rel.comparacao_anterior
    if c and c["variacao"] is not None and c["variacao"] <= -QUEDA_AMPLITUDE:
        saida.append(Achado(paciente_id, Modalidade.MOVIMENTACAO, "amplitude_reduzida",
                            float(np.clip(0.5 + abs(c["variacao"]), 0, 0.79)), "yolov8-pose", evidencia=c,
                            momento=momento))
    return saida


def grafico(rel: RelatorioSessao, destino: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    serie = pose.extrair_com_cache(rehab24.caminho_video(rel.video, rel.camera))
    angs = metricas.angulos(serie)
    if rel.exercicio == 6:
        y = np.nanmean(np.vstack([angs["joelho_d"], angs["joelho_e"]]), axis=0)
        rotulo = "ângulo médio dos joelhos (°)"
    else:
        y = angs["abducao_ombro_d"]
        rotulo = "abdução do ombro direito (°)"
    t = serie.quadros / 30
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(t, y, color="#333", lw=0.9)
    for r in rel.repeticoes:
        cor = "#d62728" if r["prob_incorreta"] >= LIMIAR_INCORRETA else "#2ca02c"
        ax.axvspan(r["inicio_s"], r["fim_s"], color=cor, alpha=0.15)
        marca = "✓" if r["rotulo_fisioterapeuta"] == "correta" else "✗"
        ax.text((r["inicio_s"] + r["fim_s"]) / 2, np.nanmax(y) * 1.02 if np.isfinite(np.nanmax(y)) else 0,
                f"{r['repeticao']}{marca}", ha="center", fontsize=8)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel(rotulo)
    ax.set_title(f"{rel.video} · {rehab24.EXERCICIOS[rel.exercicio]} · faixa vermelha = prevista incorreta, "
                 f"✓/✗ = rótulo do fisioterapeuta")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=110)
    plt.close(fig)
    return destino


def markdown(rel: RelatorioSessao) -> str:
    n, inc = len(rel.repeticoes), rel.incorretas
    acertos = sum((r["prob_incorreta"] >= LIMIAR_INCORRETA) == (r["rotulo_fisioterapeuta"] == "incorreta")
                  for r in rel.repeticoes)
    rotulos = np.array([r["rotulo_fisioterapeuta"] == "incorreta" for r in rel.repeticoes])
    ordem = None
    if 0 < rotulos.sum() < len(rotulos):
        from sklearn.metrics import roc_auc_score

        ordem = roc_auc_score(rotulos, [r["prob_incorreta"] for r in rel.repeticoes])
    linhas = [f"# Relatório da sessão {rel.video}", "",
              f"- **Exercício:** {rehab24.EXERCICIOS[rel.exercicio]} · câmera {rel.camera} · pessoa {rel.pessoa}",
              f"- **Repetições:** {n} · **previstas incorretas:** {len(inc)} "
              f"· concordância com o fisioterapeuta: {acertos}/{n}"
              + (f" · ordenação das repetições (AUC da sessão): {ordem:.2f}".replace(".", ",") if ordem is not None else ""),
              ""]
    if rel.comparacao_anterior:
        c = rel.comparacao_anterior
        linhas.append(f"- **Comparação com a sessão anterior** ({c['sessao_anterior']}): amplitude mediana "
                      f"{c['atual']:.0f}° vs {c['anterior']:.0f}° ({c['variacao']:+.0%})".replace(".", ","))
    if rel.objetos:
        linhas.append("- **Em cena:** " + ", ".join(f"{k} ({v:.0%})" for k, v in list(rel.objetos.items())[:6]))
    if rel.eventos:
        linhas.append("- **Áreas críticas:** " + "; ".join(f"{e['tipo'].replace('_', ' ')} {e['inicio_s']}–{e['fim_s']} s"
                                                           f" ({e['detalhe']})" for e in rel.eventos[:5]))
    linhas += ["", "| Rep. | Intervalo | P(incorreta) | Fisioterapeuta | Desvios |", "|---|---|---|---|---|"]
    for r in rel.repeticoes:
        linhas.append(f"| {r['repeticao']} | {r['inicio_s']}–{r['fim_s']} s | {r['prob_incorreta']:.2f} | "
                      f"{r['rotulo_fisioterapeuta']} | {'; '.join(r['desvios']) or '—'} |")
    linhas += ["", f"![ângulo principal](sessao_{rel.video}.png)", ""]
    return "\n".join(linhas)


def salvar(rel: RelatorioSessao, destino: Path = VIDEO_RESULTS) -> tuple[Path, Path, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    md = destino / f"sessao_{rel.video}.md"
    md.write_text(markdown(rel), encoding="utf-8")
    js = destino / f"sessao_{rel.video}.json"
    js.write_text(json.dumps(rel.as_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    png = grafico(rel, destino / f"sessao_{rel.video}.png")
    return md, js, png
