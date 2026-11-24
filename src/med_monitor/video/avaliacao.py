"""Detecção de execução incorreta por repetição, avaliada contra o rótulo do fisioterapeuta.

Para cada exercício avaliado:

1. pose de todos os vídeos (cache) → ângulos → métricas por repetição;
2. **regressão logística** sobre as métricas do exercício, validada com
   **uma pessoa deixada de fora por vez** (leave-one-person-out): o modelo
   nunca é testado em alguém que viu no treino — é o cenário real, um
   paciente novo;
3. AUC por métrica isolada, para entender *o que* o fisioterapeuta chamou de
   incorreto em cada exercício;
4. explicação por repetição: as métricas mais distantes das repetições
   corretas de *outras* pessoas (z robusto), em linguagem de relatório.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SEED, VIDEO_RESULTS
from ..dados import rehab24
from . import metricas, objetos, pose

FEATURES = {
    1: ["abducao_ombro_d_max", "abducao_ombro_d_amp", "cotovelo_d_min", "inclinacao_tronco_max",
        "desnivel_ombros_max", "abducao_ombro_e_max", "duracao_s", "mudancas_direcao"],
    6: ["joelho_media_min", "joelho_media_amp", "quadril_d_min", "quadril_e_min", "inclinacao_tronco_max",
        "assimetria_joelhos_max", "duracao_s", "mudancas_direcao"],
}
ROTULO_FEATURE = {
    "abducao_ombro_d_max": "elevação máxima do braço", "abducao_ombro_d_amp": "amplitude da abdução",
    "cotovelo_d_min": "extensão do cotovelo", "inclinacao_tronco_max": "inclinação do tronco",
    "desnivel_ombros_max": "desnível dos ombros", "abducao_ombro_e_max": "elevação do braço oposto",
    "duracao_s": "duração da repetição", "mudancas_direcao": "hesitações no movimento",
    "joelho_media_min": "flexão máxima dos joelhos (ângulo mínimo)", "joelho_media_amp": "amplitude dos joelhos",
    "quadril_d_min": "flexão do quadril (direito)", "quadril_e_min": "flexão do quadril (esquerdo)",
    "assimetria_joelhos_max": "assimetria entre os joelhos",
}
UNIDADE = {"duracao_s": "s", "mudancas_direcao": ""}


def tabela_repeticoes(exercicios=rehab24.EXERCICIOS_AVALIADOS, passo: int = 2, modelo=None,
                      so_baixados: bool = True) -> pd.DataFrame:
    reps = rehab24.repeticoes()
    linhas = []
    for video_id, camera in rehab24.selecao(exercicios):
        arquivo = rehab24.caminho_video(video_id, camera)
        if not arquivo.exists():
            if so_baixados:
                continue
            rehab24.baixar_videos([video_id], camera)
        modelo = modelo or pose.modelo_pose()
        serie = pose.extrair_com_cache(arquivo, passo=passo, modelo=modelo)
        angs = metricas.angulos(serie)
        na_area = objetos.pessoa_na_area(serie)
        visivel = objetos.pessoa_visivel(serie)
        for r in (r for r in reps if r.video_id == video_id):
            m = metricas.metricas_repeticao(serie, angs, r.quadro_inicio, r.quadro_fim, r.exercicio)
            no_trecho = (serie.quadros >= r.quadro_inicio) & (serie.quadros <= r.quadro_fim)
            m["fracao_pessoa_visivel"] = float(np.mean(visivel[no_trecho])) if no_trecho.any() else np.nan
            m["fracao_pessoa_na_area"] = float(np.mean(na_area[no_trecho])) if no_trecho.any() else np.nan
            m["pessoa_extra_rotulo"] = r.pessoa_extra(camera)
            linhas.append({"video": video_id, "camera": camera, "pessoa": r.pessoa, "exercicio": r.exercicio,
                           "repeticao": r.numero, "inicio": r.quadro_inicio, "fim": r.quadro_fim,
                           "orientacao": r.orientacao_cam17, "correta": r.correta, **m})
    return pd.DataFrame(linhas)


def _pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def avaliar_exercicio(df: pd.DataFrame, exercicio: int) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    d = df[(df.exercicio == exercicio) & (df.pose_valida >= 0.5)].reset_index(drop=True)
    feats = FEATURES[exercicio]
    X, y_incorreta, grupos = d[feats].to_numpy(float), (~d.correta).to_numpy(int), d.pessoa.to_numpy()
    prob = np.zeros(len(d))
    for treino, teste in LeaveOneGroupOut().split(X, y_incorreta, grupos):
        if len(set(y_incorreta[treino])) < 2:
            prob[teste] = y_incorreta[treino].mean()
            continue
        prob[teste] = _pipeline().fit(X[treino], y_incorreta[treino]).predict_proba(X[teste])[:, 1]
    pred = prob >= 0.5
    vp = int(np.sum(pred & (y_incorreta == 1)))
    univariada = {}
    for f in feats:
        x = d[f].to_numpy(float)
        ok = ~np.isnan(x)
        if ok.sum() > 10 and len(set(y_incorreta[ok])) == 2:
            auc = roc_auc_score(y_incorreta[ok], x[ok])
            univariada[f] = round(float(max(auc, 1 - auc)), 3)
    por_pessoa = {}
    for pessoa in sorted(set(grupos)):
        sel = grupos == pessoa
        if len(set(y_incorreta[sel])) == 2:
            por_pessoa[int(pessoa)] = round(float(roc_auc_score(y_incorreta[sel], prob[sel])), 3)
    return {
        "exercicio": int(exercicio), "nome": rehab24.EXERCICIOS[int(exercicio)],
        "repeticoes": int(len(d)), "incorretas": int(y_incorreta.sum()), "pessoas": int(len(set(grupos))),
        "videos": int(d.video.nunique()),
        "auc": round(float(roc_auc_score(y_incorreta, prob)), 3),
        # ordenação das repetições dentro de cada paciente — o uso clínico real
        "auc_por_pessoa": por_pessoa,
        "auc_mediana_por_pessoa": round(float(np.median(list(por_pessoa.values()))), 3) if por_pessoa else None,
        "acuracia_balanceada": round(float(balanced_accuracy_score(y_incorreta, pred)), 3),
        "sensibilidade": round(vp / max(1, int(y_incorreta.sum())), 3),
        "especificidade": round(int(np.sum(~pred & (y_incorreta == 0))) / max(1, int((y_incorreta == 0).sum())), 3),
        "auc_por_metrica": dict(sorted(univariada.items(), key=lambda kv: -kv[1])),
        "probabilidades": {f"{v}#{r}": round(float(p), 3) for v, r, p in zip(d.video, d.repeticao, prob)},
    }


def referencia_correta(df: pd.DataFrame, exercicio: int, excluir_pessoa: int | None = None) -> pd.DataFrame:
    """Mediana e escala robusta das repetições **corretas** de outras pessoas."""
    d = df[(df.exercicio == exercicio) & df.correta & (df.pose_valida >= 0.5)]
    if excluir_pessoa is not None:
        d = d[d.pessoa != excluir_pessoa]
    feats = FEATURES[exercicio]
    med = d[feats].median()
    mad = (d[feats] - med).abs().median() * 1.4826
    return pd.DataFrame({"mediana": med, "escala": mad.clip(lower=1e-6)})


def explicar(linha: pd.Series, ref: pd.DataFrame, n: int = 2, z_min: float = 1.5) -> list[str]:
    """As métricas mais fora do padrão de execução correta, em frases."""
    z = ((linha[ref.index] - ref.mediana) / ref.escala).dropna()
    frases = []
    for f in z.abs().sort_values(ascending=False).index[:n]:
        if abs(z[f]) < z_min:
            break
        un = UNIDADE.get(f, "°")
        direcao = "acima" if z[f] > 0 else "abaixo"
        frases.append(f"{ROTULO_FEATURE.get(f, f)} {linha[f]:.0f}{un} — {direcao} do padrão correto "
                      f"({ref.mediana[f]:.0f}{un}, z={z[f]:+.1f})".replace(".", ","))
    return frases


def avaliar_pessoas_extras(df: pd.DataFrame, limiar_fracao: float = 0.3) -> dict:
    """Detector de outra pessoa visível vs rótulo do dataset.

    O rótulo vai de 0 (ninguém) a 3 (grande parte de outra pessoa visível); 2 é
    "parte perceptível", como um braço inteiro na borda. O YOLO detecta
    *pessoas*, não membros soltos — por isso o desempenho é medido nos dois
    cortes. A "pessoa na área do paciente" (evento clínico do relatório) é
    reportada à parte: o dataset não a rotula.
    """
    d = df.dropna(subset=["fracao_pessoa_visivel"])
    pred = d.fracao_pessoa_visivel >= limiar_fracao

    def medidas(real: pd.Series) -> dict:
        vp, fp = int((pred & real).sum()), int((pred & ~real).sum())
        fn, vn = int((~pred & real).sum()), int((~pred & ~real).sum())
        return {"positivos": int(real.sum()), "sensibilidade": round(vp / max(1, vp + fn), 3),
                "especificidade": round(vn / max(1, vn + fp), 3), "precisao": round(vp / max(1, vp + fp), 3)}

    return {"repeticoes": int(len(d)), "limiar_fracao_quadros": limiar_fracao,
            "pessoa_inteira": medidas(d.pessoa_extra_rotulo >= 3),
            "parte_perceptivel_ou_mais": medidas(d.pessoa_extra_rotulo >= 2),
            "fracao_por_rotulo": {int(k): round(float(v), 3)
                                  for k, v in d.groupby("pessoa_extra_rotulo").fracao_pessoa_visivel.mean().items()},
            "repeticoes_com_pessoa_na_area": int((d.fracao_pessoa_na_area >= limiar_fracao).sum())}


def avaliar(df: pd.DataFrame) -> dict:
    return {"por_exercicio": [avaliar_exercicio(df, ex) for ex in sorted(df.exercicio.unique())
                              if ex in FEATURES],
            "pessoas_extras": avaliar_pessoas_extras(df)}


def tabela_markdown(res: dict) -> str:
    linhas = ["| Exercício | Vídeos · pessoas | Repetições (incorretas) | AUC global | AUC mediana por pessoa | "
              "Acurácia balanceada | Sensibilidade | Especificidade | Métricas mais discriminativas |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in res["por_exercicio"]:
        top = ", ".join(f"{ROTULO_FEATURE.get(f, f)} ({a:.2f})" for f, a in list(e["auc_por_metrica"].items())[:3])
        linhas.append(f"| {e['exercicio']}. {e['nome']} | {e['videos']} · {e['pessoas']} | {e['repeticoes']} "
                      f"({e['incorretas']}) | {e['auc']:.2f} | {e['auc_mediana_por_pessoa']:.2f} | "
                      f"{e['acuracia_balanceada']:.0%} | "
                      f"{e['sensibilidade']:.0%} | {e['especificidade']:.0%} | {top} |")
    pe = res.get("pessoas_extras")
    if pe:
        linhas += ["", "Outra pessoa visível (YOLOv8, confiança ≥ 0,6, em ≥ 30% dos quadros da repetição) vs rótulo do dataset:",
                   "", "| Corte do rótulo | Positivos | Sensibilidade | Especificidade | Precisão |", "|---|---|---|---|---|"]
        for nome, chave in (("grande parte de outra pessoa (3)", "pessoa_inteira"),
                            ("parte perceptível ou mais (≥ 2)", "parte_perceptivel_ou_mais")):
            m = pe[chave]
            linhas.append(f"| {nome} | {m['positivos']}/{pe['repeticoes']} | {m['sensibilidade']:.0%} | "
                          f"{m['especificidade']:.0%} | {m['precisao']:.0%} |")
        linhas += ["", "Fração média de quadros com outra pessoa visível, por rótulo: "
                   + " · ".join(f"{k}: {v:.0%}" for k, v in pe["fracao_por_rotulo"].items()),
                   f"Repetições com outra pessoa **na área do paciente** (evento clínico, sem rótulo no dataset): "
                   f"{pe['repeticoes_com_pessoa_na_area']}"]
    return "\n".join(linhas) + "\n"


def salvar(df: pd.DataFrame, res: dict, destino: Path = VIDEO_RESULTS) -> tuple[Path, Path, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    csv = destino / "metricas_repeticoes.csv"
    df.to_csv(csv, index=False)
    js = destino / "avaliacao_execucao.json"
    js.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    md = destino / "avaliacao_execucao.md"
    md.write_text("# Avaliação — execução incorreta (REHAB24-6, validação por pessoa)\n\n" + tabela_markdown(res),
                  encoding="utf-8")
    return csv, js, md
