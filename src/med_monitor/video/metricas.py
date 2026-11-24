"""Ângulos articulares por quadro e métricas por repetição.

Os ângulos são calculados no plano da imagem a partir dos keypoints. Por isso
cada exercício é analisado na câmera que enxerga o plano do movimento (ver
`dados.rehab24.camera_ideal`): a abdução do braço na câmera de frente, o
agachamento na de lado.

| Ângulo | Definição | Uso |
|---|---|---|
| `abducao_ombro_{d,e}` | entre o tronco (ombro→quadril) e o braço (ombro→cotovelo); 0° braço junto ao corpo, 90° na horizontal | amplitude da abdução |
| `cotovelo_{d,e}` | ombro–cotovelo–punho; 180° estendido | compensação dobrando o cotovelo |
| `joelho_{d,e}` | quadril–joelho–tornozelo; 180° estendido | profundidade do agachamento |
| `quadril_{d,e}` | ombro–quadril–joelho | flexão de quadril |
| `inclinacao_tronco` | desvio do eixo meio-ombro→meio-quadril em relação à vertical | compensação lateral ou à frente |
| `desnivel_ombros` | inclinação da linha entre os ombros | elevação compensatória do ombro |

Por repetição: mínimo, máximo e amplitude (p95 − p5) de cada ângulo, duração,
fração de quadros com pose válida e **suavidade** (quantas vezes o ângulo
principal muda de direção — movimento hesitante ou em "degraus").
"""

from __future__ import annotations

import warnings

import numpy as np

from .pose import SeriePose


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ângulo em `b` (graus) formado por a–b–c, vetorizado em (T, 2)."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _inclinacao(de: np.ndarray, para: np.ndarray) -> np.ndarray:
    """Ângulo (graus) do vetor de→para em relação à vertical da imagem (0° = alinhado)."""
    v = para - de
    return np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1]) + 1e-9))


def angulos(s: SeriePose) -> dict[str, np.ndarray]:
    p = s.ponto
    meio_ombro = (p("ombro_e") + p("ombro_d")) / 2
    meio_quadril = (p("quadril_e") + p("quadril_d")) / 2
    saida = {}
    for lado, L in (("d", "d"), ("e", "e")):
        ombro, cotovelo, punho = p(f"ombro_{L}"), p(f"cotovelo_{L}"), p(f"punho_{L}")
        quadril, joelho, tornozelo = p(f"quadril_{L}"), p(f"joelho_{L}"), p(f"tornozelo_{L}")
        saida[f"abducao_ombro_{lado}"] = angulo(quadril, ombro, cotovelo)
        saida[f"cotovelo_{lado}"] = angulo(ombro, cotovelo, punho)
        saida[f"joelho_{lado}"] = angulo(quadril, joelho, tornozelo)
        saida[f"quadril_{lado}"] = angulo(ombro, quadril, joelho)
    saida["inclinacao_tronco"] = _inclinacao(meio_quadril, meio_ombro)
    v = p("ombro_d") - p("ombro_e")
    saida["desnivel_ombros"] = np.degrees(np.arctan2(np.abs(v[:, 1]), np.abs(v[:, 0]) + 1e-9))
    return saida


# Ângulo principal de cada exercício (para suavidade e para o gráfico do relatório)
ANGULO_PRINCIPAL = {1: "abducao_ombro_d", 6: "joelho_media"}


def _suavizar(x: np.ndarray, janela: int = 5) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < janela:
        return x
    preenchido = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    return np.convolve(preenchido, np.ones(janela) / janela, mode="same")


def mudancas_de_direcao(x: np.ndarray, limiar_graus: float = 3.0) -> int:
    """Quantas vezes o ângulo inverte o sentido com amplitude ≥ limiar (histerese)."""
    x = _suavizar(x)
    if len(x) < 3 or np.all(np.isnan(x)):
        return 0
    # `ref` é o ponto de partida até a primeira direção se firmar; depois, o extremo atual
    mudancas, direcao, ref = 0, 0, x[0]
    for v in x[1:]:
        if direcao == 0:
            if abs(v - ref) > limiar_graus:
                direcao, ref = (1 if v > ref else -1), v
        elif direcao == 1:
            if v > ref:
                ref = v
            elif v < ref - limiar_graus:
                mudancas, direcao, ref = mudancas + 1, -1, v
        else:
            if v < ref:
                ref = v
            elif v > ref + limiar_graus:
                mudancas, direcao, ref = mudancas + 1, 1, v
    return mudancas


def metricas_repeticao(s: SeriePose, angs: dict[str, np.ndarray], inicio: int, fim: int,
                       exercicio: int) -> dict[str, float]:
    sel = (s.quadros >= inicio) & (s.quadros <= fim)
    # s.fps é o fps efetivo da série (vídeo / passo); os quadros são do vídeo original
    m: dict[str, float] = {"duracao_s": (fim - inicio + 1) / (s.fps * _passo(s))}
    m["pose_valida"] = float(np.mean(~np.isnan(angs["inclinacao_tronco"][sel]))) if sel.any() else 0.0
    trecho = {k: v[sel] for k, v in angs.items()}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # quadros sem nenhum joelho visível
        trecho["joelho_media"] = np.nanmean(np.vstack([trecho["joelho_d"], trecho["joelho_e"]]), axis=0) \
            if sel.any() else np.array([])
    trecho["assimetria_joelhos"] = np.abs(trecho["joelho_d"] - trecho["joelho_e"])
    for nome, x in trecho.items():
        x = x[~np.isnan(x)]
        if len(x) < 3:
            m[f"{nome}_min"] = m[f"{nome}_max"] = m[f"{nome}_amp"] = np.nan
            continue
        m[f"{nome}_min"] = float(np.percentile(x, 5))
        m[f"{nome}_max"] = float(np.percentile(x, 95))
        m[f"{nome}_amp"] = m[f"{nome}_max"] - m[f"{nome}_min"]
    principal = ANGULO_PRINCIPAL.get(exercicio)
    m["mudancas_direcao"] = float(mudancas_de_direcao(trecho[principal])) if principal in trecho else np.nan
    return m


def _passo(s: SeriePose) -> int:
    return int(np.median(np.diff(s.quadros))) if len(s.quadros) > 1 else 1
