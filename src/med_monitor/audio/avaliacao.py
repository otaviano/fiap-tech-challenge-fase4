"""Avaliação do detector acústico com consultas sintéticas de condição conhecida.

Para cada voz: o basal é uma fala normal (roteiro `basal_1`); as consultas
"atuais" usam **outros** roteiros (mesmo falante, outro conteúdo — como na
vida real) em cinco condições: normal, fadiga, fadiga leve, dispneia, dispneia
leve. Medimos, com e sem basal:

- **detecção**: a alteração esperada foi declarada?
- **confusão**: foi declarada a alteração errada?
- **falso alarme**: alguma alteração declarada numa fala normal?
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import AUDIO_RESULTS
from . import acustica
from .amostras import ROTEIROS, motor_padrao, sintetizar

VOZES = ("pt-BR-AntonioNeural", "pt-BR-FranciscaNeural", "pt-BR-DonatoNeural", "pt-BR-BrendaNeural")
ROTEIROS_ATUAIS = ("basal_2", "basal_3", "estavel")
CONDICOES = ("normal", "fadiga", "fadiga_leve", "dispneia", "dispneia_leve")
ESPERADO = {"normal": None, "fadiga": "fadiga_vocal", "fadiga_leve": "fadiga_vocal",
            "dispneia": "fala_ofegante", "dispneia_leve": "fala_ofegante"}


def avaliar(motor: str | None = None) -> dict:
    """`motor`: "azure" (vozes neurais) ou "espeak"; padrão: Azure se configurado."""
    motor = motor or motor_padrao()
    linhas = []
    for iv, voz in enumerate(VOZES):
        basal = acustica.extrair(sintetizar(ROTEIROS["basal_1"], "normal", voz, semente=iv, motor=motor))
        for ir, roteiro in enumerate(ROTEIROS_ATUAIS):
            for condicao in CONDICOES:
                f = acustica.extrair(sintetizar(ROTEIROS[roteiro], condicao, voz, semente=100 + iv * 10 + ir,
                                                motor=motor))
                for modo, ref in (("com basal", basal), ("sem basal", None)):
                    tipos = sorted(a.tipo for a in acustica.detectar(f, ref))
                    linhas.append({"voz": voz, "roteiro": roteiro, "condicao": condicao, "modo": modo,
                                   "declarado": tipos})
    resumo = {}
    for modo in ("com basal", "sem basal"):
        por_cond = {}
        for c in CONDICOES:
            casos = [l for l in linhas if l["modo"] == modo and l["condicao"] == c]
            esperado = ESPERADO[c]
            if esperado is None:
                por_cond[c] = {"n": len(casos), "falso_alarme": sum(bool(l["declarado"]) for l in casos)}
            else:
                por_cond[c] = {"n": len(casos),
                               "detectado": sum(esperado in l["declarado"] for l in casos),
                               "confusao": sum(any(t != esperado for t in l["declarado"]) for l in casos)}
        resumo[modo] = por_cond
    return {"motor": motor, "vozes": list(VOZES), "roteiros_atuais": list(ROTEIROS_ATUAIS), "resumo": resumo,
            "casos": linhas}


def tabela_markdown(res: dict) -> str:
    motor = {"azure": "vozes neurais do Azure Speech", "espeak": "espeak-ng"}.get(res.get("motor"), res.get("motor"))
    linhas = [f"{len(res['vozes'])} vozes ({motor}) × {len(res['roteiros_atuais'])} roteiros por condição "
              "(basal: outro roteiro, mesma voz, fala normal)", "",
              "| Condição | Com basal | Sem basal (referência populacional) |", "|---|---|---|"]
    for c in CONDICOES:
        celulas = []
        for modo in ("com basal", "sem basal"):
            d = res["resumo"][modo][c]
            if "falso_alarme" in d:
                celulas.append(f"{d['falso_alarme']}/{d['n']} falsos alarmes")
            else:
                celulas.append(f"{d['detectado']}/{d['n']} detectados · {d['confusao']} confusões")
        linhas.append(f"| {c} | " + " | ".join(celulas) + " |")
    return "\n".join(linhas) + "\n"


def salvar(res: dict, destino: Path = AUDIO_RESULTS) -> tuple[Path, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    js, md = destino / "avaliacao_acustica.json", destino / "avaliacao_acustica.md"
    js.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md.write_text("# Avaliação — alterações vocais (consultas sintéticas)\n\n" + tabela_markdown(res), encoding="utf-8")
    return js, md
