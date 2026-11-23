"""Termos críticos na fala do paciente — léxico local com tratamento de negação.

Roda sempre, com ou sem Azure:

- **sem Azure**, é a única leitura clínica do texto;
- **com Azure**, complementa o Text Analytics for Health: o léxico é do
  hospital (inclui coloquialismos como "coração disparado" e "fala enrolada"),
  e uma entidade que o Azure também marcou ganha corroboração.

**Negação** é tratada explicitamente — "não tenho falta de ar" não é achado.
Um termo é negado quando um marcador ("não", "nem", "nunca", "sem") aparece
até 3 palavras antes dele, na mesma oração. Termos que já contêm a negação
("não consigo respirar") não são afetados por ela.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

# categoria → (escore, termos). O escore reflete a urgência típica do sintoma relatado.
LEXICO: dict[str, tuple[float, tuple[str, ...]]] = {
    "risco_autolesao": (0.95, ("vontade de morrer", "quero morrer", "me matar", "acabar com tudo",
                                "nao aguento mais viver")),
    "respiratorio": (0.80, ("falta de ar", "sem ar", "nao consigo respirar", "dificuldade para respirar",
                            "dificuldade de respirar", "cansaco para respirar", "chiado no peito",
                            "sufocado", "sufocada", "sufocando", "puxando o ar")),
    "cardiovascular": (0.80, ("dor no peito", "aperto no peito", "peito apertado", "pressao no peito",
                              "coracao disparado", "palpitacao", "palpitacoes")),
    "neurologico": (0.80, ("desmaio", "desmaiei", "apaguei", "fala enrolada", "boca torta",
                           "formigamento", "perdi a forca", "nao sinto o braco", "nao sinto a perna",
                           "visao turva", "confuso", "confusa")),
    "fadiga": (0.55, ("cansaco", "cansado", "cansada", "exausto", "exausta", "sem forca", "fraqueza",
                      "sem energia", "muito sono")),
    "dor": (0.55, ("dor forte", "muita dor", "dor insuportavel", "doendo muito")),
    "geral": (0.45, ("febre", "calafrio", "tontura", "tonto", "tonta", "vomitei", "vomito", "nausea",
                     "sangramento", "sangrando")),
}

NEGADORES = {"nao", "nem", "nunca", "sem", "jamais", "nenhum", "nenhuma"}
SEPARADORES_ORACAO = re.compile(r"[.,;:!?]|\bmas\b|\bporem\b")


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento)


@dataclass(frozen=True)
class TermoEncontrado:
    termo: str
    categoria: str
    escore: float
    negado: bool
    trecho: str

    def as_dict(self) -> dict:
        return asdict(self)


def _negado(oracao_antes: str, termo: str) -> bool:
    if termo.startswith("nao "):
        return False
    palavras = oracao_antes.split()[-3:]
    return any(p in NEGADORES for p in palavras)


def encontrar(texto: str) -> list[TermoEncontrado]:
    norm = normalizar(texto)
    achados: list[TermoEncontrado] = []
    ocupado: list[tuple[int, int]] = []
    # termos mais longos primeiro: "nao consigo respirar" não deve virar também "respirar"
    candidatos = sorted(((t, cat, esc) for cat, (esc, termos) in LEXICO.items() for t in termos),
                        key=lambda x: -len(x[0]))
    for termo, categoria, escore in candidatos:
        for m in re.finditer(rf"\b{re.escape(termo)}\b", norm):
            if any(a < m.end() and m.start() < b for a, b in ocupado):
                continue
            ocupado.append((m.start(), m.end()))
            antes = SEPARADORES_ORACAO.split(norm[:m.start()])[-1]
            trecho = norm[max(0, m.start() - 40): m.end() + 20].strip()
            achados.append(TermoEncontrado(termo, categoria, escore, _negado(antes, termo), trecho))
    return sorted(achados, key=lambda t: norm.find(t.termo))
