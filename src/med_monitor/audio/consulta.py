"""Pipeline de uma consulta gravada: áudio → transcrição → texto → achados.

```
consulta.wav ─┬─▶ acústica (librosa) ─────────── vs basal ─▶ fala_ofegante / fadiga_vocal  (AUDIO)
              └─▶ Azure Speech to Text ──▶ texto ─┬─▶ léxico do hospital ─▶ termo_critico   (TEXTO)
                  (ou roteiro de referência)      └─▶ Azure Text Analytics ─▶ sentimento,
                                                      Text Analytics for Health ─▶ sintomas (TEXTO)
```

O relatório registra **quais serviços foram de fato usados** (`servicos`): sem
credenciais, o Azure aparece como "indisponível" e a transcrição como
"referência" — nunca se finge que a nuvem foi chamada.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..azure.language import AnaliseTexto, ClienteLanguage
from ..azure.speech import ClienteSpeech, Transcricao, transcricao_de_referencia
from ..config import ROOT
from ..domain import Achado, Modalidade
from . import acustica, termos

SENTIMENTO_NEGATIVO_MIN = 0.75
# O Text Analytics for Health às vezes marca como sintoma um estado bom ("dorme bem", "respirando melhor").
_POSITIVO = re.compile(r"\b(bem|melhor|normal|normais|tranquil\w*|[óo]tim\w*)\b", re.IGNORECASE)


def _relativo(caminho: Path | str | None) -> str | None:
    """Caminho relativo ao projeto quando possível: relatórios não devem expor o diretório de quem rodou."""
    if caminho is None:
        return None
    caminho = Path(caminho).resolve()
    try:
        return str(caminho.relative_to(ROOT))
    except ValueError:
        return caminho.name


@dataclass
class RelatorioConsulta:
    paciente_id: str
    arquivo: str
    basal: str | None
    servicos: dict[str, str]
    features: dict
    features_basal: dict | None
    alteracoes: list[dict]
    transcricao: dict | None
    termos: list[dict]
    analise_texto: dict | None
    achados: list[Achado] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "achados"}
        d["achados"] = [a.as_dict() for a in self.achados]
        return d


def _achados_texto(paciente_id: str, momento: datetime, encontrados: list[termos.TermoEncontrado],
                   analise: AnaliseTexto | None) -> list[Achado]:
    saida = []
    sintomas_azure = {e.texto.lower() for e in (analise.entidades if analise else [])
                      if e.categoria == "SymptomOrSign" and not e.negada and not _POSITIVO.search(e.texto)}
    por_categoria: dict[str, list[termos.TermoEncontrado]] = {}
    for t in encontrados:
        if not t.negado:
            por_categoria.setdefault(t.categoria, []).append(t)
    for categoria, lista in por_categoria.items():
        corroborado = any(termos.normalizar(s) in t.termo or t.termo in termos.normalizar(s)
                          for s in sintomas_azure for t in lista)
        escore = max(t.escore for t in lista) + (0.05 if corroborado else 0.0)
        saida.append(Achado(paciente_id, Modalidade.TEXTO, "termo_critico", min(escore, 1.0),
                            "lexico-hospital" + ("+azure-ta4h" if corroborado else ""),
                            evidencia={"categoria": categoria, "termos": [t.termo for t in lista],
                                       "trechos": [t.trecho for t in lista],
                                       "negados_ignorados": [t.termo for t in encontrados if t.negado]},
                            momento=momento))
    if analise:
        cobertos = {t.termo for t in encontrados}
        extras = sorted(s for s in sintomas_azure if not any(termos.normalizar(s) in c or c in termos.normalizar(s)
                                                              for c in cobertos))
        if extras:
            saida.append(Achado(paciente_id, Modalidade.TEXTO, "sintoma_relatado", 0.5, "azure-ta4h",
                                evidencia={"sintomas": extras}, momento=momento))
        neg = analise.confianca_sentimento.get("negative", 0.0)
        if analise.sentimento == "negative" and neg >= SENTIMENTO_NEGATIVO_MIN:
            saida.append(Achado(paciente_id, Modalidade.TEXTO, "sentimento_negativo", round(neg * 0.6, 3),
                                "azure-language", evidencia={"confianca": neg, "opinioes": analise.opinioes},
                                momento=momento))
    return saida


def analisar_consulta(arquivo: Path | str, paciente_id: str, basal: Path | str | None = None,
                      speech: ClienteSpeech | None = None, language: ClienteLanguage | None = None,
                      usar_azure: bool = True, momento: datetime | None = None) -> RelatorioConsulta:
    arquivo = Path(arquivo)
    momento = momento or datetime.now(timezone.utc)
    speech = speech or ClienteSpeech()
    language = language or ClienteLanguage()
    erros: list[str] = []

    # --- acústica
    feats = acustica.extrair(acustica.carregar(arquivo))
    feats_basal = acustica.extrair(acustica.carregar(basal)) if basal else None
    alteracoes = acustica.detectar(feats, feats_basal)
    achados = [Achado(paciente_id, Modalidade.AUDIO, a.tipo, a.escore, "librosa", evidencia=a.evidencia(),
                      momento=momento) for a in alteracoes]

    # --- transcrição
    servicos = {"speech": "indisponível", "language": "indisponível"}
    transcricao: Transcricao | None = None
    if usar_azure and speech.disponivel:
        try:
            transcricao = speech.transcrever(arquivo)
            servicos["speech"] = "azure-speech"
        except Exception as exc:  # falha de nuvem não derruba a análise local
            erros.append(f"speech: {exc}")
    if transcricao is None:
        transcricao = transcricao_de_referencia(arquivo)
        if transcricao:
            servicos["speech"] = "referencia (roteiro .txt)"

    # --- texto
    encontrados: list[termos.TermoEncontrado] = []
    analise: AnaliseTexto | None = None
    if transcricao and transcricao.texto:
        encontrados = termos.encontrar(transcricao.texto)
        if usar_azure and language.disponivel:
            try:
                analise = language.analisar(transcricao.texto)
                servicos["language"] = "azure-language"
            except Exception as exc:
                erros.append(f"language: {exc}")
        achados += _achados_texto(paciente_id, momento, encontrados, analise)

    return RelatorioConsulta(
        paciente_id=paciente_id, arquivo=_relativo(arquivo), basal=_relativo(basal),
        servicos=servicos, features=feats.as_dict(),
        features_basal=feats_basal.as_dict() if feats_basal else None,
        alteracoes=[{"tipo": a.tipo, "escore": a.escore, **a.evidencia()} for a in alteracoes],
        transcricao=transcricao.as_dict() if transcricao else None,
        termos=[t.as_dict() for t in encontrados],
        analise_texto=analise.as_dict() if analise else None,
        achados=achados, erros=erros)


def salvar(rel: RelatorioConsulta, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"consulta_{Path(rel.arquivo).stem}.json"
    caminho.write_text(json.dumps(rel.as_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return caminho
