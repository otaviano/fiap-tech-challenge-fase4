"""Azure AI Language — sentimento e Text Analytics for Health sobre a transcrição.

- **Sentimento com opinion mining** (`pt-BR`): o tom geral do paciente e a
  quais aspectos ele se refere ("respirar" → negativo).
- **Text Analytics for Health** (`pt`, decisão D5): entidades clínicas —
  sintomas, medicamentos, dosagens — com a **asserção** de cada uma. A asserção
  é o que separa "estou com falta de ar" de "não tenho falta de ar": o serviço
  marca a segunda com certeza negativa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config import AzureSettings

# Categorias do Text Analytics for Health que interessam ao monitoramento.
CATEGORIAS_SAUDE = {"SymptomOrSign", "Diagnosis", "MedicationName", "Dosage", "BodyStructure",
                    "ExaminationName", "MeasurementValue", "MeasurementUnit"}


@dataclass(frozen=True)
class EntidadeSaude:
    texto: str
    categoria: str
    confianca: float
    negada: bool = False
    normalizado: str | None = None


@dataclass
class AnaliseTexto:
    fonte: str = "azure-language"
    sentimento: str | None = None                  # positive | neutral | negative | mixed
    confianca_sentimento: dict[str, float] = field(default_factory=dict)
    opinioes: list[dict] = field(default_factory=list)
    entidades: list[EntidadeSaude] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class ClienteLanguage:
    def __init__(self, settings: AzureSettings | None = None):
        self.settings = settings or AzureSettings()

    @property
    def disponivel(self) -> bool:
        return self.settings.language_configurado

    def _cliente(self):
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        return TextAnalyticsClient(self.settings.language_endpoint, AzureKeyCredential(self.settings.language_key))

    def analisar(self, texto: str) -> AnaliseTexto:
        if not self.disponivel:
            raise RuntimeError("Azure Language não configurado (AZURE_LANGUAGE_ENDPOINT / AZURE_LANGUAGE_KEY)")
        cliente = self._cliente()
        saida = AnaliseTexto()

        [sent] = cliente.analyze_sentiment([texto], language="pt-BR", show_opinion_mining=True)
        if not sent.is_error:
            saida.sentimento = sent.sentiment
            saida.confianca_sentimento = {k: round(getattr(sent.confidence_scores, k), 3)
                                          for k in ("positive", "neutral", "negative")}
            for frase in sent.sentences:
                for op in frase.mined_opinions:
                    saida.opinioes.append({"alvo": op.target.text, "sentimento": op.target.sentiment,
                                           "avaliacoes": [a.text for a in op.assessments]})

        poller = cliente.begin_analyze_healthcare_entities([texto], language=self.settings.text_language)
        for doc in poller.result():
            if doc.is_error:
                continue
            for e in doc.entities:
                if e.category not in CATEGORIAS_SAUDE:
                    continue
                negada = bool(e.assertion and getattr(e.assertion, "certainty", None)
                              and "negative" in str(e.assertion.certainty).lower())
                saida.entidades.append(EntidadeSaude(e.text, e.category, round(e.confidence_score, 3),
                                                     negada, e.normalized_text))
        return saida
