"""Entrega técnica 2 — Análise de áudio.

Escopo:
- `transcricao`: Azure Speech to Text (pt-BR) sobre gravações de consulta;
- `texto`: Azure Text Analytics — sentimento, opinião e Text Analytics for
  Health (entidades clínicas: sintomas, medicamentos, dosagens) sobre a
  transcrição; lista de termos críticos do hospital;
- `acustica`: features locais (librosa) para sinais vocais — taxa de fala,
  pausas, jitter/shimmer, energia — indicativos de fadiga, disartria ou
  dificuldade respiratória, comparados com a linha de base do paciente.

Saída: `Achado(modalidade=AUDIO | TEXTO, ...)`.
"""
