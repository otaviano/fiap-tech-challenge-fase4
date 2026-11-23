"""Entrega técnica 3 — Detecção de anomalias.

Três frentes, cada uma com seu detector:
- `sinais_vitais`: séries temporais de FC, PA e SpO2 (PhysioNet) — limiares
  clínicos + desvio robusto em janela deslizante (z-score/mediana) + Isolation
  Forest multivariado;
- `prescricoes`: evolução de prescrições — mudança inesperada de dose,
  duplicidade terapêutica, suspensão abrupta (dados sintéticos no padrão do
  prontuário da Fase 3);
- `movimentacao`: padrões de movimentação na internação (derivados dos
  keypoints do vídeo ou de série sintética de atividade).

Todos operam em modo *streaming* (um ponto por vez), para o monitoramento em
tempo real. Saída: `Achado(modalidade=SINAIS_VITAIS | PRESCRICAO | MOVIMENTACAO)`.
"""
