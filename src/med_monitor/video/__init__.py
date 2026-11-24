"""Entrega técnica 1 — Análise de vídeo das sessões de fisioterapia (REHAB24-6).

- `pose`: YOLOv8-pose (17 keypoints COCO, o formato do OpenPose) com rastreio do paciente e cache;
- `metricas`: ângulos articulares por quadro e métricas por repetição;
- `avaliacao`: execução incorreta (regressão logística validada por pessoa) e pessoa extra vs rótulo;
- `objetos`: YOLOv8 de objetos, área crítica do paciente e queda;
- `relatorio`: relatório automático da sessão (Markdown + gráfico) e `Achado`s para a fusão.

Ver docs/video.md.
"""
