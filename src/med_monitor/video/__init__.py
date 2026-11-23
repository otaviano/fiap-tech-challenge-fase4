"""Entrega técnica 1 — Análise de vídeo.

Escopo:
- ler vídeos clínicos (sessões de fisioterapia; cirurgia como extensão);
- `pose`: keypoints por quadro com YOLOv8-pose (formato COCO-17, o mesmo do
  OpenPose — ver decisão D1 em docs/plano.md), ângulos articulares e amplitude
  de movimento por repetição;
- `objetos`: YOLOv8 para objetos e áreas críticas (pessoa fora da área segura,
  equipamento ausente, queda);
- `relatorio`: relatório automático do procedimento com os desvios encontrados.

Saída: `Achado(modalidade=VIDEO | MOVIMENTACAO, ...)`.
"""
