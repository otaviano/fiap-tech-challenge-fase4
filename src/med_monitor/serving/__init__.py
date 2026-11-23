"""API REST + painel de alertas em tempo real.

- `POST /video`, `POST /audio`: processam uma mídia e devolvem achados/relatório;
- `GET /stream/alertas`: Server-Sent Events com os alertas à medida que surgem;
- simulador que reproduz séries do PhysioNet em velocidade acelerada, para a
  demonstração de "tempo real" no vídeo.
"""
