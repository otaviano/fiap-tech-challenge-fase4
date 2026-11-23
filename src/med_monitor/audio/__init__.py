"""Entrega técnica 2 — Análise de áudio das consultas.

- `acustica`: features da fala (pausas, grupos de fala, respiração audível, articulação, F0) e
  detecção de `fala_ofegante` / `fadiga_vocal` contra o basal do paciente;
- `termos`: léxico de termos críticos do hospital, com tratamento de negação;
- `consulta`: pipeline completo (acústica + Azure Speech to Text + Azure Text Analytics) → `Achado`s;
- `amostras`: consultas sintéticas em pt-BR com alterações controladas (espeak-ng);
- `avaliacao`: detecção, confusão e falso alarme nas consultas sintéticas.

Ver docs/audio.md.
"""
