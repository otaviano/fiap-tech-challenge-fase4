# Tech Challenge — Fase 4: Monitoramento Multimodal de Pacientes

> **FIAP Postech — IA para Devs**
> Vídeo + áudio + séries temporais → detecção de anomalias → alerta à equipe médica, com Azure AI

> 🚧 **Em construção.** Plano, decisões e status por requisito em [`docs/plano.md`](docs/plano.md).

Quarta fase de uma linha contínua: a [Fase 1](https://github.com/otaviano/fiap-tech-challenge-fase1)
construiu o classificador de diagnóstico, a
[Fase 2](https://github.com/otaviano/fiap-tech-challenge-fase2) otimizou-o por
algoritmo genético e trouxe um LLM local, e a
[Fase 3](https://github.com/otaviano/fiap-tech-challenge-fase3) transformou o LLM
em assistente clínico com alertas e auditoria. Agora o Hospital Santa Helena
(fictício) passa a **monitorar pacientes continuamente** — vídeos de
fisioterapia, gravações de consulta, sinais vitais, prescrições — e a fundir os
achados em alertas para a equipe médica.

---

## Requisitos → módulos

| Requisito do enunciado | Onde está |
|---|---|
| Análise de vídeo (OpenPose, YOLOv8, relatório de desvios) | [`src/med_monitor/video/`](src/med_monitor/video/) |
| Análise de áudio (Azure Speech to Text, Azure Text Analytics, alterações vocais) | [`src/med_monitor/audio/`](src/med_monitor/audio/), [`src/med_monitor/azure/`](src/med_monitor/azure/) |
| Anomalias em sinais vitais ✅ | [`anomalias/sinais_vitais.py`](src/med_monitor/anomalias/sinais_vitais.py) — NEWS2 + desvio robusto + Isolation Forest; recall 96% em 336 h de UTI ([detalhes](docs/sinais-vitais.md)) |
| Anomalias em prescrições e movimentação | [`src/med_monitor/anomalias/`](src/med_monitor/anomalias/) |
| Fusão multimodal e alertas em tempo real | [`src/med_monitor/fusao/`](src/med_monitor/fusao/), [`src/med_monitor/alertas/`](src/med_monitor/alertas/), [`src/med_monitor/serving/`](src/med_monitor/serving/) |
| Serviços gerenciados em nuvem (Azure) | [`infra/`](infra/) — Terraform |
| Fluxo multimodal | [`docs/arquitetura.md`](docs/arquitetura.md) |

---

## Como executar

Requer **Python ≥ 3.10**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                 # núcleo
pip install -e ".[video,audio,anomalias,serving,dev]"   # tudo
cp .env.example .env                    # credenciais Azure (ver infra/README.md)
med-monitor info
pytest
```

### Sinais vitais

```bash
pip install -e ".[anomalias,dev]"
med-monitor vitais baixar                                   # 10 registros do MIMIC-III (PhysioNet)
med-monitor vitais analisar --sintetico --grafico           # série sintética com eventos injetados
med-monitor vitais analisar --registro p000020-2183-04-28-17-47n --grafico
med-monitor vitais avaliar                                  # recall/latência/carga de alarme (~10 min)
```

---

## Dados e créditos

- **REHAB24-6** — vídeos de fisioterapia com execuções corretas e incorretas.
  Černek, Sedmidubský e Budíková, *REHAB24-6: Physical Therapy Dataset for
  Analyzing Pose Estimation Methods*, SISAP 2024.
  [Zenodo 13305826](https://zenodo.org/records/13305826), licença CC BY-NC 4.0.
  Não é redistribuído aqui: baixe com `./scripts/baixar_rehab24.sh`.
- **MIMIC-III Waveform Database Matched Subset** — sinais vitais de UTI.
  Moody B, Moody G, Villarroel M, Clifford GD, Silva I. PhysioNet, 2020.
  <https://doi.org/10.13026/c2294b>, licença ODbL v1.0. Não é redistribuído:
  baixe com `med-monitor vitais baixar`.
- **PhysioNet** — Goldberger AL et al. *PhysioBank, PhysioToolkit, and
  PhysioNet*. Circulation 101(23), 2000; e Pollard T et al. *PhysioNet as a
  global platform for biomedical research*. Nature Health, 2026.

---

*FIAP Postech — IA para Devs | Tech Challenge Fase 4 | 2026*
