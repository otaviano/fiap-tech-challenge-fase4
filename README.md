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
| Análise de vídeo ✅ | [`video/`](src/med_monitor/video/) — YOLOv8-pose (keypoints do OpenPose) e YOLOv8 de objetos; execução incorreta com AUC por pessoa 0,87–0,95 no REHAB24-6; relatório automático da sessão ([detalhes](docs/video.md)) |
| Análise de áudio ✅ (Azure 🟡 aguardando conta) | [`audio/`](src/med_monitor/audio/) — fala ofegante e fadiga vs basal do paciente, termos críticos com negação; [`azure/`](src/med_monitor/azure/) — Speech to Text, sentimento e Text Analytics for Health ([detalhes](docs/audio.md)) |
| Anomalias em sinais vitais ✅ | [`anomalias/sinais_vitais.py`](src/med_monitor/anomalias/sinais_vitais.py) — NEWS2 + desvio robusto + Isolation Forest; recall 96% em 336 h de UTI ([detalhes](docs/sinais-vitais.md)) |
| Anomalias em prescrições ✅ | [`anomalias/prescricoes.py`](src/med_monitor/anomalias/prescricoes.py) — regras do formulário + variação de dose aprendida por medicamento; recall 92% com 1,2% de alertas em internações normais ([detalhes](docs/prescricoes.md)) |
| Padrões de movimentação ✅ | amplitude vs sessão anterior e queda, no [vídeo](docs/video.md) |
| Fusão multimodal e alertas em tempo real ✅ | [`fusao/motor.py`](src/med_monitor/fusao/motor.py), [`alertas/despacho.py`](src/med_monitor/alertas/despacho.py), [`serving/`](src/med_monitor/serving/) — painel SSE da equipe médica, escalonamento por concordância entre modalidades, ciência auditada ([detalhes](docs/fusao-alertas.md)) |
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

### Áudio das consultas

```bash
pip install -e ".[audio,dev]"
med-monitor audio amostras           # consultas sintéticas (requer espeak-ng)
med-monitor audio analisar data/raw/audio/PRT-2026-200102_atual.wav \
    --basal data/raw/audio/PRT-2026-200102_basal.wav [--api http://localhost:8000]
med-monitor audio avaliar
```

### Vídeo das sessões de fisioterapia

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[video,anomalias,dev]"
med-monitor video baixar             # 22 vídeos do REHAB24-6 (download seletivo, ~475 MB)
med-monitor video avaliar            # pose + validação por pessoa (~12 min na 1ª vez)
med-monitor video relatorio PM_016 [--api http://localhost:8000]
```

### Prescrições

```bash
med-monitor prescricoes avaliar
med-monitor prescricoes exemplo --tipo erro_decimal [--api http://localhost:8000]
```

### Painel da equipe médica (tempo real)

```bash
pip install -e ".[anomalias,serving,dev]"
med-monitor serve --port 8000        # http://localhost:8000 → "Iniciar plantão"
med-monitor monitorar --horas 6      # o mesmo plantão no terminal
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
