# Plano de trabalho — Fase 4

Enunciado resumido: o Hospital Santa Helena (fictício, o mesmo das fases
anteriores) quer **monitorar pacientes continuamente com dados multimodais**
(vídeo, áudio, texto e séries temporais), detectar sinais precoces de risco e
**alertar a equipe médica em tempo real**, usando serviços gerenciados do Azure.

## Continuidade com as fases anteriores

| Fase | O que entregou | O que a Fase 4 aproveita |
|---|---|---|
| 1 | Classificador de câncer de mama (Wisconsin) | Disciplina de métrica: recall acima de accuracy — aqui, **falso negativo de alerta** é o erro caro |
| 2 | GA para hiperparâmetros + LLM local + IaC (AWS) | Padrão de IaC em Terraform (agora Azure) e de experimentos comparáveis |
| 3 | Assistente clínico (LangGraph), alertas determinísticos, auditoria, prontuário SQLite | Escala de severidade, filosofia "modelo observa, regra decide", trilha de auditoria (`audit.py` portado), formato de prontuário para as prescrições sintéticas |

## Entregas técnicas → módulos

| # | Requisito | Módulo | Status |
|---|---|---|---|
| 1.1 | Processar vídeos clínicos (fisioterapia / cirurgia) | `video/` | ⬜ |
| 1.2 | Análise postural (OpenPose) | `video/pose.py` | ⬜ |
| 1.3 | Objetos e áreas críticas (YOLOv8) | `video/objetos.py` | ⬜ |
| 1.4 | Relatório automático de desvios | `video/relatorio.py` | ⬜ |
| 2.1 | Processar áudios de consultas | `audio/` | ⬜ |
| 2.2 | Alterações vocais (cansaço, dificuldade respiratória) | `audio/acustica.py` | ⬜ |
| 2.3 | Azure Speech to Text | `azure/speech.py` + `audio/transcricao.py` | ⬜ |
| 2.4 | Termos críticos e sentimento (Azure Text Analytics) | `azure/language.py` + `audio/texto.py` | ⬜ |
| 3.1 | Anomalias em sinais vitais (FC, PA, SpO2) | `anomalias/sinais_vitais.py` | ⬜ |
| 3.2 | Anomalias na evolução de prescrições | `anomalias/prescricoes.py` | ⬜ |
| 3.3 | Padrões de movimentação na internação | `anomalias/movimentacao.py` | ⬜ |
| 3.4 | Alertas automáticos à equipe médica | `fusao/` + `alertas/` + `serving/` | ⬜ |
| — | Contratos `Achado`/`Alerta`, config, auditoria, CLI | `domain.py`, `config.py`, `audit.py`, `cli.py` | ✅ |
| — | Recursos Azure (Speech + Language) | `infra/` | ✅ (falta `apply`) |

### Entregáveis

- [ ] Código-fonte completo
- [ ] Relatório técnico: fluxo multimodal, modelos por tipo de dado, resultados e exemplos de anomalias
- [ ] Vídeo ≤ 15 min (YouTube/Vimeo, não listado): áudio e vídeo na prática, detecção e resposta a anomalias, integração Azure, fluxo do alerta até a equipe
- [ ] Documento de entrega em PDF (mesmo padrão das fases 1–3)

## Ordem sugerida

1. **Anomalias em sinais vitais** — dados prontos (PhysioNet), não depende de
   nuvem nem de GPU, e já valida `Achado` → fusão → `Alerta` → painel.
2. **Fusão + alertas + painel SSE** — o esqueleto do "tempo real" que todo o
   resto vai alimentar.
3. **Áudio com Azure** — `terraform apply`, transcrição, Text Analytics, features acústicas.
4. **Vídeo** — pose + objetos + relatório; plugar movimentação na fusão.
5. **Prescrições** — gerador sintético + detector.
6. Relatório, notebook de demonstração, roteiro e gravação.

## Decisões

**D1 — OpenPose via YOLOv8-pose.** O OpenPose original (CMU, Caffe) não tem
pacote pip mantido e exige build com CUDA; a máquina de desenvolvimento não tem
GPU. O YOLOv8-pose produz os mesmos 17 keypoints do **modelo COCO do OpenPose**,
roda em CPU e vem no mesmo pacote (`ultralytics`) do YOLOv8 de objetos — um
único runtime para as duas análises de vídeo. O relatório documenta a
equivalência de formato e a troca. *(Alternativa considerada: MediaPipe Pose,
33 pontos — descartada para não misturar dois formatos de esqueleto.)*

**D2 — Fusão tardia (late fusion) por regras.** Cada modalidade tem
taxa, latência e confiabilidade diferentes; fundir no nível de *achado* (e não
de feature) permite explicar cada alerta e degradar com elegância quando uma
modalidade está ausente. Concordância entre modalidades eleva a severidade.

**D3 — "Tempo real" por replay acelerado.** O monitoramento é demonstrado
reproduzindo registros do PhysioNet ponto a ponto (ex.: 1 h em 1 min) por um
simulador; os detectores são *streaming* de verdade, só a fonte é gravada.

**D4 — Azure com modo offline declarado.** Sem credenciais, os clientes Azure
não fingem: o pipeline roda com o que for local e o relatório marca que o
serviço não foi chamado. Para a entrega e o vídeo, rodar com Azure real.

## Dados

| Modalidade | Fonte candidata | Observação |
|---|---|---|
| Sinais vitais | PhysioNet — *BIDMC PPG and Respiration* (FC, FR, SpO2 a 1 Hz, acesso aberto) | PA: avaliar *MIMIC-III Waveform Matched Subset* (aberto) ou sintetizar |
| Sons respiratórios | AudioSet (Google) — classes *Cough*, *Wheeze*, *Breathing*, *Gasp* | Só metadados + IDs do YouTube; baixar trechos curtos |
| Consultas (voz) | Gravações autorais do grupo, roteirizadas (paciente com fala normal × cansada/ofegante) | Consentimento implícito (somos nós); nada de dado real de paciente |
| Vídeo de fisioterapia | Gravações autorais de exercícios (execução correta × compensação/desvio) | Avaliar também datasets públicos de reabilitação (ex.: KIMORE, UI-PRMD) |
| Prescrições | Sintéticas, no formato do prontuário da Fase 3 | Injetar anomalias rotuladas para medir recall |

## Em aberto

- [ ] Assinatura Azure: conta de estudante (crédito) ou F0 em conta pessoal?
- [ ] Vídeos de fisioterapia: gravar ou usar dataset público (licença)?
- [ ] Text Analytics for Health em pt-BR: confirmar suporte/região.
- [ ] Repositório público no GitHub (`fiap-tech-challenge-fase4`), como nas fases anteriores?
