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
| 1.1 | Processar vídeos clínicos (fisioterapia / cirurgia) | `video/` — [docs](video.md) | ✅ fisioterapia (REHAB24-6, 22 vídeos) |
| 1.2 | Análise postural (OpenPose) | `video/pose.py` + `video/metricas.py` | ✅ YOLOv8-pose (keypoints COCO do OpenPose), 36 q/s em CPU |
| 1.3 | Objetos e áreas críticas (YOLOv8) | `video/objetos.py` | ✅ pessoa visível 83%/96%, área do paciente, queda |
| 1.4 | Relatório automático de desvios | `video/relatorio.py` | ✅ AUC por pessoa 0,87 (braço) e 0,95 (agachamento) |
| 2.1 | Processar áudios de consultas | `audio/consulta.py` — [docs](audio.md) | ✅ |
| 2.2 | Alterações vocais (cansaço, dificuldade respiratória) | `audio/acustica.py` | ✅ dispneia 24/24, fadiga 12/12, 0 falsos alarmes (sintético) |
| 2.3 | Azure Speech to Text | `azure/speech.py` | 🟡 implementado e testado com dublê; falta rodar com a conta |
| 2.4 | Termos críticos e sentimento (Azure Text Analytics) | `azure/language.py` + `audio/termos.py` | 🟡 léxico com negação ✅; Azure falta rodar com a conta |
| 3.1 | Anomalias em sinais vitais (FC, PA, SpO2) | `anomalias/sinais_vitais.py` — [docs](sinais-vitais.md) | ✅ recall 96% em 336 h reais |
| 3.2 | Anomalias na evolução de prescrições | `anomalias/prescricoes.py` — [docs](prescricoes.md) | ✅ recall 92%, 1,2% de alertas em internações normais |
| 3.3 | Padrões de movimentação na internação | `video/relatorio.py` (amplitude vs sessão anterior, queda) | ✅ parcial: sessões de fisioterapia |
| 3.4 | Alertas automáticos à equipe médica | `fusao/` + `alertas/` + `serving/` — [docs](fusao-alertas.md) | ✅ fusão, painel SSE, ciência auditada; áudio/vídeo entram por `POST /achados` |
| — | Contratos `Achado`/`Alerta`, config, auditoria, CLI | `domain.py`, `config.py`, `audit.py`, `cli.py` | ✅ |
| — | Recursos Azure (Speech + Language + orçamento) | `infra/` | ✅ (falta conta e `apply`) |

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

**D5 — Text Analytics for Health com `language="pt"`.** A API hospedada aceita
português só pela tag genérica `pt` (não há `pt-br` nem `pt-pt` na lista). O
texto em pt-BR vai com `pt`. O sentimento e o reconhecimento de entidades comuns
continuam com `pt-BR`/`pt`. Variável: `AZURE_TEXT_LANGUAGE`.

**D6 — Dataset de vídeo: REHAB24-6.** Dataset público de fisioterapia ([Zenodo
13305826](https://zenodo.org/records/13305826)), com 65 gravações de 6
exercícios (abdução de braço, "arm VW", flexão apoiada na mesa, abdução de
perna, avanço e agachamento) feitas por 10 pessoas. Cada exercício foi executado pelo
menos 5 vezes **corretamente** e 5 vezes **incorretamente** sob orientação de um
fisioterapeuta. O vídeo é RGB a 30 FPS, de duas câmeras, e vem com segmentação
por repetição e com esqueleto 3D de referência. Isso dá *ground truth* para
medir o acerto do detector de desvios, e não só para demonstrá-lo. Licença
**CC BY-NC 4.0**: uso acadêmico permitido, com citação obrigatória no relatório
e no README. Os vídeos (2,65 GB) **não** entram no repositório: são baixados por
`scripts/baixar_rehab24.sh`. Cirurgia fica fora do escopo, porque o enunciado
diz "ex.: fisioterapia *ou* cirurgia".

**D7 — Orçamento Azure.** O teto é o crédito de US$ 100/ano, e o projeto foi
desenhado para não consumir nada dele:
- os dois recursos ficam na camada **F0 (gratuita)**. O Speech F0 dá 5 h de
  transcrição em tempo real por mês. O Language F0 dá 5.000 registros de texto
  por mês, sendo 1 registro = 1.000 caracteres. No F0 não existe cobrança
  excedente: passou do limite, a chamada é recusada (*throttle*);
- a estimativa de uso cabe com folga. São cerca de 30 consultas de 2–3 min,
  menos de 2 h de áudio. Cada transcrição dá cerca de 3 registros por análise,
  e com 3 análises por áudio são uns 300 registros por mês;
- risco conhecido: a documentação não deixa claro se o Text Analytics for Health
  entra na cota gratuita. Se o F0 recusar a chamada, só o recurso Language passa
  para a camada `S`, e o Speech continua no F0;
- há uma rede de proteção: um orçamento (`azurerm_consumption_budget_subscription`)
  com alerta por e-mail em 50% e 90% de US$ 20, criado só quando
  `budget_alert_email` é informado.

O Language fica em `eastus` e o Speech em `brazilsouth`. A documentação não
confirma o Text Analytics for Health no Brazil South, e o East US é região
principal para todos os recursos de Language. Como os dados são sintéticos, a
residência de dados não é uma restrição.

## Dados

| Modalidade | Fonte candidata | Observação |
|---|---|---|
| Sinais vitais | PhysioNet — **MIMIC-III Waveform Matched Subset**, registros numéricos (FC, PA invasiva, FR, SpO2 a 1/min; ODbL) | 10 registros curados, 336 h. BIDMC descartado: 8 min por registro e sem PA |
| Sons respiratórios | AudioSet (Google) — classes *Cough*, *Wheeze*, *Breathing*, *Gasp* | Só metadados + IDs do YouTube; baixar trechos curtos |
| Consultas (voz) | Sintéticas (espeak-ng, condição conhecida) para desenvolvimento e avaliação; gravações autorais do grupo com os mesmos roteiros para o vídeo | Nada de dado real de paciente. AudioSet não usado: só traz IDs do YouTube e embeddings, sem fala de consulta |
| Vídeo de fisioterapia | **REHAB24-6** (Zenodo, CC BY-NC 4.0): execuções corretas × incorretas rotuladas | Decisão D6. KIMORE (sob pedido) e UI-PRMD (só esqueleto, sem RGB) descartados |
| Prescrições | Sintéticas, no formato do prontuário da Fase 3 | Injetar anomalias rotuladas para medir recall |

## Em aberto

- [ ] Criar a conta Azure com o benefício de crédito de US$ 100/ano e confirmar
      que a assinatura aceita recursos F0 e tem limite de gasto (*spending limit*)
- [ ] No primeiro `terraform apply`, testar uma chamada de Text Analytics for
      Health no F0 (D7): se recusar, trocar `language_sku` para `S`
- [x] Vídeos de fisioterapia → dataset público REHAB24-6 (D6)
- [x] Text Analytics for Health em português → `language="pt"` (D5); região `eastus` (D7)
- [ ] Gravar os roteiros de `audio/amostras.py` com a voz do grupo (basal + atual) para o vídeo
- [x] Repositório público: <https://github.com/otaviano/fiap-tech-challenge-fase4>
