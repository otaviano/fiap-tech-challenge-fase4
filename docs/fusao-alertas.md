# Fusão multimodal e alertas em tempo real

Entrega técnica 3.4 do enunciado: *"gerar alertas automáticos para a equipe
médica com base nas anomalias detectadas"*. Também cobre o fluxo que o vídeo
precisa mostrar: *detecção e resposta a anomalias* e *fluxo final do alerta à
equipe médica*.

```
analisadores ──Achado──▶ MotorFusao ──Decisao──▶ Despachante ──┬─▶ trilha de auditoria (JSONL)
 (sinais vitais,           episódio por            alerta +    ├─▶ barramento ─▶ SSE /stream ─▶ painel
  áudio, vídeo...)         paciente                achados     └─▶ canais externos (webhook, opcional)
                                                                          │
                                         profissional ─ "Ciente" ─▶ reconhecimento (auditado)
```

Código: [`fusao/motor.py`](../src/med_monitor/fusao/motor.py),
[`alertas/despacho.py`](../src/med_monitor/alertas/despacho.py),
[`serving/simulador.py`](../src/med_monitor/serving/simulador.py),
[`serving/api.py`](../src/med_monitor/serving/api.py),
[`serving/painel.html`](../src/med_monitor/serving/painel.html).

## Do achado ao alerta

**Achado** é uma observação de um detector, como "SpO2 15 MADs abaixo do
basal". **Alerta** é uma decisão de acionar alguém. A fusão separa as duas
coisas.

### 1. Severidade base

| Origem | Severidade base | Fundamento |
|---|---|---|
| NEWS2: um parâmetro com 3 pontos | moderado | NEWS2 "low-medium": avaliação urgente à beira do leito |
| NEWS2 parcial ≥ 5 | alto | NEWS2 "medium": resposta urgente |
| NEWS2 parcial ≥ 7 | **crítico** | NEWS2 "high": emergência |
| Isolation Forest (deterioração multiparamétrica) | moderado | padrão anômalo, sem limiar clínico absoluto |
| Desvio do basal do paciente | pelo escore, **teto moderado** | mudança relativa pede olhar, não chamar o plantonista |
| Relato de risco de autolesão na consulta | **crítico** | não espera corroboração |
| Queda detectada no vídeo | alto | paciente no chão: avaliação médica imediata |
| Execução incorreta na fisioterapia, amplitude reduzida | pelo escore, no máximo moderado | vai para a **fisioterapia**, não para a enfermagem |
| Outras fontes (áudio, texto, vídeo...) | pelo escore: ≥ 0,8 alto · ≥ 0,5 moderado | ex.: "falta de ar" relatada (0,80) é alto; fala ofegante (0,75–0,90) vai de moderado a alto |

### 2. Corroboração e concordância entre canais

Cada modalidade pertence a um **canal físico**, que é a origem do dado:

| Canal | Modalidades |
|---|---|
| monitor | sinais vitais |
| consulta | áudio (como o paciente fala) e texto (o que ele diz) |
| câmera | vídeo e movimentação |
| prontuário | prescrições |

- **Corroboração (mesmo canal):** 2 ou mais detectores diferentes em 15 min
  elevam um nível, **até no máximo *alto***. Os três detectores de sinais
  vitais leem o mesmo sensor, e um oxímetro reposicionado engana os três ao
  mesmo tempo. Áudio e texto saem da mesma gravação. Em ambos os casos a
  concordância é evidência, mas não independente.
- **Concordância entre canais:** achados de canais diferentes em 15 min elevam
  mais um nível, **sem teto**. Dessaturação no monitor somada a fala ofegante
  na consulta são fontes independentes.

Resultado: **um alerta só é crítico com NEWS2 ≥ 7, com relato de risco de
autolesão, ou com canais físicos que concordam.**

### 3. Episódio e escalonamento

Um episódio começa no primeiro achado do paciente e termina depois de 30 min
sem achados. Dentro dele, **só sai alerta novo se a severidade subir**, e a
mensagem começa com `ESCALONAMENTO`. Repetir a mesma condição não gera um novo
alerta.

### 4. Destinatário

| Severidade | Quem é acionado |
|---|---|
| crítico | time de resposta rápida |
| alto | médico plantonista |
| moderado | enfermagem do leito |
| informativo (técnico, ex.: sensor sem sinal) | enfermagem do leito |

Exceção: desvio de execução e amplitude reduzida, enquanto ficam em moderado ou abaixo, vão para a **fisioterapia**.

### Exemplo: a mesma dessaturação em três momentos

| Momento | Achados recentes | Alerta |
|---|---|---|
| 07:48 | NEWS2: SpO2 89,5% (3 pontos) | **moderado** → enfermagem: "Dessaturação — SpO2 89,5%" |
| 07:56 | + desvio robusto: SpO2 85% contra basal de 93,2% | **alto** → plantonista: "ESCALONAMENTO · … corroborado por 2 detectores" |
| (+ consulta) | + fala ofegante na consulta (`med-monitor audio analisar … --api`) | **crítico** → resposta rápida: "… concordância entre consulta e monitor" |

## Resposta: reconhecimento

No painel, o profissional se identifica e clica em **Ciente**. O reconhecimento:

- vai para a trilha de auditoria (`alerta_reconhecido`) com quem deu ciência, a
  hora e os **segundos até a ciência**;
- aparece no painel para toda a equipe ("ciente · CRM-SP 111.222 · 12 s");
- não pode ser sobrescrito, porque a primeira ciência é a que vale.

A trilha permite reconstruir o caminho completo:
`achado → alerta_emitido (com os ids dos achados) → alerta_reconhecido`.

## Tempo real

**Replay acelerado** (decisão D3): 6 leitos de UTI reproduzem registros reais do
MIMIC-III minuto a minuto, de 60× a 600×. Três leitos recebem eventos injetados
em minutos fixos, para que a demonstração seja reproduzível:

| Leito | Registro | Evento injetado | Início no plantão |
|---|---|---|---|
| UTI-01 | p000020 | — (só eventos reais) | |
| UTI-02 | p000177 | dessaturação | 07:45 |
| UTI-03 | p000217 | deterioração combinada | 07:20 |
| UTI-04 | p000263 | taquicardia | 08:10 |
| UTI-05 | p000123 | — | |
| UTI-06 | p000214 | — | |

As 3 h anteriores ao plantão (a partir das 07:00) são processadas sem pausa e
sem alertar, para que os detectores aprendam o basal de cada paciente.

**Primeiras 3 h do plantão** (`med-monitor monitorar --horas 3`): 10 alertas.
Os três eventos injetados viraram alerta em 3 a 5 min:
- UTI-03: o Isolation Forest pegou a deterioração combinada 5 min depois do
  início, antes de qualquer limiar clínico;
- UTI-02: dessaturação em 3 min;
- UTI-04: taquicardia em 5 min.

Os outros 6 alertas vêm de eventos **reais** dos pacientes: taquipneia, NEWS2 ≥ 5
e FR acima do basal.

## Como executar

```bash
pip install -e ".[anomalias,serving,dev]"
med-monitor vitais baixar                  # uma vez: registros do MIMIC-III
med-monitor serve --port 8000              # painel em http://localhost:8000 → "Iniciar plantão"
med-monitor monitorar --horas 6            # alternativa: plantão no terminal, sem painel
```

Achado de outra modalidade entrando na fusão (é assim que áudio e vídeo vão se
conectar):

```bash
curl -X POST localhost:8000/achados -H 'Content-Type: application/json' -d '{
  "paciente_id": "PRT-2026-200102", "modalidade": "audio", "tipo": "fala_ofegante",
  "escore": 0.72, "fonte": "azure-speech+librosa",
  "evidencia": {"pausas_respiratorias_por_min": 14}
}'
```

Sem `momento`, o achado é carimbado com o relógio atual do plantão.

## Canais externos

`CanalWebhook(url, minima="alto")` envia alertas a partir da severidade
mínima para um webhook (Teams, Slack, pager). Vem desligado. Uma falha no canal
fica registrada na trilha (`canal_falhou`) e não interrompe o despacho para o
painel.
