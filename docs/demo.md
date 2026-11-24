# Demonstração integrada

Um plantão de UTI em que **todas as modalidades entram na mesma fusão**. É o
roteiro que o vídeo de entrega mostra, reproduzível de ponta a ponta. Código em
[`serving/demo.py`](../src/med_monitor/serving/demo.py).

```bash
med-monitor serve --port 8000        # http://localhost:8000 → "Iniciar demonstração"
med-monitor monitorar --demo --horas 3   # o mesmo roteiro no terminal
```

## Roteiro

6 leitos reproduzem registros reais do MIMIC-III a partir das 07:00. Em
horários fixos entram eventos dos outros canais:

| Horário | Leito | Canal | Evento | Alerta |
|---|---|---|---|---|
| 07:20 | UTI-03 | monitor | deterioração combinada (injetada) | moderado 07:25 → **alto** 07:31 |
| 07:45 | UTI-02 | monitor | dessaturação (injetada) | moderado 07:48 |
| **07:52** | UTI-02 | **consulta** | áudio: fala ofegante, "falta de ar", "peito apertado" (Azure Speech + Language) | **crítico**: concordância entre consulta e monitor, para o time de resposta rápida |
| **08:00** | UTI-05 | **prontuário** | morfina 4 mg → **40 mg** | **alto**: possível erro decimal, para o médico prescritor e a farmácia clínica |
| 08:10 | UTI-04 | monitor | taquicardia (injetada) | alto 08:15 |
| **08:25** | UTI-05 | monitor | depressão respiratória (FR 7, SpO2 88, injetada) | **crítico** 08:31: concordância entre monitor e prontuário |
| **08:45** | UTI-03 | **consulta** | áudio: fadiga vocal, "cansada", "febre" | moderado → **alto** |
| **09:00** | UTI-06 | **consulta** | áudio: "estou bem, não tenho falta de ar, nem dor no peito…" | **nenhum achado**, negações respeitadas |
| 09:13 | UTI-06 | monitor | NEWS2 6 (evento real do paciente) | alto |
| **09:20** | UTI-04 | **câmera** | sessão de fisioterapia (REHAB24-6 PM_016): 5 de 10 repetições incorretas | moderado, para a **fisioterapia** |

**O que o roteiro mostra:**
- **Concordância entre canais:** dois críticos que nenhum canal produziria
  sozinho. A dessaturação sozinha é moderada; com a fala ofegante, é crítico. O
  erro de morfina sozinho é alto; com a depressão respiratória que aparece meia
  hora depois, é crítico. A prescrição vale 6 h na fusão.
- **Especificidade:** a consulta das 09:00 tem "falta de ar" e "dor no peito"
  no texto, ambos negados, e não gera alerta.
- **Relato não substitui sinal:** na UTI-06 o paciente diz que está bem às 09:00,
  e o monitor mostra NEWS2 6 treze minutos depois.
- **Destinatário certo:** enfermagem, plantonista, resposta rápida, farmácia ou
  fisioterapia, conforme a severidade e o tipo.

## Como funciona

- **Relógio comum.** Os leitos andam juntos, minuto a minuto: um leito só
  processa o minuto *k* depois que todos terminaram o *k − 1*. Sem isso,
  diferenças no tempo de carga das séries viravam minutos de defasagem, e a
  consulta das 07:52 chegava à fusão depois de achados das 07:56.
- **O plantão pausa durante a análise.** No horário de um evento, o produtor
  roda numa thread enquanto os leitos esperam. Os achados recebem o horário do
  disparo e se alinham com o que o monitor mostrava naquele minuto.
- **Linha do tempo multimodal no painel:** cada evento aparece como
  "analisando…" e depois com os achados, os serviços usados (Azure Speech e
  Language), as negações ignoradas e a severidade dos alertas que gerou.
- **Falha de um canal não derruba o plantão.** Se o Azure cair, o evento aparece
  como "falhou" na linha do tempo, a falha fica na trilha de auditoria e o
  monitor segue.

## Tempo de execução

A 120×, a demo leva cerca de **2,5 minutos** (07:00 → 09:30). Cada consulta
pausa o plantão por **12 a 28 s** enquanto o Azure transcreve e analisa. O Text
Analytics for Health é uma operação assíncrona no serviço: o pedido é enviado e
o resultado é consultado até ficar pronto. Durante a pausa, a linha do tempo
mostra "analisando…".

Pré-requisitos (uma vez):

```bash
med-monitor vitais baixar             # registros do MIMIC-III
med-monitor audio amostras            # consultas (vozes neurais com Azure; espeak-ng sem)
med-monitor video relatorio PM_016    # relatório da sessão de fisioterapia usada às 09:20
```
