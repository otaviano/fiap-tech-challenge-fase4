# Anomalias na evolução de prescrições

Entrega técnica 3.2 do enunciado: *"aplicar técnicas de detecção de anomalias
em evolução de prescrições (alterações inesperadas no tratamento)"*. Código em
[`anomalias/prescricoes.py`](../src/med_monitor/anomalias/prescricoes.py) e
[`dados/prescricoes.py`](../src/med_monitor/dados/prescricoes.py).

## Dados

Prescrições reais de pacientes não são públicas, então geramos **internações
sintéticas** no formato do prontuário da Fase 3 (`medicamento`, `dose`, `via`,
`frequencia`). Acrescentamos a dose numérica e o intervalo em horas, que
permitem calcular a dose diária.

- **Formulário do hospital:** 19 medicamentos em 11 classes. Cada um tem dose
  usual, dose máxima única e diária, intervalos, apresentação (passo de dose),
  marcação de **alta vigilância** (lista ISMP) e limiar de **desmame**.
  **Valores ilustrativos, não referência clínica.**
- **Evolução normal:**
  - revisão diária às 08:00;
  - insulina titulada com frequência;
  - antibióticos em curso de 7 a 10 dias;
  - desmame de corticoide e benzodiazepínico;
  - analgesia opioide suspensa com a melhora;
  - medicamentos de dose fixa (antibiótico, IBP, dipirona) sem titulação;
  - raros **escalonamentos legítimos** (enoxaparina de profilaxia para dose
    plena). São mudanças grandes que *não* são erro, e deixam a avaliação
    honesta.
- **7 tipos de anomalia injetados com rótulo:**

| Tipo | Exemplo |
|---|---|
| erro decimal | morfina 4 mg → **40 mg** |
| dose acima da máxima | tramadol 200 mg (máx. 100) |
| salto de dose | 2,2 a 3,5× a dose anterior, dentro da máxima |
| duplicidade terapêutica | heparina iniciada com enoxaparina ativa |
| suspensão abrupta | prednisona 40 mg/dia suspensa sem desmame |
| frequência inesperada | ceftriaxona de 24/24 h para 6/6 h |
| **ajuste atípico** | clonazepam 1,5 a 1,9× no meio de um desmame, **abaixo** do limiar das regras |

## Dois detectores

| Detector | Pergunta | Técnica |
|---|---|---|
| **Regras do formulário** | "Viola algo que o hospital definiu?" | dose ou dose diária acima da máxima; razão ≥ 8× (provável vírgula); classe de risco duplicada; suspensão acima do limiar de desmame sem redução nas últimas 48 h; intervalo 3× diferente; dose diária ≥ 2× |
| **Variação aprendida** | "É atípico **para este medicamento**?" | distribuição robusta do log da razão entre dose diária nova e anterior, aprendida no histórico do hospital; \|z\| ≥ 3,5 |

**Variação típica aprendida no histórico:**
- insulina regular ±41%;
- prednisona e clonazepam ±19% (desmames);
- hidrocortisona, vancomicina e enoxaparina ±16%.

O que é normal para a insulina é atípico para o clonazepam.

**A correção não é um novo erro.** Quando a dose volta ao normal depois de uma
prescrição que já foi sinalizada, o detector entende que é a resposta ao
alerta. Sem essa regra, cada erro decimal gerava um segundo alerta na sua
própria correção, porque a dose cai 10×.

## Resultados

O detector foi treinado em 600 internações sem anomalia (o histórico) e
testado em 800 internações novas, 60% delas com uma anomalia injetada.

| Detector | Ajuste atípico | Os outros 6 tipos | Recall total | Alertas em internações sem anomalia |
|---|---|---|---|---|
| Regras | **0/49** | 444/444 | 90% | 23/1.997 (1,1%) |
| Regras + variação aprendida | **9/49** | 444/444 | **92%** | 24/1.997 (1,2%) |

**Leitura:**
- **Os 6 tipos "de regra" são 100% detectados.** São violações explícitas do
  formulário, e um sistema de prescrição deveria pegar todas.
- **A variação aprendida é o que pega o ajuste atípico**, que nenhuma regra
  alcança. Ela acerta quando o histórico é previsível (desmames de corticoide e
  benzodiazepínico) e perde quando a titulação normal já é ampla. O ganho é de
  +2 pontos no recall total, por +1 alerta em 2.000 prescrições.
- **Dos 23 alertas em internações sem anomalia, quase todos são o escalonamento
  legítimo da enoxaparina** (40 mg/dia → 80 mg de 12/12 h, 4× a dose diária). É
  exatamente o tipo de mudança que um farmacêutico clínico quer conferir.

```bash
med-monitor prescricoes avaliar                        # ~10 s; salva o modelo de variação
med-monitor prescricoes exemplo --tipo erro_decimal    # uma internação, evento a evento
med-monitor prescricoes exemplo --tipo ajuste_atipico --semente 5
```

## Integração com a fusão

As prescrições são o canal **prontuário**, independente do monitor, da
consulta e da câmera.

| Severidade | Destinatário |
|---|---|
| moderado (duplicidade, salto, intervalo, suspensão, ajuste atípico) | **farmácia clínica** |
| alto (erro decimal, dose acima da máxima) | **médico prescritor e farmácia clínica** |
| crítico (com outro canal concordando) | time de resposta rápida |

Um achado de prescrição **vale por 6 h** na fusão, enquanto o medicamento age;
um do monitor vale 15 min. Por isso a fusão liga a prescrição errada ao efeito
que aparece depois:

```
10:00  Possível erro decimal na prescrição: morfina 40 mg IV 4/4h (10× a dose de referência de 4)
       → [alto] médico prescritor e farmácia clínica
10:40  Bradipneia — FR 6 irpm (monitor)
       → [crítico] ESCALONAMENTO · … · concordância entre monitor e prontuario → time de resposta rápida
```

```bash
med-monitor prescricoes exemplo --tipo erro_decimal --paciente PRT-2026-200104 --api http://localhost:8000
```

## Limitações

- **Dados sintéticos:** os formatos de anomalia são plausíveis, mas não vêm de
  incidentes reais de medicação. O recall de 100% nas regras mede a
  implementação, não a cobertura clínica.
- **Sem interações medicamentosas, função renal ou peso.** Por exemplo, a
  vancomicina deveria ser ajustada pelo clearance. São regras de formulário que
  se acrescentam sem mudar o desenho.
- **O formulário é ilustrativo** e precisaria ser substituído pelo do hospital.
