# Análise de áudio das consultas

Entrega técnica 2 do enunciado:
- processar áudios de consultas médicas;
- detectar alterações vocais indicativas de condições médicas, como cansaço e
  dificuldade respiratória;
- usar **Azure Speech to Text** para transcrever e analisar os áudios;
- identificar termos críticos e sentimentos com **Azure Text Analytics**.

Código em [`src/med_monitor/audio/`](../src/med_monitor/audio/) e
[`src/med_monitor/azure/`](../src/med_monitor/azure/).

## Fluxo

```
consulta.wav ─┬─▶ acústica (librosa, local) ────── vs basal do paciente ─▶ fala_ofegante · fadiga_vocal   (AUDIO)
              │
              └─▶ Azure Speech to Text (pt-BR) ─▶ transcrição ─┬─▶ léxico do hospital + negação ─▶ termo_critico (TEXTO)
                  sem Azure: roteiro .txt                       └─▶ Azure Text Analytics
                                                                    · sentimento + opinion mining (pt-BR)
                                                                    · Text Analytics for Health (pt) ─▶ sintoma_relatado
                                                                                                         sentimento_negativo
                                                       achados ─▶ POST /achados ─▶ fusão ─▶ alerta no painel
```

O **Azure** responde à pergunta *o que* o paciente disse. A **acústica**
responde *como* ele disse. Um paciente que diz "estou bem" com a fala
entrecortada de quem está sem ar é justamente o caso que o texto sozinho perde.

## Alterações vocais (acústica)

| Feature | Medida | Sinaliza |
|---|---|---|
| grupo de fala médio | duração dos trechos de fala entre pausas | **dispneia**: o paciente para para respirar |
| pausas por minuto | pausas ≥ 200 ms | dispneia, fadiga |
| inspirações por minuto | trechos sem voz, ≥ 150 ms, ≥ 6 dB mais fracos que a fala, centroide < 3 kHz | **dispneia**: respiração audível |
| taxa de articulação | núcleos silábicos por segundo de fala (de Jong & Wempe, 2009) | **fadiga**: fala lenta |
| F0 e variabilidade | pYIN; variabilidade em semitons | fadiga: voz grave e monótona |
| intensidade | RMS dos trechos de fala | fadiga: voz fraca |
| planicidade espectral | nos trechos vozeados | soprosidade |

Três decisões de método:

1. **Fala é energia com vozeamento.** Um detector de voz só por energia confunde
   respiração audível com fala. No paciente dispneico, em que a inspiração
   ruidosa preenche a pausa, ele veria *menos* pausas. Foi exatamente o que a
   primeira versão fez com a dispneia leve: detectou 1 caso em 12. Com o
   vozeamento do pYIN separando voz de sopro, passaram a ser 12 em 12.
2. **Respiração não é fricativa.** "S", "f" e "ch" também não têm voz e se
   alongam na fala lenta. A respiração se distingue por ser mais fraca e mais
   grave (centroide < 3 kHz). Sem esse filtro, a fadiga gerava falsas
   "inspirações".
3. **Comparação com o basal do próprio paciente.** Valores absolutos dependem
   de microfone, sala e falante. O que é clínico é a *mudança*. A alteração só
   é declarada quando **≥ 2 critérios concordam**. Pausas a mais com fala lenta
   e sem respiração audível são atribuídas à **fadiga**, não à dispneia: na
   dispneia, a articulação dentro de cada grupo de fala continua no ritmo
   normal.

### Avaliação

Não existe dataset público de consultas em pt-BR com rótulo de fadiga ou
dispneia. Geramos então consultas sintéticas com **espeak-ng**, com a condição
conhecida ([`audio/amostras.py`](../src/med_monitor/audio/amostras.py)):
- 4 vozes (2 masculinas, 2 femininas);
- 3 roteiros por condição, sempre diferentes do roteiro do basal;
- versões **leves** de cada alteração, para medir a sensibilidade;
- piso de ruído de sala e ruído de inspiração nas pausas da dispneia.

| Condição | Com basal | Sem basal |
|---|---|---|
| normal | **0/12 falsos alarmes** | 0/12 falsos alarmes |
| fadiga | **12/12**, 0 confusões | 0/12 |
| fadiga leve | 4/12, 2 confusões | 0/12 |
| dispneia | **12/12**, 0 confusões | 12/12 |
| dispneia leve | **12/12**, 0 confusões | 12/12 |

**Leitura:**
- Com basal, o detector é **específico**: nenhum falso alarme e nenhuma
  confusão nas alterações marcadas.
- A **fadiga leve** é o limite. Fala 15% mais lenta e voz 2,5 dB mais fraca
  ficam dentro da variação que a mesma pessoa tem de um dia para outro.
- **Sem basal, a fadiga é invisível**, porque fala lenta não é anormal por si
  só. Já a dispneia aparece mesmo sem basal: respiração audível entre frases
  curtas é anormal em qualquer pessoa. Isso sustenta a recomendação clínica de
  gravar uma consulta basal na admissão.
- **Limitação:** a voz é sintética e as alterações são caricatas. Isto valida a
  *mecânica*, não a acurácia clínica. Para o vídeo, o ideal é o grupo gravar
  os roteiros com a própria voz.

```bash
med-monitor audio avaliar          # ~1 min; resultados em results/audio/avaliacao_acustica.md
```

**Disartria** (fala arrastada por lesão neurológica) aparece no desafio do
enunciado, mas exige medidas de articulação que um proxy de energia não captura
com confiança. Ela fica fora do detector acústico e é coberta pelo texto
("fala enrolada", "boca torta" e outros termos estão no léxico neurológico).

## Termos críticos e sentimento (texto)

**Léxico do hospital** ([`audio/termos.py`](../src/med_monitor/audio/termos.py)),
que roda sempre, com ou sem Azure. Ele tem 7 categorias com escore por urgência:

- risco de autolesão (0,95, sempre **crítico**);
- respiratório, cardiovascular e neurológico (0,80);
- fadiga e dor (0,55);
- geral (0,45).

Inclui coloquialismos ("coração disparado", "puxando o ar").

**Negação é tratada.** "Não tenho falta de ar" não vira achado: um marcador
("não", "nem", "nunca", "sem") até 3 palavras antes do termo, na mesma oração,
nega o termo. Termos que já contêm a negação ("não consigo respirar") não são
afetados. É a mesma lição da Fase 2, em que o avaliador reprovava o disclaimer
correto por ignorar a negação.

**Azure Text Analytics** ([`azure/language.py`](../src/med_monitor/azure/language.py)):

- **Sentimento com opinion mining** (`pt-BR`): sentimento negativo com
  confiança ≥ 0,75 vira o achado `sentimento_negativo`, de baixa severidade
  (sinal de apoio, não de alarme).
- **Text Analytics for Health** (`language="pt"`, decisão D5): entidades
  clínicas com **asserção**. O Azure também marca a negação, por isso "febre"
  negada não entra. Um sintoma que o Azure encontrou e o léxico também confirma
  ganha corroboração (`lexico-hospital+azure-ta4h`). Um sintoma que só o Azure
  encontrou vira `sintoma_relatado`.

## Integração com o Azure e modo offline

| Situação | Transcrição | Texto |
|---|---|---|
| Azure configurado (`.env`) | Azure Speech to Text, reconhecimento contínuo, pt-BR, com tempo por palavra | léxico + Azure (sentimento, TA4H) |
| Sem Azure | roteiro `.txt` ao lado do `.wav` (amostras) | só o léxico |
| Azure falha (rede, cota) | cai para o roteiro, se houver; o erro fica no relatório | só o léxico |

O relatório de cada consulta registra em `servicos` **o que foi de fato
usado**. Nunca se finge que a nuvem foi chamada (decisão D4).

> **Status:** os clientes Azure estão implementados e testados com dublês, mas
> **ainda não rodaram contra o serviço real**, porque a conta está sendo
> criada. Depois do `terraform apply` e do `.env`, basta rodar `med-monitor
> audio analisar` de novo: o pipeline passa a usar o Azure sem mudança de código.

## Integração com a fusão

Os achados entram na mesma fusão dos sinais vitais (`POST /achados`). Na
fusão, **áudio e texto são o mesmo canal físico** (a consulta): eles se
corroboram, com teto em *alto*, mas não contam como evidência independente.
*Crítico* exige outro canal concordando, como o monitor.

Exemplo real com o painel rodando: a UTI-02 tinha uma dessaturação aberta como
*moderado* (SpO2 89,5%) quando a consulta entrou.

```bash
med-monitor audio analisar data/raw/audio/PRT-2026-200102_atual.wav \
    --basal data/raw/audio/PRT-2026-200102_basal.wav --api http://localhost:8000
#  → alerta [critico] ESCALONAMENTO · Fala ofegante na consulta (grupos de fala ≥ 35% mais
#    curtos que o basal; pausas ≥ 50% mais frequentes; respiração audível ≥ 4/min acima do basal)
#    · Também: Dessaturação — SpO2 89,5% · Severidade elevada: concordância entre consulta e monitor
#    → time de resposta rápida
```

## Como executar

```bash
pip install -e ".[audio,dev]"
sudo dnf install espeak-ng         # ou apt install espeak-ng — só para gerar as amostras
med-monitor audio amostras         # 3 pacientes: basal + consulta atual (dispneia, fadiga, estável)
med-monitor audio analisar data/raw/audio/PRT-2026-200103_atual.wav --basal data/raw/audio/PRT-2026-200103_basal.wav
med-monitor audio avaliar
```
