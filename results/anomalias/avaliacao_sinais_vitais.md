# Avaliação — anomalias em sinais vitais

Fundo real: 10 registros do MIMIC-III, 336.4 h de monitoramento · 30 execuções · 139 eventos clínicos avaliados (29 descartados por sensor sem sinal) · tolerância 10 min

| Detector | Recall | Latência mediana (min) | Achados fora de evento / 24 h | Achados causados por artefato | Carga no registro sem injeção / 24 h |
|---|---|---|---|---|---|
| NEWS2 (limiar clínico) | 72% | 7.0 | 9.5 | 0 | 11.2 |
| Desvio robusto (basal do paciente) | 80% | 6.0 | 2.9 | 0 | 3.6 |
| Isolation Forest (multivariado) | 21% | 13.0 | 0.9 | 0 | 1.5 |
| **Sistema (união)** | 96% | 5.0 | 13.3 | 0 | 16.3 |

Recall por tipo de evento:

| Detector | taquicardia | bradicardia | hipotensao | dessaturacao | taquipneia | deterioracao_combinada |
|---|---|---|---|---|---|---|
| NEWS2 (limiar clínico) | 64% | 50% | 63% | 96% | 80% | 81% |
| Desvio robusto (basal do paciente) | 100% | 88% | 42% | 87% | 73% | 81% |
| Isolation Forest (multivariado) | 16% | 12% | 21% | 17% | 13% | 62% |
| **Sistema (união)** | 100% | 92% | 95% | 96% | 97% | 100% |

n por tipo: taquicardia=25, bradicardia=26, hipotensao=19, dessaturacao=23, taquipneia=30, deterioracao_combinada=16
