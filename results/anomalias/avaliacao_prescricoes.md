# Avaliação — anomalias em prescrições

Treino: 600 internações sem anomalia (histórico) · teste: 800 internações novas, 60% com uma anomalia injetada

| Detector | ajuste_atipico | dose_acima_maxima | duplicidade | erro_decimal | frequencia_inesperada | salto_dose | suspensao_abrupta | Recall | Alertas em internações sem anomalia |
|---|---|---|---|---|---|---|---|---|---|
| regras | 0/49 | 83/83 | 91/91 | 116/116 | 72/72 | 34/34 | 48/48 | 90% | 23/1997 (1.1%) |
| regras + variação aprendida | 9/49 | 83/83 | 91/91 | 116/116 | 72/72 | 34/34 | 48/48 | 92% | 24/1997 (1.2%) |

Variação típica aprendida por medicamento: insulina regular ±41% (n=164), hidrocortisona ±16% (n=83), prednisona ±19% (n=210), clonazepam ±19% (n=130), vancomicina ±16% (n=39), enoxaparina ±16% (n=32)
