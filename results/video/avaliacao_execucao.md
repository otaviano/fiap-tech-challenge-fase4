# Avaliação — execução incorreta (REHAB24-6, validação por pessoa)

| Exercício | Vídeos · pessoas | Repetições (incorretas) | AUC global | AUC mediana por pessoa | Acurácia balanceada | Sensibilidade | Especificidade | Métricas mais discriminativas |
|---|---|---|---|---|---|---|---|---|
| 1. abdução de braço | 12 · 8 | 168 (83) | 0.81 | 0.87 | 70% | 71% | 68% | extensão do cotovelo (0.65), hesitações no movimento (0.61), elevação do braço oposto (0.58) |
| 6. agachamento | 9 · 9 | 195 (61) | 0.64 | 0.95 | 63% | 59% | 67% | flexão do quadril (direito) (0.72), amplitude dos joelhos (0.70), inclinação do tronco (0.69) |

Outra pessoa visível (YOLOv8, confiança ≥ 0,6, em ≥ 30% dos quadros da repetição) vs rótulo do dataset:

| Corte do rótulo | Positivos | Sensibilidade | Especificidade | Precisão |
|---|---|---|---|---|
| grande parte de outra pessoa (3) | 6/373 | 83% | 96% | 25% |
| parte perceptível ou mais (≥ 2) | 49/373 | 24% | 98% | 60% |

Fração média de quadros com outra pessoa visível, por rótulo: 0: 3% · 1: 0% · 2: 16% · 3: 84%
Repetições com outra pessoa **na área do paciente** (evento clínico, sem rótulo no dataset): 0
