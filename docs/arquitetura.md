# Arquitetura — fluxo multimodal

```mermaid
flowchart LR
    subgraph Fontes
        V[Vídeo<br/>fisioterapia / cirurgia]
        A[Áudio<br/>consulta]
        S[Sinais vitais<br/>MIMIC-III replay]
        P[Prescrições<br/>prontuário]
    end

    subgraph Analisadores
        VP[video.pose<br/>YOLOv8-pose · COCO-17]
        VO[video.objetos<br/>YOLOv8]
        AT[audio.transcricao<br/>Azure Speech to Text]
        AX[audio.texto<br/>Azure Text Analytics]
        AC[audio.acustica<br/>librosa]
        SV[anomalias.sinais_vitais<br/>limiar + z-score + Isolation Forest]
        PR[anomalias.prescricoes]
        MV[anomalias.movimentacao]
    end

    V --> VP & VO
    VP --> MV
    A --> AT --> AX
    A --> AC
    S --> SV
    P --> PR

    VP & VO & AX & AC & SV & PR & MV -->|Achado| F[fusao<br/>late fusion por paciente/janela]
    F -->|Alerta| D[alertas<br/>despacho]
    D --> AU[(auditoria.jsonl)]
    D --> SSE[API · SSE /stream]
    SSE --> EQ[Painel da equipe médica]
    EQ -->|Ciente| D
    EXT[outras modalidades<br/>POST /achados] -->|Achado| F
    VP & VO --> RV[Relatório do procedimento]
```

## Contratos

- **`Achado`** (`domain.py`) — saída de todo analisador: paciente, modalidade,
  tipo, escore 0–1, fonte (modelo/serviço) e evidência.
- **`Alerta`** — saída da fusão: severidade (escala da Fase 3), mensagem, e os
  ids dos achados que o fundamentam. `multimodal=True` quando mais de uma
  modalidade concorda.

Nenhum analisador conhece os outros; a fusão não conhece nenhum SDK. Isso
permite testar cada peça isoladamente e rodar a demonstração mesmo com uma
modalidade fora do ar.

## Estado atual

- Sinais vitais → fusão → alertas → painel: **implementado** ([sinais vitais](sinais-vitais.md), [fusão e alertas](fusao-alertas.md)).
- Áudio, vídeo, prescrições e movimentação: entram na mesma fusão por `POST /achados` (ou chamando `MotorFusao.processar` direto) assim que os analisadores existirem.
