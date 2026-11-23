#!/usr/bin/env bash
# Baixa o dataset REHAB24-6 (Zenodo 13305826, CC BY-NC 4.0) para data/external/rehab24-6.
# Citação obrigatória: Černek, Sedmidubský, Budíková — "REHAB24-6: Physical Therapy
# Dataset for Analyzing Pose Estimation Methods", SISAP 2024.
#
# Por padrão baixa só os vídeos RGB (2,65 GB) e a segmentação/rótulos.
# Com --completo, baixa também os esqueletos 2D/3D de referência (~3 GB a mais).
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="$RAIZ/data/external/rehab24-6"
BASE="https://zenodo.org/records/13305826/files"

ARQUIVOS=(Segmentation.csv Segmentation.txt joints_names.txt videos.zip)
if [[ "${1:-}" == "--completo" ]]; then
  ARQUIVOS+=(2d_joints.zip 3d_joints.zip)
fi

mkdir -p "$DESTINO"
for arq in "${ARQUIVOS[@]}"; do
  if [[ -f "$DESTINO/$arq" ]]; then
    echo "já existe: $arq"
    continue
  fi
  echo "baixando: $arq"
  curl -fL --retry 3 -C - -o "$DESTINO/$arq.part" "$BASE/$arq?download=1"
  mv "$DESTINO/$arq.part" "$DESTINO/$arq"
done

for zip in "$DESTINO"/*.zip; do
  pasta="${zip%.zip}"
  [[ -d "$pasta" ]] || unzip -q "$zip" -d "$pasta"
done
echo "REHAB24-6 em $DESTINO"
