#!/usr/bin/env bash
# Download the SAM 2.1 tiny checkpoint from Meta's official release into models/.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
OUT="models/sam2.1_hiera_tiny.pt"
mkdir -p models
if [ -f "$OUT" ]; then
  echo "already have $OUT"
  exit 0
fi
curl -L --fail -o "$OUT.part" "$URL"
mv "$OUT.part" "$OUT"
echo "saved $OUT"
