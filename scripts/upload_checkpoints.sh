#!/usr/bin/env bash
# Upload 75 epoch checkpoints to HuggingFace, one at a time.
# Retries up to 3 times on failure. Skips already-uploaded epochs on re-run.
#
# Usage:
#   ./scripts/upload_checkpoints.sh
#
# Prerequisites:
#   - hf CLI installed and logged in (hf login)
#   - Repo exists: soyeb-jim285/ocr-visualization-models

set -euo pipefail

REPO="soyeb-jim285/ocr-visualization-models"
SRC="public/models/checkpoints"
REMOTE_PREFIX="bn_emnist_cnn/checkpoints"
MAX_RETRIES=3

cd "$(git rev-parse --show-toplevel)"

if [ ! -d "$SRC" ]; then
  echo "Error: $SRC not found"
  exit 1
fi

TOTAL=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d | wc -l)
COUNT=0
FAILED=0

echo "Uploading $TOTAL epoch checkpoints to $REPO/$REMOTE_PREFIX"
echo ""

for epoch_dir in "$SRC"/epoch-*; do
  [ -d "$epoch_dir" ] || continue
  epoch_name=$(basename "$epoch_dir")
  COUNT=$((COUNT + 1))

  success=false
  for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[$COUNT/$TOTAL] Uploading $epoch_name (attempt $attempt)..."
    if hf upload "$REPO" "$epoch_dir/model.onnx" "$REMOTE_PREFIX/$epoch_name/model.onnx" --quiet 2>&1; then
      success=true
      break
    fi
    echo "  Retry in 5s..."
    sleep 5
  done

  if [ "$success" = false ]; then
    echo "  FAILED after $MAX_RETRIES attempts: $epoch_name"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "Done! Uploaded $((COUNT - FAILED))/$COUNT checkpoints."
[ "$FAILED" -gt 0 ] && echo "Failed: $FAILED (re-run script to retry)"
