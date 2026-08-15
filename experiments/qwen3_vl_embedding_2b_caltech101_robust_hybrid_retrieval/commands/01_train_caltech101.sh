#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
MODULE="experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval"
CONFIG="$EXPERIMENT_DIR/configs/release/caltech101_robust_hybrid_moe4.yaml"
RUN_DIR="$EXPERIMENT_DIR/runs/caltech101_robust_hybrid_moe4"

cd "$REPO_ROOT"

if [[ -e "$RUN_DIR/last_checkpoint.pt" ]]; then
  echo "Refusing to overwrite an existing Caltech101 checkpoint: $RUN_DIR" >&2
  echo "Move the run directory before starting a new fixed-schedule run." >&2
  exit 2
fi

echo "[1/4] Download/verify Caltech101 and create the fixed retrieval manifest"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$CONFIG" --phase prepare_data

echo "[2/4] Cache frozen Qwen3-VL Teacher embeddings"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$CONFIG" --phase cache_teacher_embeddings

echo "[3/4] Train 40 natural epochs without per-epoch test selection"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$CONFIG" --phase train \
  2>&1 | tee "$RUN_DIR/train.log"

echo "[4/4] Evaluate the chronologically fixed final EMA checkpoint"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$CONFIG" --phase evaluate \
  --checkpoint "$RUN_DIR/ema_last_checkpoint.pt"

echo "Caltech101 retrieval run complete: $RUN_DIR"
