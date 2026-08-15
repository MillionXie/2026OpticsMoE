#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Override with, for example, GPU_ID=2 PYTHON_BIN=python.
GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
MODULE="experiments.qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval"
STAGE1_CONFIG="$EXPERIMENT_DIR/configs/retrain/stage1_grocery31_pretrain.yaml"
STAGE2_CONFIG="$EXPERIMENT_DIR/configs/retrain/stage2_grocery10_finetune.yaml"
STAGE1_RUN="$EXPERIMENT_DIR/runs/robust_hybrid_grocery31_pretrain"
STAGE2_RUN="$EXPERIMENT_DIR/runs/robust_hybrid_grocery10_from_grocery31"
LEGACY31_CACHE="$REPO_ROOT/experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/release_moe4_grocery31_pretrain/teacher_cache"
CURRENT10_CACHE="$EXPERIMENT_DIR/runs/robust_hybrid_moe4_from_scratch/teacher_cache"

cd "$REPO_ROOT"

if [[ -e "$STAGE1_RUN/last_checkpoint.pt" || -e "$STAGE2_RUN/last_checkpoint.pt" ]]; then
  echo "Refusing to overwrite an existing two-stage run." >&2
  echo "Move the existing stage output directories or set new output_dir values." >&2
  exit 2
fi

reuse_teacher_cache() {
  local source_dir="$1"
  local target_dir="$2"
  if [[ -f "$source_dir/teacher_embeddings.pt" && -f "$source_dir/metadata.json" ]]; then
    mkdir -p "$target_dir"
    cp "$source_dir/teacher_embeddings.pt" "$target_dir/teacher_embeddings.pt"
    cp "$source_dir/metadata.json" "$target_dir/metadata.json"
  fi
}

echo "[1/6] Validate/cache frozen Teacher embeddings for Grocery31"
reuse_teacher_cache "$LEGACY31_CACHE" "$STAGE1_RUN/teacher_cache"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$STAGE1_CONFIG" --phase cache_teacher_embeddings

echo "[2/6] Train robust hybrid model on Grocery31 (fixed 60 natural epochs)"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$STAGE1_CONFIG" --phase train \
  2>&1 | tee "$STAGE1_RUN/stage1_train.log"

echo "[3/6] Evaluate fixed stage-1 EMA checkpoint"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$STAGE1_CONFIG" --phase evaluate \
  --checkpoint "$STAGE1_RUN/ema_last_checkpoint.pt"

echo "[4/6] Validate/cache frozen Teacher embeddings for Grocery10"
reuse_teacher_cache "$CURRENT10_CACHE" "$STAGE2_RUN/teacher_cache"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$STAGE2_CONFIG" --phase cache_teacher_embeddings

echo "[5/6] Fine-tune Grocery31 EMA weights on Grocery10"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$STAGE2_CONFIG" --phase train \
  --resume-checkpoint "$STAGE1_RUN/ema_last_checkpoint.pt" \
  2>&1 | tee "$STAGE2_RUN/stage2_train.log"

echo "[6/6] Evaluate fixed stage-2 EMA checkpoint"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m "$MODULE" \
  --config "$STAGE2_CONFIG" --phase evaluate \
  --checkpoint "$STAGE2_RUN/ema_last_checkpoint.pt"

echo "Two-stage retraining complete: $STAGE2_RUN"
