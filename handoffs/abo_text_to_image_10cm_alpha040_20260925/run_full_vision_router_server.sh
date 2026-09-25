#!/usr/bin/env bash
set -euo pipefail
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
WORKTREE="$ROOT/.worktrees/t08_text_to_image_20260920"
OUT="$ROOT/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925/01_vision_router_export"
cd "$WORKTREE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$WORKTREE:$ROOT/.codex_tmp/t08_full_test_20260925"
exec /home/guest3/miniconda3/envs/xml/bin/python -u "$ROOT/.codex_tmp/t08_full_test_20260925/export_full_vision_router.py" \
  --config "$WORKTREE/LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_text_to_image_64_10cm_compact_e0p5.yaml" \
  --checkpoint "$ROOT/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/optical_text_to_image_64_10cm_compact_e0p5_seed42_20260925/best_checkpoint.pt" \
  --data-root "$ROOT/data/abo_easy100_dataset_20260906" \
  --output "$OUT" --batch-size 8
