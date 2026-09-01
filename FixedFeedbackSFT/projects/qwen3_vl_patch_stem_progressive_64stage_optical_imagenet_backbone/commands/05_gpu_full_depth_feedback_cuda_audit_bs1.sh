#!/usr/bin/env bash
set -euo pipefail

# Engineering-only 64/100-stage full-depth backward audit of all three optical
# feedback modes. Every stage executes at alpha=1. It does not read ImageNet,
# save a trainable checkpoint, or launch pretraining.
export DEPTHS="${DEPTHS:-64,100}"
export FEEDBACK_METHODS="${FEEDBACK_METHODS:-bp_current,fa_source,fa_random}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ "${ALPHA_MODE:-full_depth}" != "full_depth" ]]; then
  echo "05 full-depth audit refuses ALPHA_MODE other than full_depth." >&2
  exit 2
fi
export ALPHA_MODE="full_depth"
export OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-FixedFeedbackSFT/runs/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/p13_full_depth_feedback_cuda_audit_bs1}"
exec bash "$(dirname "$0")/03_gpu_engineering_sweep_bs1.sh"
