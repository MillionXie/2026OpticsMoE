#!/usr/bin/env bash
set -euo pipefail
export DEPTHS=16
export OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-FixedFeedbackSFT/runs/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/p13_gpu_engineering_depth016_bs1}"
exec bash "$(dirname "$0")/03_gpu_engineering_sweep_bs1.sh"
