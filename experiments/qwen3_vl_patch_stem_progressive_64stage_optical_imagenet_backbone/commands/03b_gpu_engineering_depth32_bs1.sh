#!/usr/bin/env bash
set -euo pipefail
export DEPTHS=32
export OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/runs/p13_gpu_engineering_depth032_bs1}"
exec bash "$(dirname "$0")/03_gpu_engineering_sweep_bs1.sh"
