#!/usr/bin/env bash
set -euo pipefail
: "${PHYSICAL_GPU_INDEX:?Set PHYSICAL_GPU_INDEX to the nvidia-smi GPU index}"
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((PHYSICAL_GPU_INDEX + 1))p")"
: "${GPU_UUID:?Could not resolve the requested physical GPU}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 
  --config experiments/d2nn_cifar100c10_fixed_feedback_20stage400/configs/main.yaml 
  --phase run --method fa_random
