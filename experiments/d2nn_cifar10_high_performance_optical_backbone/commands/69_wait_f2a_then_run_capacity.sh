#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

while pgrep -f "general_backbone_pretraining.*p06_imagenet_full_supervised_refine.yaml" >/dev/null; do
  echo "[$(date -Iseconds)] waiting for P06-F2A before starting the capacity run"
  sleep 120
done

while true; do
  gpu4_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '5p')"
  gpu5_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '6p')"
  if (( gpu4_memory < 1024 && gpu5_memory < 1024 )); then
    break
  fi
  echo "[$(date -Iseconds)] waiting for free GPUs 4/5: ${gpu4_memory}/${gpu5_memory} MiB"
  sleep 120
done

bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/68_launch_p06_imagenet_capacity_12x192.sh
