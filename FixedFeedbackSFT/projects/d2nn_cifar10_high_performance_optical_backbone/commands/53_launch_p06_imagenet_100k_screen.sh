#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-3,5}"
RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_100k_screen"
mkdir -p "${RUN_DIR}"

if pgrep -af "general_backbone_pretraining.*p06_imagenet_100k_screen.yaml" >/dev/null; then
  echo "P06 100k screen is already running; refusing a duplicate launch." >&2
  exit 2
fi

nohup env PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES}" \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/52_run_p06_imagenet_100k_screen.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P06 100k screen pid=${pid} physical_gpus=${PHYSICAL_GPU_INDICES} log=${RUN_DIR}/train.log"
