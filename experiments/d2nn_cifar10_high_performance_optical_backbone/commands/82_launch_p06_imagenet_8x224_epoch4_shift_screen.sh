#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p06_imagenet_8x224_epoch4_shift_screen"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_robustness.*p06_imagenet_8x224_epoch4_shift_screen.yaml" >/dev/null; then
  echo "P06 8x224 shift screen is already running; refusing a duplicate." >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-4}" \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/81_run_p06_imagenet_8x224_epoch4_shift_screen.sh \
  > "${RUN_DIR}/run.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P06 8x224 epoch-4 shift screen pid=${pid} physical_gpu=${PHYSICAL_GPU_INDEX:-4}"
