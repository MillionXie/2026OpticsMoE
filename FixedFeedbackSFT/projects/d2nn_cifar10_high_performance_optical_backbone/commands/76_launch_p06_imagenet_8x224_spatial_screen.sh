#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-1}"
RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_8x224_screen_spatial"
CONFIG_NAME="p06_imagenet_8x224_screen_spatial.yaml"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_pretraining.*${CONFIG_NAME}" >/dev/null; then
  echo "8x224 spatial screen is already running; refusing a duplicate." >&2
  exit 2
fi
nohup env SCREEN_VARIANT=spatial PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX}" \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/74_run_p06_imagenet_8x224_screen.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched 8x224 spatial screen pid=${pid} physical_gpu=${PHYSICAL_GPU_INDEX}"
