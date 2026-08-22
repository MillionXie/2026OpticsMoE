#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-1,5}"
RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p06_imagenet_8x224_full_spatial"
CONFIG_NAME="p06_imagenet_8x224_full_spatial.yaml"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_pretraining.*${CONFIG_NAME}" >/dev/null; then
  echo "8x224 full spatial training is already running; refusing a duplicate." >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES}" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/77_run_p06_imagenet_8x224_full_spatial.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched 8x224 full spatial training pid=${pid} physical_gpus=${PHYSICAL_GPU_INDICES}"
