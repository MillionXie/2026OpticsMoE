#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-4,5}"
RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_capacity_12x192"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_pretraining.*p06_imagenet_capacity_12x192.yaml" >/dev/null; then
  echo "P06-F2C is already running; refusing a duplicate." >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES}" \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/67_run_p06_imagenet_capacity_12x192.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P06-F2C pid=${pid} physical_gpus=${PHYSICAL_GPU_INDICES}"
