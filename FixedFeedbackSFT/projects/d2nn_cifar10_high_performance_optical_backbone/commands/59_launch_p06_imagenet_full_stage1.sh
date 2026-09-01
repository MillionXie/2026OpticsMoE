#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-2,5}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_full_stage1"
mkdir -p "${RUN_DIR}"

if pgrep -af "general_backbone_pretraining.*p06_imagenet_full_stage1.yaml" >/dev/null; then
  echo "P06 full ImageNet stage 1 is already running; refusing a duplicate launch." >&2
  exit 2
fi

nohup env PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/58_run_p06_imagenet_full_stage1.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P06 full ImageNet stage 1 pid=${pid} physical_gpus=${PHYSICAL_GPU_INDICES} nccl_p2p=${NCCL_P2P_DISABLE} nccl_ib=${NCCL_IB_DISABLE} log=${RUN_DIR}/train.log"
