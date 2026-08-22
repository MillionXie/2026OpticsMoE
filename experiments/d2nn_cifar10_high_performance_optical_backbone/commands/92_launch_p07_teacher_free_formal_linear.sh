#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-2,5}"
RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p07_teacher_free_formal_linear"
CONFIG_NAME="p07_teacher_free_formal_linear.yaml"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_pretraining.*${CONFIG_NAME}" >/dev/null; then
  echo "P07-F teacher-free formal training is already running" >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES}" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/91_run_p07_teacher_free_formal_linear.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
echo "$!" > "${RUN_DIR}/launcher.pid"
echo "Launched P07-F pid=$! physical_gpus=${PHYSICAL_GPU_INDICES}"
