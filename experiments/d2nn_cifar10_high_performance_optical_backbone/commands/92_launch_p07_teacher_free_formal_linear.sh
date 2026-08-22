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
pid="$!"
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P07-F pid=${pid} physical_gpus=${PHYSICAL_GPU_INDICES}"
# Do not return during the short launcher->torchrun hand-off. A watcher
# started immediately after this command must see the real training pattern.
for _ in $(seq 1 30); do
  if pgrep -af "general_backbone_pretraining.*${CONFIG_NAME}" >/dev/null; then
    echo "Confirmed P07-F torchrun is visible"
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "P07-F launcher exited before torchrun became visible" >&2
    exit 1
  fi
  sleep 1
done
echo "P07-F launcher is alive but torchrun was not visible after 30 seconds" >&2
exit 1
