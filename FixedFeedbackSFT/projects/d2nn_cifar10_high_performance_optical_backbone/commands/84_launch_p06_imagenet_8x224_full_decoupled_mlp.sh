#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_8x224_full_decoupled_mlp"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_pretraining.*p06_imagenet_8x224_full_decoupled_mlp.yaml" >/dev/null; then
  echo "P06-F3B decoupled MLP is already running; refusing a duplicate." >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-4}" \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/83_run_p06_imagenet_8x224_full_decoupled_mlp.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P06-F3B pid=${pid} physical_gpu=${PHYSICAL_GPU_INDEX:-4}"
