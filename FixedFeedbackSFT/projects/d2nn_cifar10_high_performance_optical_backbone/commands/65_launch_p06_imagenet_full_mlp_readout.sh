#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-4}"
RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_full_mlp_readout"
mkdir -p "${RUN_DIR}"
if pgrep -af "general_backbone_pretraining.*p06_imagenet_full_mlp_readout.yaml" >/dev/null; then
  echo "P06-F2B is already running; refusing a duplicate." >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX}" \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/64_run_p06_imagenet_full_mlp_readout.sh \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/launcher.pid"
echo "Launched P06-F2B pid=${pid} physical_gpu=${PHYSICAL_GPU_INDEX}"
