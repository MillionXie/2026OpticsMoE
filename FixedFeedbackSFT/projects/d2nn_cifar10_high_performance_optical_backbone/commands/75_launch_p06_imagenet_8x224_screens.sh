#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

launch_screen() {
  local variant="$1"
  local gpu="$2"
  local run_name="$3"
  local config_name="p06_imagenet_8x224_screen_${variant}.yaml"
  local run_dir="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/${run_name}"
  if pgrep -af "general_backbone_pretraining.*${config_name}" >/dev/null; then
    echo "8x224 ${variant} screen is already running; refusing a duplicate." >&2
    return 2
  fi
  mkdir -p "${run_dir}"
  nohup env SCREEN_VARIANT="${variant}" PHYSICAL_GPU_INDEX="${gpu}" \
    bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/74_run_p06_imagenet_8x224_screen.sh \
    > "${run_dir}/train.log" 2>&1 &
  local pid=$!
  echo "${pid}" > "${run_dir}/launcher.pid"
  echo "Launched 8x224 ${variant} screen pid=${pid} physical_gpu=${gpu}"
}

launch_screen projected "${PROJECTED_GPU_INDEX:-2}" p06_imagenet_8x224_screen_projected
launch_screen mlp "${MLP_GPU_INDEX:-4}" p06_imagenet_8x224_screen_mlp
launch_screen supervised_mlp "${SUPERVISED_MLP_GPU_INDEX:-5}" p06_imagenet_8x224_screen_supervised_mlp
