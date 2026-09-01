#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

launch_one() {
  local name="$1"
  local gpu="$2"
  local config="FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p07_teacher_free_screen_${name}.yaml"
  local run_dir="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p07_teacher_free_screen_${name}"
  local pattern="general_backbone_pretraining.*p07_teacher_free_screen_${name}.yaml"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/result.json" ]]; then
    echo "P07 ${name} already complete"
    return
  fi
  if pgrep -af "${pattern}" >/dev/null; then
    echo "P07 ${name} already running"
    return
  fi
  nohup env PHYSICAL_GPU_INDEX="${gpu}" P07_CONFIG="${config}" \
    bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/87_run_p07_teacher_free_head_screen.sh \
    > "${run_dir}/train.log" 2>&1 &
  echo "$!" > "${run_dir}/launcher.pid"
  echo "Launched P07 ${name} pid=$! physical_gpu=${gpu}"
}

launch_one linear "${LINEAR_GPU_INDEX:-2}"
launch_one mlp "${MLP_GPU_INDEX:-4}"
launch_one conv_mlp "${CONV_MLP_GPU_INDEX:-5}"
