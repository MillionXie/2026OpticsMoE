#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-4}"
select_gpu
"${PYTHON_BIN}" -u -m experiments.d2nn_cifar10_high_performance_optical_backbone.general_backbone_robustness \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p06_imagenet_8x224_epoch4_shift_screen.yaml
