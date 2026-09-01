#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-4}"
select_gpu
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
"${PYTHON_BIN}" -u -m experiments.d2nn_cifar10_high_performance_optical_backbone.general_backbone_pretraining \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p06_imagenet_8x224_full_decoupled_mlp.yaml \
  --resume
