#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

: "${SCREEN_VARIANT:?Set SCREEN_VARIANT to projected, mlp or supervised_mlp}"
case "${SCREEN_VARIANT}" in
  projected)
    CONFIG_NAME="p06_imagenet_8x224_screen_projected.yaml"
    PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-2}"
    ;;
  mlp)
    CONFIG_NAME="p06_imagenet_8x224_screen_mlp.yaml"
    PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-4}"
    ;;
  supervised_mlp)
    CONFIG_NAME="p06_imagenet_8x224_screen_supervised_mlp.yaml"
    PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-5}"
    ;;
  *)
    echo "Unsupported SCREEN_VARIANT=${SCREEN_VARIANT}" >&2
    exit 2
    ;;
esac

select_gpu
"${PYTHON_BIN}" -u -m experiments.d2nn_cifar10_high_performance_optical_backbone.general_backbone_pretraining \
  --config "experiments/d2nn_cifar10_high_performance_optical_backbone/configs/${CONFIG_NAME}" \
  --resume
