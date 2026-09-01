#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

METHODS_CSV="${METHODS_CSV:-noft,bp,fa_pretrained,fa_random}"
CONDITIONS_CSV="${CONDITIONS_CSV:-global_shift_0p125px,global_shift_0p25px,layerwise_shift_0p125px,layerwise_shift_0p25px}"
IFS=',' read -r -a methods <<< "${METHODS_CSV}"
IFS=',' read -r -a conditions <<< "${CONDITIONS_CSV}"

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.deployment_adaptation \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p04_deployment_adaptation_screen.yaml \
  --phase run \
  --methods "${methods[@]}" \
  --conditions "${conditions[@]}"
