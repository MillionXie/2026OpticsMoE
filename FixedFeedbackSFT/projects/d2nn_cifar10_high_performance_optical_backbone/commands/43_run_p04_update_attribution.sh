#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

ADAPTATION_CONFIG="${ADAPTATION_CONFIG:-FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p04_deployment_adaptation_screen.yaml}"
CONDITIONS_CSV="${CONDITIONS_CSV:-global_shift_0p25px,layerwise_shift_0p25px}"
IFS=',' read -r -a conditions <<< "${CONDITIONS_CSV}"

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.deployment_adaptation_attribution \
  --config "${ADAPTATION_CONFIG}" \
  --conditions "${conditions[@]}"
