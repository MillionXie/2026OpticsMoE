#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ROBUSTNESS_CONFIG="${ROBUSTNESS_CONFIG:-FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p03_deployment_robustness_screen.yaml}"
export CUDA_VISIBLE_DEVICES=""
"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.deployment_robustness \
  --config "${ROBUSTNESS_CONFIG}" \
  --phase compare
