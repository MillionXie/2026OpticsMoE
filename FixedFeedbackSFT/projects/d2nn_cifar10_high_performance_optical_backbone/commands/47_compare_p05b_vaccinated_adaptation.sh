#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.deployment_adaptation \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p05b_vaccinated_deployment_adaptation.yaml \
  --phase compare
