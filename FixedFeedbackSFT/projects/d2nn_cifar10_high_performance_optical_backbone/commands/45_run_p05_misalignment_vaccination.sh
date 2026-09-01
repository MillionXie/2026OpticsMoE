#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

RUN_SEED="${RUN_SEED:-2026}"

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.misalignment_vaccination \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p05_misalignment_vaccination.yaml \
  --seed "${RUN_SEED}"
