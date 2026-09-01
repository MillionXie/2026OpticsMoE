#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/a13_multiseed_replication.yaml \
  --phase train \
  --seed 2028
