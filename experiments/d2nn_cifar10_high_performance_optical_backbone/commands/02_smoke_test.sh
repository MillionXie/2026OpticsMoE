#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

"${PYTHON_BIN}" -m pytest -q \
  experiments/d2nn_cifar10_high_performance_optical_backbone/tests
"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/smoke.yaml \
  --phase train --force
