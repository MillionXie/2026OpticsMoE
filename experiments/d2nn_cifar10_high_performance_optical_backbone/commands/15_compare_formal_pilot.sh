#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.formal_run \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/formal_pilot.yaml \
  --phase compare
