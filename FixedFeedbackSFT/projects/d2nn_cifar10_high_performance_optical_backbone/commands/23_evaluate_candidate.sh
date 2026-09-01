#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

: "${CONFIG_PATH:?Set CONFIG_PATH to a candidate YAML path}"
"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config "${CONFIG_PATH}" \
  --phase evaluate
