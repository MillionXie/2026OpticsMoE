#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

EXTRA_ARGS=()
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.formal_run \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/formal_a13_high_performance.yaml \
  --phase head_warmup "${EXTRA_ARGS[@]}"
