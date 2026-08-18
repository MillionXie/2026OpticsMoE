#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

CONFIG="experiments/d2nn_cifar10_high_performance_optical_backbone/configs/formal_smoke.yaml"
MODULE="experiments.d2nn_cifar10_high_performance_optical_backbone.formal_run"

"${PYTHON_BIN}" -m "${MODULE}" --config "${CONFIG}" --phase head_warmup --force
for method in noft bp fa_pretrained fa_random; do
  "${PYTHON_BIN}" -m "${MODULE}" --config "${CONFIG}" --phase run --method "${method}" --force
done
"${PYTHON_BIN}" -m "${MODULE}" --config "${CONFIG}" --phase compare
