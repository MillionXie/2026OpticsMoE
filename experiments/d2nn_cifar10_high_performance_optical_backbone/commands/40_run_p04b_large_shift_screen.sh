#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

METHODS_CSV="${METHODS_CSV:-noft,bp,fa_pretrained,fa_random}"
: "${CONDITIONS_CSV:?Set CONDITIONS_CSV to a comma-separated subset of the P04-S2 conditions}"
IFS=',' read -r -a methods <<< "${METHODS_CSV}"
IFS=',' read -r -a conditions <<< "${CONDITIONS_CSV}"

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.deployment_adaptation \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/p04b_deployment_adaptation_large_shift_screen.yaml \
  --phase run \
  --methods "${methods[@]}" \
  --conditions "${conditions[@]}"
