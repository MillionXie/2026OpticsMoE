#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES=""
"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/a13_multiseed_replication.yaml \
  --phase aggregate
