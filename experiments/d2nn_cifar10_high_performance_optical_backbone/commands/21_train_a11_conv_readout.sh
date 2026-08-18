#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/a11_pointwise_conv_readout.yaml \
  --phase train
