#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

: "${P07_CONFIG:?Set P07_CONFIG to one teacher-free screen YAML}"
select_gpu
"${PYTHON_BIN}" -u -m experiments.d2nn_cifar10_high_performance_optical_backbone.general_backbone_pretraining \
  --config "${P07_CONFIG}" --resume
