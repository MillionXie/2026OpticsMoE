#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

: "${RUN_SEED:?Set RUN_SEED to 2026 or 2027}"
case "${RUN_SEED}" in
  2026|2027) ;;
  *)
    echo "A13 replication RUN_SEED must be 2026 or 2027" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/a13_multiseed_replication.yaml \
  --phase train \
  --seed "${RUN_SEED}"
