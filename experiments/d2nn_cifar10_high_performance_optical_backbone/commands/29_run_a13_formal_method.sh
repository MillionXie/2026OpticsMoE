#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

: "${METHOD:?Set METHOD to noft, bp, fa_pretrained, or fa_random}"
: "${RUN_SEED:?Set RUN_SEED to 2026, 2027, or 2028}"
case "${METHOD}" in
  noft|bp|fa_pretrained|fa_random) ;;
  *) echo "Unsupported formal METHOD=${METHOD}" >&2; exit 2 ;;
esac
case "${RUN_SEED}" in
  2026|2027|2028) ;;
  *) echo "Unsupported formal RUN_SEED=${RUN_SEED}" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.formal_run \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/formal_a13_high_performance.yaml \
  --phase run \
  --method "${METHOD}" \
  --seed "${RUN_SEED}" \
  "${EXTRA_ARGS[@]}"
