#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
select_gpu

: "${METHODS_CSV:?Set METHODS_CSV to a comma-separated subset of noft,bp,fa_pretrained,fa_random}"
TRAINING_SEED="${TRAINING_SEED:-2026}"
case "${TRAINING_SEED}" in
  2026) ;;
  *) echo "P03 screen is locked to TRAINING_SEED=2026" >&2; exit 2 ;;
esac

IFS=',' read -r -a methods <<< "${METHODS_CSV}"
for method in "${methods[@]}"; do
  case "${method}" in
    noft|bp|fa_pretrained|fa_random) ;;
    *) echo "Unsupported P03 method=${method}" >&2; exit 2 ;;
  esac
done

"${PYTHON_BIN}" -m experiments.d2nn_cifar10_high_performance_optical_backbone.deployment_robustness \
  --config experiments/d2nn_cifar10_high_performance_optical_backbone/configs/p03_deployment_robustness_screen.yaml \
  --phase run \
  --methods "${methods[@]}" \
  --training-seeds "${TRAINING_SEED}"
