#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

POLL_SECONDS="${POLL_SECONDS:-30}"
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "POLL_SECONDS must be a positive integer, got ${POLL_SECONDS}" >&2
  exit 2
}

result_root="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/formal_a13_high_performance"
methods=(noft bp fa_pretrained fa_random)
seeds=(2026 2027 2028)

while true; do
  missing=()
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      result_path="${result_root}/${method}/seed_${seed}/result.json"
      [[ -f "${result_path}" ]] || missing+=("${method}/seed_${seed}")
    done
  done
  if (( ${#missing[@]} == 0 )); then
    break
  fi
  printf '%s waiting for %s\n' "$(date -Is)" "${missing[*]}"
  sleep "${POLL_SECONDS}"
done

echo "All 12 formal results are present; building the locked four-group comparison."
bash "${SCRIPT_DIR}/30_compare_a13_formal.sh"
