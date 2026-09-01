#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

RESULT_ROOT="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p04b_deployment_adaptation_large_shift_screen"
EXPECTED_RESULTS=24
while true; do
  completed="$(find "${RESULT_ROOT}" -name result.json -type f 2>/dev/null | wc -l)"
  echo "[p04b_wait] completed=${completed}/${EXPECTED_RESULTS}"
  if [[ "${completed}" -ge "${EXPECTED_RESULTS}" ]]; then
    break
  fi
  if ! pgrep -f 'p04b_deployment_adaptation_large_shift_screen.yaml.*--phase run' >/dev/null; then
    echo "P04-S2 launchers exited before all results completed" >&2
    exit 1
  fi
  sleep 60
done

bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/41_compare_p04b_large_shift_screen.sh
