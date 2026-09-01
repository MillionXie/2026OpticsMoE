#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

vaccination_result="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p05_misalignment_vaccination/seed_2026/result.json"
adaptation_root="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p05b_vaccinated_deployment_adaptation"
expected_results=16
: "${GLOBAL_GPU_INDEX:=3}"
: "${LAYERWISE_GPU_INDEX:=4}"
mkdir -p "${adaptation_root}"

while [[ ! -f "${vaccination_result}" ]]; do
  echo "[$(date -Iseconds)] waiting for P05 vaccination"
  sleep 60
done

GLOBAL_GPU_INDEX="${GLOBAL_GPU_INDEX}" LAYERWISE_GPU_INDEX="${LAYERWISE_GPU_INDEX}" \
  bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/48_launch_p05b_vaccinated_adaptation_dual_gpu.sh

while true; do
  completed="$(find "${adaptation_root}" -name result.json -type f 2>/dev/null | wc -l)"
  echo "[$(date -Iseconds)] P05-B completed=${completed}/${expected_results}"
  if [[ "${completed}" -ge "${expected_results}" ]]; then
    break
  fi
  sleep 60
done

bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/47_compare_p05b_vaccinated_adaptation.sh
echo "[$(date -Iseconds)] P05 prevention + calibration pipeline completed"
