#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/logs

vaccinated_checkpoint="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p05_misalignment_vaccination/seed_2026/best.pt"
: "${GLOBAL_GPU_INDEX:=3}"
: "${LAYERWISE_GPU_INDEX:=4}"
if [[ ! -f "${vaccinated_checkpoint}" ]]; then
  echo "Missing vaccinated checkpoint: ${vaccinated_checkpoint}" >&2
  exit 1
fi

PHYSICAL_GPU_INDEX="${GLOBAL_GPU_INDEX}" \
CONDITIONS_CSV=heldout_global_shift_1px,heldout_global_shift_2px \
nohup bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/46_run_p05b_vaccinated_adaptation.sh \
  > FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/logs/p05b_global_gpu${GLOBAL_GPU_INDEX}.log 2>&1 &
global_pid=$!

PHYSICAL_GPU_INDEX="${LAYERWISE_GPU_INDEX}" \
CONDITIONS_CSV=heldout_layerwise_shift_1px,heldout_layerwise_shift_2px \
nohup bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/46_run_p05b_vaccinated_adaptation.sh \
  > FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/logs/p05b_layerwise_gpu${LAYERWISE_GPU_INDEX}.log 2>&1 &
layerwise_pid=$!

echo "P05-B launched: global GPU${GLOBAL_GPU_INDEX} pid=${global_pid}; layerwise GPU${LAYERWISE_GPU_INDEX} pid=${layerwise_pid}"
