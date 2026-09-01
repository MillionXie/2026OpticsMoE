#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/logs

PHYSICAL_GPU_INDEX=4 \
CONDITIONS_CSV=global_shift_0p125px,global_shift_0p25px \
nohup bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/37_run_p04_adaptation_screen.sh \
  > FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/logs/p04_global_gpu4.log 2>&1 &
global_pid=$!

PHYSICAL_GPU_INDEX=5 \
CONDITIONS_CSV=layerwise_shift_0p125px,layerwise_shift_0p25px \
nohup bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/37_run_p04_adaptation_screen.sh \
  > FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/logs/p04_layerwise_gpu5.log 2>&1 &
layerwise_pid=$!

echo "P04 screen launched: global GPU4 pid=${global_pid}; layerwise GPU5 pid=${layerwise_pid}"
