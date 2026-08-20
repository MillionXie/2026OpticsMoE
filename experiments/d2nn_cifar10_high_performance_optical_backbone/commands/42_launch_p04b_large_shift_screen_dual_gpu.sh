#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p experiments/d2nn_cifar10_high_performance_optical_backbone/logs

PHYSICAL_GPU_INDEX=4 \
CONDITIONS_CSV=global_shift_0p5px,global_shift_1px,global_shift_2px \
nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/40_run_p04b_large_shift_screen.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/logs/p04b_global_gpu4.log 2>&1 &
global_pid=$!

PHYSICAL_GPU_INDEX=5 \
CONDITIONS_CSV=layerwise_shift_0p5px,layerwise_shift_1px,layerwise_shift_2px \
nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/40_run_p04b_large_shift_screen.sh \
  > experiments/d2nn_cifar10_high_performance_optical_backbone/logs/p04b_layerwise_gpu5.log 2>&1 &
layerwise_pid=$!

echo "P04-S2 launched: global GPU4 pid=${global_pid}; layerwise GPU5 pid=${layerwise_pid}"
