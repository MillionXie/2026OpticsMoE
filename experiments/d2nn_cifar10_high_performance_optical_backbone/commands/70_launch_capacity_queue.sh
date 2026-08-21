#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p06_imagenet_capacity_12x192"
mkdir -p "${RUN_DIR}"
if pgrep -af "[6]9_wait_f2a_then_run_capacity.sh" >/dev/null; then
  echo "P06-F2C queue is already running; refusing a duplicate." >&2
  exit 2
fi
nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/69_wait_f2a_then_run_capacity.sh \
  > "${RUN_DIR}/queue.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/queue.pid"
echo "Launched P06-F2C queue pid=${pid}"
