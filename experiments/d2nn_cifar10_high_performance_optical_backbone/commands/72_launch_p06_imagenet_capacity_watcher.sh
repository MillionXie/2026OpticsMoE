#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p06_imagenet_capacity_12x192"
mkdir -p "${RUN_DIR}"

if pgrep -af "[7]1_watch_p06_imagenet_capacity_12x192.sh" >/dev/null; then
  echo "P06-F2C watcher is already running; refusing a duplicate." >&2
  exit 2
fi

nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/71_watch_p06_imagenet_capacity_12x192.sh \
  > "${RUN_DIR}/watcher.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/watcher.pid"
echo "Launched P06-F2C watcher pid=${pid} log=${RUN_DIR}/watcher.log"
