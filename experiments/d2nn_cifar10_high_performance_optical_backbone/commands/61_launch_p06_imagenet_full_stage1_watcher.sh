#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p06_imagenet_full_stage1"
mkdir -p "${RUN_DIR}"

if pgrep -af "[6]0_watch_p06_imagenet_full_stage1.sh" >/dev/null; then
  echo "P06 full ImageNet watcher is already running; refusing a duplicate." >&2
  exit 2
fi

nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/60_watch_p06_imagenet_full_stage1.sh \
  > "${RUN_DIR}/watcher.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/watcher.pid"
echo "Launched P06 full ImageNet watcher pid=${pid} log=${RUN_DIR}/watcher.log"
