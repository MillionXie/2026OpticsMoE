#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p07_teacher_free_formal_linear"
mkdir -p "${RUN_DIR}"
if pgrep -af "[9]3_watch_p07_teacher_free_formal_linear.sh" >/dev/null; then
  echo "P07-F watcher already running" >&2
  exit 2
fi
nohup bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/93_watch_p07_teacher_free_formal_linear.sh \
  > "${RUN_DIR}/watcher.log" 2>&1 &
echo "$!" > "${RUN_DIR}/watcher.pid"
echo "Launched P07-F watcher pid=$!"
