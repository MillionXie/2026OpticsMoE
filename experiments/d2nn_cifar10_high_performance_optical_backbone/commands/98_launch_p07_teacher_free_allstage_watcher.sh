#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="experiments/d2nn_cifar10_high_performance_optical_backbone/runs/p07_teacher_free_allstage_pool5"
mkdir -p "${RUN_DIR}"
if pgrep -af "[9]7_watch_p07_teacher_free_allstage_pool5.sh" >/dev/null; then
  echo "P07-A watcher already running" >&2
  exit 2
fi
nohup env PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-1,3,4}" \
  bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/97_watch_p07_teacher_free_allstage_pool5.sh \
  > "${RUN_DIR}/watcher.log" 2>&1 &
echo "$!" > "${RUN_DIR}/watcher.pid"
echo "Launched P07-A watcher pid=$!"
