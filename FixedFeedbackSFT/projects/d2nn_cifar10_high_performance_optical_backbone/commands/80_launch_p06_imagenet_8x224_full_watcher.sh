#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_8x224_full_spatial"
mkdir -p "${RUN_DIR}"
if pgrep -af "[7]9_watch_p06_imagenet_8x224_full_spatial.sh" >/dev/null; then
  echo "8x224 full watcher is already running; refusing a duplicate." >&2
  exit 2
fi
nohup bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/79_watch_p06_imagenet_8x224_full_spatial.sh \
  > "${RUN_DIR}/watcher.log" 2>&1 &
pid=$!
echo "${pid}" > "${RUN_DIR}/watcher.pid"
echo "Launched 8x224 full watcher pid=${pid}"
