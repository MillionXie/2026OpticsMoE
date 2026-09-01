#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

run_dir="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p07_teacher_free_head_screens"
mkdir -p "${run_dir}"
pattern="89_watch_p07_teacher_free_head_screens.sh"
if pgrep -af "${pattern}" >/dev/null; then
  echo "P07 watcher already running; refusing duplicate" >&2
  exit 2
fi
nohup bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/89_watch_p07_teacher_free_head_screens.sh \
  > "${run_dir}/watcher.log" 2>&1 &
echo "$!" > "${run_dir}/watcher.pid"
echo "Launched P07 watcher pid=$!"
