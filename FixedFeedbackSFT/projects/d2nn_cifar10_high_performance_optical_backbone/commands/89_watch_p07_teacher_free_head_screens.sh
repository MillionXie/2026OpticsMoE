#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
restarts=0
names=(linear mlp conv_mlp)

while true; do
  complete=0
  missing=0
  for name in "${names[@]}"; do
    run_dir="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p07_teacher_free_screen_${name}"
    pattern="general_backbone_pretraining.*p07_teacher_free_screen_${name}.yaml"
    if [[ -f "${run_dir}/result.json" ]]; then
      complete="$((complete + 1))"
      latest="$(grep -E '^\[(baseline|train|epoch)\]' "${run_dir}/train.log" | tail -n 1 || true)"
      echo "[$(date -Iseconds)] ${name}=complete ${latest}"
    elif pgrep -af "${pattern}" >/dev/null; then
      latest="$(grep -E '^\[(baseline|amp|train|epoch)\]' "${run_dir}/train.log" | tail -n 1 || true)"
      echo "[$(date -Iseconds)] ${name}=running ${latest}"
    else
      missing=1
      echo "[$(date -Iseconds)] ${name}=missing"
    fi
  done
  if (( complete == ${#names[@]} )); then
    echo "All P07 teacher-free head screens are complete"
    exit 0
  fi
  if (( missing )); then
    if (( restarts >= MAX_RESTARTS )); then
      echo "P07 restart budget exhausted (${MAX_RESTARTS})" >&2
      exit 1
    fi
    restarts="$((restarts + 1))"
    bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/88_launch_p07_teacher_free_head_screens.sh
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
