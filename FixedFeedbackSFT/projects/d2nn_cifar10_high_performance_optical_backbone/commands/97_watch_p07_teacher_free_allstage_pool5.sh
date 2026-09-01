#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p07_teacher_free_allstage_pool5"
TRAIN_LOG="${RUN_DIR}/train.log"
RESULT="${RUN_DIR}/result.json"
PATTERN="general_backbone_pretraining.*p07_teacher_free_allstage_pool5.yaml"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
STALE_AFTER_SECONDS="${STALE_AFTER_SECONDS:-1800}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
restarts=0
mkdir -p "${RUN_DIR}"
while [[ ! -f "${RESULT}" ]]; do
  now="$(date +%s)"
  if [[ -f "${TRAIN_LOG}" ]]; then
    modified="$(stat -c %Y "${TRAIN_LOG}")"
    age="$((now - modified))"
    latest="$(grep -E '^\[(baseline|amp|train|epoch|resume)\]' "${TRAIN_LOG}" | tail -n 1 || true)"
  else
    age="${STALE_AFTER_SECONDS}"
    latest="log-not-created"
  fi
  if pgrep -f "${PATTERN}" >/dev/null; then
    echo "[$(date -Iseconds)] running log_age=${age}s ${latest}"
    if (( age >= STALE_AFTER_SECONDS )); then
      mapfile -t stale_pids < <(pgrep -f "${PATTERN}")
      for pid in "${stale_pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
      sleep 15
    fi
  else
    launcher_pid=""
    if [[ -f "${RUN_DIR}/launcher.pid" ]]; then
      launcher_pid="$(cat "${RUN_DIR}/launcher.pid")"
    fi
    if [[ -n "${launcher_pid}" ]] && kill -0 "${launcher_pid}" 2>/dev/null; then
      echo "[$(date -Iseconds)] launcher ${launcher_pid} is still handing off to torchrun"
      sleep 30
      continue
    fi
    if (( restarts >= MAX_RESTARTS )); then
      echo "P07-A restart budget exhausted (${MAX_RESTARTS})" >&2
      exit 1
    fi
    restarts="$((restarts + 1))"
    PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-1,3,4}" \
      bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/96_launch_p07_teacher_free_allstage_pool5.sh
    sleep 30
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
echo "P07-A result is ready: ${RESULT}"
