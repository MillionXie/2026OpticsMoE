#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="FixedFeedbackSFT/runs/d2nn_cifar10_high_performance_optical_backbone/p06_imagenet_capacity_12x192"
TRAIN_LOG="${RUN_DIR}/train.log"
RESULT="${RUN_DIR}/result.json"
PATTERN="general_backbone_pretraining.*p06_imagenet_capacity_12x192.yaml"
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
    latest="$(grep -E '^\[(baseline|amp|train|epoch|result)\]' "${TRAIN_LOG}" | tail -n 1 || true)"
  else
    age="${STALE_AFTER_SECONDS}"
    latest="log-not-created"
  fi

  if pgrep -f "${PATTERN}" >/dev/null; then
    echo "[$(date -Iseconds)] running log_age=${age}s ${latest}"
    if (( age >= STALE_AFTER_SECONDS )); then
      echo "[$(date -Iseconds)] log is stale; terminating the exact P06-F2C process group"
      mapfile -t stale_pids < <(pgrep -f "${PATTERN}")
      for pid in "${stale_pids[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
      done
      sleep 15
    fi
  else
    if (( restarts >= MAX_RESTARTS )); then
      echo "[$(date -Iseconds)] restart budget exhausted (${MAX_RESTARTS})" >&2
      exit 1
    fi
    restarts="$((restarts + 1))"
    echo "[$(date -Iseconds)] P06-F2C absent; restart ${restarts}/${MAX_RESTARTS}"
    PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-4,5}" \
      bash FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/commands/68_launch_p06_imagenet_capacity_12x192.sh
    sleep 30
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[$(date -Iseconds)] P06-F2C result is ready: ${RESULT}"
