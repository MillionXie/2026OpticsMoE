#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${P14_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
PYTHON_BIN="${P14_PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa"
CONFIG="${P14_CONFIG:-$REPO_ROOT/FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa/configs/vtab1k_six_task_50e.yaml}"
OUTPUT_ROOT="${P14_OUTPUT_ROOT:-/DATA/DATA1/guest3/2026OpticsMoE/FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa/p14_vtab1k_six_task_50e}"
SEED="${P14_SEED:-2026}"
GPU_LIST="${P14_GPU_LIST:-1,3,4}"
LOG_ROOT="$OUTPUT_ROOT/_launcher"
SCRIPT_PATH="$(readlink -f "$0")"
TASKS=(cifar100 flowers102 eurosat patch_camelyon dsprites_orientation smallnorb_azimuth)
METHODS=(noft bp fa_pretrained fa_random)

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export P14_OUTPUT_ROOT="$OUTPUT_ROOT"

gpu_is_idle() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$pids" ]]
}

gpu_uuid() {
  local gpu="$1"
  local uuid
  uuid="$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n 1 | xargs)"
  [[ "$uuid" == GPU-* ]] || {
    echo "Could not resolve physical GPU $gpu to a UUID" >&2
    return 1
  }
  echo "$uuid"
}

run_one() {
  local task="$1"
  local method="$2"
  "$PYTHON_BIN" -m "$MODULE" \
    --config "$CONFIG" --task "$task" --method "$method" --seed "$SEED"
}

run_lane() {
  local lane="$1"
  local lanes="$2"
  local task_index method task
  for task_index in "${!TASKS[@]}"; do
    if (( task_index % lanes != lane )); then
      continue
    fi
    task="${TASKS[$task_index]}"
    for method in "${METHODS[@]}"; do
      run_one "$task" "$method"
    done
  done
  touch "$LOG_ROOT/lane_${lane}.done"
}

wait_lane() {
  local lane="$1"
  local lanes="$2"
  local gpu="$3"
  local state_file="$LOG_ROOT/lane_${lane}.state"
  local uuid
  echo "waiting_for_physical_gpu=$gpu" >"$state_file"
  while ! gpu_is_idle "$gpu"; do
    sleep 20
  done
  uuid="$(gpu_uuid "$gpu")"
  echo "running_on_physical_gpu=$gpu uuid=$uuid" >"$state_file"
  if CUDA_VISIBLE_DEVICES="$uuid" P14_PHYSICAL_GPU="$gpu" P14_GPU_UUID="$uuid" \
      bash "$SCRIPT_PATH" lane "$lane" "$lanes"; then
    echo "finished_physical_gpu=$gpu" >"$state_file"
  else
    local code="$?"
    echo "failed_physical_gpu=$gpu exit_code=$code" >"$state_file"
    touch "$LOG_ROOT/lane_${lane}.failed"
    return "$code"
  fi
}

launch() {
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  if [[ "${#gpus[@]}" -ne 3 ]]; then
    echo "P14 formal launch requires exactly three GPU ids; got $GPU_LIST" >&2
    exit 2
  fi
  mkdir -p "$LOG_ROOT"
  local lane gpu existing uuid
  for lane in "${!gpus[@]}"; do
    gpu="${gpus[$lane]}"
    if ! gpu_is_idle "$gpu"; then
      echo "GPU $gpu has a compute process; refusing to overlap." >&2
      exit 3
    fi
    if [[ -f "$LOG_ROOT/lane_${lane}.pid" ]]; then
      existing="$(cat "$LOG_ROOT/lane_${lane}.pid")"
      if kill -0 "$existing" 2>/dev/null; then
        echo "Lane $lane is already alive as PID $existing" >&2
        exit 4
      fi
    fi
  done
  rm -f "$LOG_ROOT"/lane_*.done
  for lane in "${!gpus[@]}"; do
    gpu="${gpus[$lane]}"
    uuid="$(gpu_uuid "$gpu")"
    nohup env CUDA_VISIBLE_DEVICES="$uuid" \
      P14_PHYSICAL_GPU="$gpu" P14_GPU_UUID="$uuid" \
      P14_REPO_ROOT="$REPO_ROOT" P14_PYTHON_BIN="$PYTHON_BIN" \
      P14_CONFIG="$CONFIG" P14_OUTPUT_ROOT="$OUTPUT_ROOT" P14_SEED="$SEED" \
      bash "$SCRIPT_PATH" lane "$lane" "${#gpus[@]}" \
      >"$LOG_ROOT/lane_${lane}.log" 2>&1 &
    echo "$!" >"$LOG_ROOT/lane_${lane}.pid"
    echo "launched lane=$lane physical_gpu=$gpu pid=$!"
  done
  nohup env P14_REPO_ROOT="$REPO_ROOT" P14_PYTHON_BIN="$PYTHON_BIN" \
    P14_CONFIG="$CONFIG" P14_OUTPUT_ROOT="$OUTPUT_ROOT" P14_SEED="$SEED" \
    bash "$SCRIPT_PATH" monitor >"$LOG_ROOT/monitor.log" 2>&1 &
  echo "$!" >"$LOG_ROOT/monitor.pid"
}

launch_deferred() {
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  if [[ "${#gpus[@]}" -ne 3 ]]; then
    echo "P14 deferred launch requires exactly three GPU ids; got $GPU_LIST" >&2
    exit 2
  fi
  mkdir -p "$LOG_ROOT"
  local lane gpu existing
  for lane in "${!gpus[@]}"; do
    if [[ -f "$LOG_ROOT/lane_${lane}.pid" ]]; then
      existing="$(cat "$LOG_ROOT/lane_${lane}.pid")"
      if kill -0 "$existing" 2>/dev/null; then
        echo "Lane $lane is already alive as PID $existing" >&2
        exit 4
      fi
    fi
  done
  rm -f "$LOG_ROOT"/lane_*.done "$LOG_ROOT"/lane_*.failed \
    "$LOG_ROOT"/lane_*.state "$LOG_ROOT"/all_lanes_finished \
    "$LOG_ROOT"/one_or_more_lanes_failed
  for lane in "${!gpus[@]}"; do
    gpu="${gpus[$lane]}"
    nohup env P14_REPO_ROOT="$REPO_ROOT" P14_PYTHON_BIN="$PYTHON_BIN" \
      P14_CONFIG="$CONFIG" P14_OUTPUT_ROOT="$OUTPUT_ROOT" P14_SEED="$SEED" \
      bash "$SCRIPT_PATH" wait-lane "$lane" "${#gpus[@]}" "$gpu" \
      >"$LOG_ROOT/lane_${lane}.log" 2>&1 &
    echo "$!" >"$LOG_ROOT/lane_${lane}.pid"
    echo "supervising lane=$lane physical_gpu=$gpu pid=$!"
  done
  nohup env P14_REPO_ROOT="$REPO_ROOT" P14_PYTHON_BIN="$PYTHON_BIN" \
    P14_CONFIG="$CONFIG" P14_OUTPUT_ROOT="$OUTPUT_ROOT" P14_SEED="$SEED" \
    bash "$SCRIPT_PATH" monitor >"$LOG_ROOT/monitor.log" 2>&1 &
  echo "$!" >"$LOG_ROOT/monitor.pid"
}

monitor() {
  local any pid_file pid
  while true; do
    any=0
    for pid_file in "$LOG_ROOT"/lane_*.pid; do
      [[ -f "$pid_file" ]] || continue
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" 2>/dev/null; then
        any=1
      fi
    done
    (( any == 0 )) && break
    sleep 20
  done
  "$PYTHON_BIN" -m "$MODULE.summarize" --root "$OUTPUT_ROOT"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    >"$LOG_ROOT/gpu_after_completion.csv"
  if compgen -G "$LOG_ROOT/lane_*.failed" >/dev/null; then
    touch "$LOG_ROOT/one_or_more_lanes_failed"
  else
    touch "$LOG_ROOT/all_lanes_finished"
  fi
}

status() {
  mkdir -p "$LOG_ROOT"
  local pid_file pid state
  for pid_file in "$LOG_ROOT"/lane_*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid="$(cat "$pid_file")"
    state="stopped"
    kill -0 "$pid" 2>/dev/null && state="running"
    detail=""
    [[ -f "${pid_file%.pid}.state" ]] && detail=" $(cat "${pid_file%.pid}.state")"
    echo "$(basename "$pid_file" .pid): pid=$pid state=$state$detail"
  done
  find "$OUTPUT_ROOT" -mindepth 4 -maxdepth 4 -name result.json -type f \
    | wc -l | awk '{print "result_files=" $1}'
  "$PYTHON_BIN" -m "$MODULE.summarize" --root "$OUTPUT_ROOT"
}

smoke() {
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  local gpu="${gpus[0]}"
  gpu_is_idle "$gpu" || { echo "GPU $gpu is not idle" >&2; exit 3; }
  local uuid
  uuid="$(gpu_uuid "$gpu")"
  local smoke_root
  smoke_root="$(mktemp -d /tmp/p14_vtab_smoke.XXXXXX)"
  trap "rm -rf -- '$smoke_root'" EXIT
  CUDA_VISIBLE_DEVICES="$uuid" P14_PHYSICAL_GPU="$gpu" P14_GPU_UUID="$uuid" \
    "$PYTHON_BIN" -m "$MODULE" \
    --config "$CONFIG" --task cifar100 --method noft --seed 2026 \
    --output-root "$smoke_root" --smoke
  CUDA_VISIBLE_DEVICES="$uuid" P14_PHYSICAL_GPU="$gpu" P14_GPU_UUID="$uuid" \
    "$PYTHON_BIN" -m "$MODULE" \
    --config "$CONFIG" --task cifar100 --method bp --seed 2026 \
    --output-root "$smoke_root" --smoke
  echo "P14 smoke passed on physical GPU $gpu"
}

case "${1:-status}" in
  launch) launch ;;
  launch-deferred) launch_deferred ;;
  lane) run_lane "$2" "$3" ;;
  wait-lane) wait_lane "$2" "$3" "$4" ;;
  monitor) monitor ;;
  status) status ;;
  smoke) smoke ;;
  *) echo "usage: $0 {launch|launch-deferred|status|smoke}" >&2; exit 2 ;;
esac
