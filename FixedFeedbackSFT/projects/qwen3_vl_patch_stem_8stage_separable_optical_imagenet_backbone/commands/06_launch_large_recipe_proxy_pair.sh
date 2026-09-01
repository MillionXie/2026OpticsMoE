#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem
ensure_frozen_p11_asset

P11_PROXY_LOW_GPUS="${P11_PROXY_LOW_GPUS:?Set two comma-separated GPU indices}"
P11_PROXY_HIGH_GPUS="${P11_PROXY_HIGH_GPUS:?Set two different comma-separated GPU indices}"
P11_PROXY_ACTION="${P11_PROXY_ACTION:?Set fresh or resume explicitly}"
case "${P11_PROXY_ACTION}" in
  fresh) MODE="--fresh" ;;
  resume) MODE="--resume" ;;
  *) echo "P11_PROXY_ACTION must be fresh or resume." >&2; exit 1 ;;
esac

indices_to_uuids() {
  local value="$1"
  local indices=()
  local uuids=()
  IFS=',' read -r -a indices <<< "${value}"
  if [[ "${#indices[@]}" -ne 2 ]]; then
    echo "Each proxy requires exactly two physical GPUs, got ${value}." >&2
    return 1
  fi
  local index
  for index in "${indices[@]}"; do
    [[ "${index}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index ${index}." >&2; return 1; }
    uuids+=("$(gpu_uuid "${index}")")
  done
  (IFS=','; echo "${uuids[*]}")
}

IFS=',' read -r -a low_indices <<< "${P11_PROXY_LOW_GPUS}"
IFS=',' read -r -a high_indices <<< "${P11_PROXY_HIGH_GPUS}"
LOW_VISIBLE="$(indices_to_uuids "${P11_PROXY_LOW_GPUS}")"
HIGH_VISIBLE="$(indices_to_uuids "${P11_PROXY_HIGH_GPUS}")"
declare -A seen=()
for index in "${low_indices[@]}" "${high_indices[@]}"; do
  if [[ -n "${seen[${index}]+x}" ]]; then
    echo "Proxy GPU sets overlap at physical GPU ${index}." >&2
    exit 1
  fi
  seen[${index}]=1
  require_idle_gpu "${index}"
done
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
[[ -x "${TORCHRUN_BIN}" ]] || { echo "Missing torchrun: ${TORCHRUN_BIN}" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

LOW_CONFIG="${EXPERIMENT}/configs/large_recipe_proxy_5e_phase2e3_2gpu_gb384.yaml"
HIGH_CONFIG="${EXPERIMENT}/configs/large_recipe_proxy_5e_phase7e3_2gpu_gb384.yaml"
LOW_RUN="${RUNS_DIR}/p11_large_recipe_proxy_5e_phase2e3_2gpu_gb384"
HIGH_RUN="${RUNS_DIR}/p11_large_recipe_proxy_5e_phase7e3_2gpu_gb384"
LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "${LOW_RUN}" "${HIGH_RUN}" "${LOG_DIR}"

validate_run() {
  local run_dir="$1"
  if [[ "${P11_PROXY_ACTION}" == "fresh" ]] && \
    [[ -e "${run_dir}/manifest.json" || -d "${run_dir}/checkpoints" ]]; then
    echo "Fresh mode refuses existing artifacts in ${run_dir}." >&2
    return 1
  fi
  if [[ "${P11_PROXY_ACTION}" == "resume" ]] && \
    [[ ! -f "${run_dir}/checkpoints/last.pt" ]]; then
    echo "Resume requires ${run_dir}/checkpoints/last.pt." >&2
    return 1
  fi
  if [[ -f "${run_dir}/launch.pid" ]]; then
    local old_pid
    old_pid="$(cat "${run_dir}/launch.pid")"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Run already has live launcher PID ${old_pid}: ${run_dir}." >&2
      return 1
    fi
  fi
}
validate_run "${LOW_RUN}"
validate_run "${HIGH_RUN}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOW_LOG="${LOG_DIR}/p11_proxy_phase2e3.${STAMP}.${P11_PROXY_ACTION}.log"
HIGH_LOG="${LOG_DIR}/p11_proxy_phase7e3.${STAMP}.${P11_PROXY_ACTION}.log"
printf '[launch] utc=%s action=%s physical_gpus=%s config=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P11_PROXY_ACTION}" \
  "${P11_PROXY_LOW_GPUS}" "${LOW_CONFIG}" > "${LOW_LOG}"
printf '[launch] utc=%s action=%s physical_gpus=%s config=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P11_PROXY_ACTION}" \
  "${P11_PROXY_HIGH_GPUS}" "${HIGH_CONFIG}" > "${HIGH_LOG}"

CUDA_VISIBLE_DEVICES="${LOW_VISIBLE}" nohup "${TORCHRUN_BIN}" --standalone \
  --nproc_per_node=2 -m "${MODULE}.large_scale_continue" \
  --config "${LOW_CONFIG}" "${MODE}" >> "${LOW_LOG}" 2>&1 < /dev/null &
LOW_PID=$!
printf '%s\n' "${LOW_PID}" > "${LOW_RUN}/launch.pid.tmp"
mv "${LOW_RUN}/launch.pid.tmp" "${LOW_RUN}/launch.pid"

CUDA_VISIBLE_DEVICES="${HIGH_VISIBLE}" nohup "${TORCHRUN_BIN}" --standalone \
  --nproc_per_node=2 -m "${MODULE}.large_scale_continue" \
  --config "${HIGH_CONFIG}" "${MODE}" >> "${HIGH_LOG}" 2>&1 < /dev/null &
HIGH_PID=$!
printf '%s\n' "${HIGH_PID}" > "${HIGH_RUN}/launch.pid.tmp"
mv "${HIGH_RUN}/launch.pid.tmp" "${HIGH_RUN}/launch.pid"

sleep 10
kill -0 "${LOW_PID}"
kill -0 "${HIGH_PID}"
ln -sfn "$(basename "${LOW_LOG}")" "${LOG_DIR}/p11_proxy_phase2e3.latest.log"
ln -sfn "$(basename "${HIGH_LOG}")" "${LOG_DIR}/p11_proxy_phase7e3.latest.log"
echo "Started low-phase proxy PID=${LOW_PID}, GPUs=${P11_PROXY_LOW_GPUS}, GB=384."
echo "Started high-phase proxy PID=${HIGH_PID}, GPUs=${P11_PROXY_HIGH_GPUS}, GB=384."
