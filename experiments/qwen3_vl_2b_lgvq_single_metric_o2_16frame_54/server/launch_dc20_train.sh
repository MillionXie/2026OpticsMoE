#!/usr/bin/env bash
set -Eeuo pipefail

# Launch only the three final DC20 optical-on runs. Qwen caches must already
# pass preflight. Run this from the repository root in the intended conda env.

MODULE="experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"
PROJECT="experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPATIAL_GPU="${SPATIAL_GPU:-5}"
TEMPORAL_GPU="${TEMPORAL_GPU:-4}"
TEMPORAL_ACCURACY_GPU="${TEMPORAL_ACCURACY_GPU:-2}"

[[ -d "${PROJECT}" ]] || { echo "Run from repository root" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID

check_idle() {
  local gpu="$1" pids
  nvidia-smi -i "${gpu}" >/dev/null 2>&1 || { echo "GPU ${gpu} does not exist" >&2; exit 2; }
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "${pids}" ]] || { echo "GPU ${gpu} is not idle: ${pids}" >&2; exit 2; }
}

CONFIGS=(
  "${PROJECT}/configs/release/spatial.yaml"
  "${PROJECT}/configs/release/temporal.yaml"
  "${PROJECT}/configs/release/temporal_accuracy.yaml"
)
for config in "${CONFIGS[@]}"; do
  "${PYTHON_BIN}" -m "${MODULE}" --config "${config}" --phase preflight |
    grep -q '"status": "ready"' || { echo "Preflight failed: ${config}" >&2; exit 2; }
done

for gpu in "${SPATIAL_GPU}" "${TEMPORAL_GPU}" "${TEMPORAL_ACCURACY_GPU}"; do check_idle "${gpu}"; done
[[ "$(printf '%s\n' "${SPATIAL_GPU}" "${TEMPORAL_GPU}" "${TEMPORAL_ACCURACY_GPU}" | sort -u | wc -l)" -eq 3 ]] || {
  echo "The three training GPUs must be different" >&2; exit 2;
}

RUN_DIR="${PROJECT}/runs/server_jobs/dc20_$(date -u +'%Y%m%dT%H%M%SZ')"
mkdir -p "${RUN_DIR}"
printf '%s\n' "$(realpath "${RUN_DIR}")" > "${PROJECT}/runs/server_jobs/latest_run.txt"

launch() {
  local name="$1" gpu="$2" config="$3"
  nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -u -m "${MODULE}" --config "${config}" --phase train \
    > "${RUN_DIR}/${name}.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RUN_DIR}/${name}.pid"
  printf '%s GPU=%s PID=%s\n' "${name}" "${gpu}" "$!"
}

launch spatial "${SPATIAL_GPU}" "${CONFIGS[0]}"
launch temporal "${TEMPORAL_GPU}" "${CONFIGS[1]}"
launch temporal_accuracy "${TEMPORAL_ACCURACY_GPU}" "${CONFIGS[2]}"
sleep 4
for file in "${RUN_DIR}"/*.pid; do
  pid="$(cat "${file}")"
  kill -0 "${pid}" 2>/dev/null || { echo "Startup failed: ${file%.pid}.log" >&2; exit 2; }
done
printf 'Run directory: %s\n' "${RUN_DIR}"
printf 'Monitor: bash %s/server/monitor_dc20_runs.sh --run-dir %s --interval 30\n' "${PROJECT}" "${RUN_DIR}"
