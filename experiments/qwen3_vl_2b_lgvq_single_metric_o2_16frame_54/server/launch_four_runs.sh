#!/usr/bin/env bash
set -Eeuo pipefail

# Cache the frozen Qwen front once, validate all four formal contracts, then
# launch four independent training processes.  This script deliberately does
# not support overwriting/resuming an existing output directory.

PROJECT_MODULE="experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"
EXPECTED_DATASET="/DATA/DATA1/lixinyue/xyli/data/LGVQ"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

REPO_ROOT="${DEFAULT_REPO_ROOT}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_GPU="${CACHE_GPU:-0}"
SPATIAL_GPU="${SPATIAL_GPU:-0}"
SPATIAL_SEED43_GPU="${SPATIAL_SEED43_GPU:-1}"
ROBUST_GPU="${ROBUST_GPU:-3}"
TEMPORAL_GPU="${TEMPORAL_GPU:-4}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-2}"
CACHE_CHUNK_ROWS="${CACHE_CHUNK_ROWS:-16}"

usage() {
  cat <<'EOF'
Usage:
  bash launch_four_runs.sh --qwen-model /ABS/PATH/Qwen3-VL-2B-Instruct [options]

Required (an environment variable is also accepted):
  --qwen-model PATH           Local Qwen3-VL-2B-Instruct directory
                              or set QWEN_MODEL_PATH.

Options:
  --repo-root PATH            Repository root (normally auto-detected)
  --python PATH               Python executable (default: python)
  --cache-gpu ID              Qwen-front cache GPU (default: 0)
  --spatial-gpu ID            Spatial formal GPU (default: 0)
  --spatial-seed43-gpu ID     Spatial seed-43 GPU (default: 1)
  --robust-gpu ID             Spatial strong-robust GPU (default: 3)
  --temporal-gpu ID           Temporal GPU (default: 4)
  --cache-batch-size N        Videos per cache batch (default: 2)
  --cache-chunk-rows N        Resumable cache shard size (default: 16)
  -h, --help                  Show this help

The cache GPU may equal one training GPU because caching completes before the
four jobs start.  The four training GPU IDs must be distinct and idle.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --qwen-model) QWEN_MODEL_PATH="${2:?missing value for --qwen-model}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?missing value for --repo-root}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --cache-gpu) CACHE_GPU="${2:?missing value for --cache-gpu}"; shift 2 ;;
    --spatial-gpu) SPATIAL_GPU="${2:?missing value for --spatial-gpu}"; shift 2 ;;
    --spatial-seed43-gpu) SPATIAL_SEED43_GPU="${2:?missing value for --spatial-seed43-gpu}"; shift 2 ;;
    --robust-gpu) ROBUST_GPU="${2:?missing value for --robust-gpu}"; shift 2 ;;
    --temporal-gpu) TEMPORAL_GPU="${2:?missing value for --temporal-gpu}"; shift 2 ;;
    --cache-batch-size) CACHE_BATCH_SIZE="${2:?missing value for --cache-batch-size}"; shift 2 ;;
    --cache-chunk-rows) CACHE_CHUNK_ROWS="${2:?missing value for --cache-chunk-rows}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${QWEN_MODEL_PATH}" ]] || die "--qwen-model (or QWEN_MODEL_PATH) is required; the script will not guess or download it"
[[ "${QWEN_MODEL_PATH}" = /* ]] || die "Qwen model path must be absolute: ${QWEN_MODEL_PATH}"
[[ -d "${QWEN_MODEL_PATH}" ]] || die "Qwen model directory does not exist: ${QWEN_MODEL_PATH}"
[[ "${REPO_ROOT}" = /* ]] || die "repository root must be absolute: ${REPO_ROOT}"
[[ -d "${REPO_ROOT}/experiments" ]] || die "not a repository root: ${REPO_ROOT}"
[[ -d "${EXPECTED_DATASET}" ]] || die "formal LGVQ dataset is missing: ${EXPECTED_DATASET}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"

for value in "${CACHE_GPU}" "${SPATIAL_GPU}" "${SPATIAL_SEED43_GPU}" "${ROBUST_GPU}" "${TEMPORAL_GPU}" \
             "${CACHE_BATCH_SIZE}" "${CACHE_CHUNK_ROWS}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || die "GPU IDs and cache sizes must be non-negative integers; got ${value}"
done
(( CACHE_BATCH_SIZE > 0 )) || die "cache batch size must be positive"
(( CACHE_CHUNK_ROWS > 0 )) || die "cache chunk rows must be positive"
TRAIN_GPUS=("${SPATIAL_GPU}" "${SPATIAL_SEED43_GPU}" "${ROBUST_GPU}" "${TEMPORAL_GPU}")
[[ "$(printf '%s\n' "${TRAIN_GPUS[@]}" | sort -u | wc -l)" -eq 4 ]] || \
  die "the four training GPU IDs must be distinct"

cd -- "${REPO_ROOT}"
PROJECT_DIR="${REPO_ROOT}/experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"
CONFIG_DIR="${PROJECT_DIR}/configs/release"
SPATIAL_CONFIG="${CONFIG_DIR}/spatial.yaml"
SPATIAL_SEED43_CONFIG="${CONFIG_DIR}/spatial_seed43.yaml"
ROBUST_CONFIG="${CONFIG_DIR}/spatial_strong_robust.yaml"
TEMPORAL_CONFIG="${CONFIG_DIR}/temporal.yaml"
for config in "${SPATIAL_CONFIG}" "${SPATIAL_SEED43_CONFIG}" "${ROBUST_CONFIG}" "${TEMPORAL_CONFIG}"; do
  [[ -f "${config}" ]] || die "missing release config: ${config}"
done

JOB_ROOT="${PROJECT_DIR}/runs/server_jobs"
mkdir -p -- "${JOB_ROOT}"

# Refuse a second launcher while any prior PID tracked by this experiment is
# still running this module. A recycled PID owned by an unrelated process does
# not create a false lock.
while IFS= read -r pid_file; do
  pid="$(tr -cd '0-9' < "${pid_file}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    process_args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${process_args}" == *"${PROJECT_MODULE}"* ]]; then
      die "tracked job is already running (PID ${pid}, ${pid_file}); use monitor_four_runs.sh"
    fi
  fi
done < <(find "${JOB_ROOT}" -type f -name '*.pid' -print 2>/dev/null)

# Resolve every data/output path through the same settings loader used by the
# trainer.  This catches a wrong checkout, missing soft targets, or an
# accidental dataset change before the expensive cache begins.
mapfile -t OUTPUT_DIRS < <("${PYTHON_BIN}" - "${SPATIAL_CONFIG}" "${SPATIAL_SEED43_CONFIG}" "${ROBUST_CONFIG}" "${TEMPORAL_CONFIG}" "${EXPECTED_DATASET}" <<'PY'
from pathlib import Path
import sys

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.settings import load_settings

configs = [Path(value) for value in sys.argv[1:5]]
expected_dataset = Path(sys.argv[5]).resolve()
for config in configs:
    settings = load_settings(config)
    if settings.dataset_root != expected_dataset:
        raise SystemExit(
            f"dataset mismatch in {config}: {settings.dataset_root} != {expected_dataset}"
        )
    if settings.training_soft_targets_path is None:
        raise SystemExit(f"training soft targets are not configured in {config}")
    if not settings.training_soft_targets_path.is_file():
        raise SystemExit(
            f"training soft targets are missing: {settings.training_soft_targets_path}"
        )
    if settings.output_dir is None:
        raise SystemExit(f"output_dir is missing in {config}")
    print(settings.output_dir)
PY
)
[[ "${#OUTPUT_DIRS[@]}" -eq 4 ]] || die "could not resolve all four output directories"

# Training has no resume contract.  Refuse directories containing any real run
# artifact instead of silently replacing a previous result.
RUN_MARKERS=(
  last_checkpoint.pt
  best_observed_test_checkpoint.pt
  training_summary.json
  train_history.json
  initialization_report.json
  parameter_breakdown.json
)
for output_dir in "${OUTPUT_DIRS[@]}"; do
  for marker in "${RUN_MARKERS[@]}"; do
    [[ ! -e "${output_dir}/${marker}" ]] || \
      die "existing training artifact would be overwritten: ${output_dir}/${marker}; move the entire output directory first"
  done
done

check_gpu_exists_and_idle() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" >/dev/null 2>&1 || die "GPU ${gpu} does not exist"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "${pids}" ]] || die "GPU ${gpu} already has compute PID(s): ${pids//$'\n'/, }"
}

# Training GPUs must be idle now.  The cache GPU only needs to be idle until the
# foreground cache completes; it may intentionally be reused by Spatial.
check_gpu_exists_and_idle "${CACHE_GPU}"
check_gpu_exists_and_idle "${SPATIAL_GPU}"
check_gpu_exists_and_idle "${SPATIAL_SEED43_GPU}"
check_gpu_exists_and_idle "${ROBUST_GPU}"
check_gpu_exists_and_idle "${TEMPORAL_GPU}"

RUN_ID="$(date -u +'%Y%m%dT%H%M%SZ')"
RUN_DIR="${JOB_ROOT}/${RUN_ID}"
mkdir -p -- "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > "${JOB_ROOT}/latest_run.txt"

cat > "${RUN_DIR}/launch_contract.txt" <<EOF
run_id=${RUN_ID}
repo_root=${REPO_ROOT}
qwen_model_path=${QWEN_MODEL_PATH}
expected_dataset=${EXPECTED_DATASET}
cache_gpu=${CACHE_GPU}
spatial_gpu=${SPATIAL_GPU}
spatial_seed43_gpu=${SPATIAL_SEED43_GPU}
robust_gpu=${ROBUST_GPU}
temporal_gpu=${TEMPORAL_GPU}
cache_batch_size=${CACHE_BATCH_SIZE}
cache_chunk_rows=${CACHE_CHUNK_ROWS}
spatial_config=${SPATIAL_CONFIG}
spatial_seed43_config=${SPATIAL_SEED43_CONFIG}
robust_config=${ROBUST_CONFIG}
temporal_config=${TEMPORAL_CONFIG}
EOF

printf '\n[1/4] Caching shared 16-frame Qwen Vision front + Spatial prompt on GPU %s\n' "${CACHE_GPU}"
CUDA_VISIBLE_DEVICES="${CACHE_GPU}" "${PYTHON_BIN}" -u -m \
  "${PROJECT_MODULE}.cache_qwen_front" \
  --config "${SPATIAL_CONFIG}" \
  --model-path "${QWEN_MODEL_PATH}" \
  --batch-size "${CACHE_BATCH_SIZE}" \
  --chunk-rows "${CACHE_CHUNK_ROWS}" \
  --device cuda 2>&1 | tee "${RUN_DIR}/cache_spatial.log"

printf '\n[2/4] Reusing the Vision cache and caching the Temporal prompt on GPU %s\n' "${CACHE_GPU}"
CUDA_VISIBLE_DEVICES="${CACHE_GPU}" "${PYTHON_BIN}" -u -m \
  "${PROJECT_MODULE}.cache_qwen_front" \
  --config "${TEMPORAL_CONFIG}" \
  --model-path "${QWEN_MODEL_PATH}" \
  --batch-size "${CACHE_BATCH_SIZE}" \
  --chunk-rows "${CACHE_CHUNK_ROWS}" \
  --device cuda 2>&1 | tee "${RUN_DIR}/cache_temporal.log"

printf '\n[3/4] Running all four formal preflights sequentially\n'
for pair in \
  "spatial:${SPATIAL_CONFIG}" \
  "spatial_seed43:${SPATIAL_SEED43_CONFIG}" \
  "spatial_strong_robust:${ROBUST_CONFIG}" \
  "temporal:${TEMPORAL_CONFIG}"; do
  name="${pair%%:*}"
  config="${pair#*:}"
  CUDA_VISIBLE_DEVICES="${CACHE_GPU}" "${PYTHON_BIN}" -u -m "${PROJECT_MODULE}" \
    --config "${config}" --phase preflight 2>&1 | tee "${RUN_DIR}/preflight_${name}.log"
  grep -q '"status": "ready"' "${RUN_DIR}/preflight_${name}.log" || \
    die "${name} preflight did not report status=ready"
done

# Re-check immediately before starting the parallel jobs.  No job is launched
# unless all four GPUs are still free, preventing a half-launched sweep.
check_gpu_exists_and_idle "${SPATIAL_GPU}"
check_gpu_exists_and_idle "${SPATIAL_SEED43_GPU}"
check_gpu_exists_and_idle "${ROBUST_GPU}"
check_gpu_exists_and_idle "${TEMPORAL_GPU}"

launch_one() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local log_file="${RUN_DIR}/${name}.log"
  local pid_file="${RUN_DIR}/${name}.pid"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m "${PROJECT_MODULE}" \
    --config "${config}" --phase train > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf '%s\tGPU=%s\tPID=%s\tlog=%s\n' "${name}" "${gpu}" "${pid}" "${log_file}"
}

printf '\n[4/4] Launching four independent optical-on training runs\n'
launch_one spatial "${SPATIAL_GPU}" "${SPATIAL_CONFIG}"
launch_one spatial_seed43 "${SPATIAL_SEED43_GPU}" "${SPATIAL_SEED43_CONFIG}"
launch_one spatial_strong_robust "${ROBUST_GPU}" "${ROBUST_CONFIG}"
launch_one temporal "${TEMPORAL_GPU}" "${TEMPORAL_CONFIG}"

sleep 3
failed=0
for pid_file in "${RUN_DIR}"/*.pid; do
  pid="$(cat "${pid_file}")"
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'A job exited during startup: %s (inspect %s.log)\n' "${pid_file}" "${pid_file%.pid}" >&2
    failed=1
  fi
done
(( failed == 0 )) || die "one or more training jobs failed during startup"

printf '\nAll jobs started. Monitor with:\n  bash %q --run-dir %q\n' \
  "${SCRIPT_DIR}/monitor_four_runs.sh" "${RUN_DIR}"
printf '\nImportant: optical-off is evaluated only as a bypass of each selected optical-on checkpoint; no pure electronic model is launched.\n'
