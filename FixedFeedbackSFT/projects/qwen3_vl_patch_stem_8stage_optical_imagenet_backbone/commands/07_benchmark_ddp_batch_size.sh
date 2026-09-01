#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-3,5}"
IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
uuids=()
for index in "${indices[@]}"; do
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((index + 1))p")"
  : "${uuid:?Could not resolve GPU ${index}}"
  uuids+=("${uuid}")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

BATCH_SIZES="${BATCH_SIZES:-64 96}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${EXPERIMENT}/configs/batch_benchmark.yaml"
RESULT_DIR="${RUNS_DIR}/ddp_batch_benchmarks/${RUN_TAG}"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
mkdir -p "${RESULT_DIR}"
SUMMARY="${RESULT_DIR}/summary.tsv"
printf 'per_rank_batch\tglobal_batch\tstatus\tsamples_per_second\tpeak_allocated_mib\tpeak_reserved_mib\tseconds\n' > "${SUMMARY}"

for batch_size in ${BATCH_SIZES}; do
  global_batch=$((batch_size * ${#indices[@]}))
  output_dir="${RESULT_DIR}/bs${batch_size}_global${global_batch}"
  config_path="/tmp/p08_ddp_batch_benchmark_${RUN_TAG}_bs${batch_size}.yaml"
  sed \
    -e "s|^output_dir:.*|output_dir: ${output_dir}|" \
    -e "s|^  batch_size:.*|  batch_size: ${batch_size}|" \
    "${BASE_CONFIG}" > "${config_path}"

  echo "[ddp-benchmark] per_rank=${batch_size} global=${global_batch} gpus=${PHYSICAL_GPU_INDICES}"
  if "${TORCHRUN_BIN}" --standalone --nproc_per_node="${#indices[@]}" \
      -m "${MODULE}.train" --config "${config_path}" \
      2>&1 | tee "${RESULT_DIR}/bs${batch_size}.log"; then
    "${PYTHON_BIN}" - "${output_dir}/metrics/latest.json" "${batch_size}" "${global_batch}" >> "${SUMMARY}" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))["train"]
print(
    f"{sys.argv[2]}\t{sys.argv[3]}\tok\t{metrics['samples_per_second']:.3f}\t"
    f"{metrics['peak_allocated_mib']:.1f}\t{metrics['peak_reserved_mib']:.1f}\t"
    f"{metrics['seconds']:.3f}"
)
PY
  else
    printf '%s\t%s\tfailed\t-\t-\t-\t-\n' "${batch_size}" "${global_batch}" >> "${SUMMARY}"
  fi
done

cat "${SUMMARY}"
