#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-1}"
select_gpu

BATCH_SIZES="${BATCH_SIZES:-64 96 128 160 192}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${EXPERIMENT}/configs/batch_benchmark.yaml"
RESULT_DIR="${EXPERIMENT}/runs/batch_benchmarks/${RUN_TAG}"
mkdir -p "${RESULT_DIR}"
SUMMARY="${RESULT_DIR}/summary.tsv"
printf 'batch\tstatus\tsamples_per_second\tpeak_allocated_mib\tpeak_reserved_mib\tseconds\n' > "${SUMMARY}"

for batch_size in ${BATCH_SIZES}; do
  output_dir="${RESULT_DIR}/bs${batch_size}"
  config_path="/tmp/p08_batch_benchmark_${RUN_TAG}_bs${batch_size}.yaml"
  sed \
    -e "s|^output_dir:.*|output_dir: ${output_dir}|" \
    -e "s|^  batch_size:.*|  batch_size: ${batch_size}|" \
    "${BASE_CONFIG}" > "${config_path}"

  echo "[benchmark] batch=${batch_size} gpu=${PHYSICAL_GPU_INDEX} output=${output_dir}"
  if "${PYTHON_BIN}" -m "${EXPERIMENT//\//.}.train" --config "${config_path}" \
      2>&1 | tee "${RESULT_DIR}/bs${batch_size}.log"; then
    "${PYTHON_BIN}" - "${output_dir}/metrics/latest.json" "${batch_size}" >> "${SUMMARY}" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))["train"]
print(
    f"{sys.argv[2]}\tok\t{metrics['samples_per_second']:.3f}\t"
    f"{metrics['peak_allocated_mib']:.1f}\t{metrics['peak_reserved_mib']:.1f}\t"
    f"{metrics['seconds']:.3f}"
)
PY
  else
    printf '%s\tfailed\t-\t-\t-\t-\n' "${batch_size}" >> "${SUMMARY}"
  fi
done

cat "${SUMMARY}"
