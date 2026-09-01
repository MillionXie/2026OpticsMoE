#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

for run_name in \
  p11_large_recipe_proxy_5e_phase2e3_2gpu_gb384 \
  p11_large_recipe_proxy_5e_phase7e3_2gpu_gb384 \
  p11_large_recipe_formal_100e_phase2e3_5gpu_gb480 \
  p11_large_recipe_formal_100e_phase7e3_5gpu_gb480; do
  run_dir="${RUNS_DIR}/${run_name}"
  echo "===== ${run_name} ====="
  if [[ -f "${run_dir}/launch.pid" ]]; then
    pid="$(cat "${run_dir}/launch.pid")"
    if kill -0 "${pid}" 2>/dev/null; then echo "launcher_pid=${pid} alive"; else echo "launcher_pid=${pid} stopped"; fi
  fi
  if [[ -f "${run_dir}/metrics/latest.json" ]]; then
    "${PYTHON_BIN}" -m json.tool "${run_dir}/metrics/latest.json"
  elif [[ -f "${run_dir}/metrics/initial_baseline.json" ]]; then
    "${PYTHON_BIN}" -m json.tool "${run_dir}/metrics/initial_baseline.json"
  else
    echo "no metrics yet"
  fi
done
