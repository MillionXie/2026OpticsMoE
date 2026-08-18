#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODULE="experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400"
EXPERIMENT_DIR="experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400"
CONFIG="${EXPERIMENT_DIR}/configs/main.yaml"
RESULT_DIR="${EXPERIMENT_DIR}/runs/main/comparison"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x /home/guest3/miniconda3/envs/xml/bin/python ]]; then
    PYTHON_BIN="/home/guest3/miniconda3/envs/xml/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "No Python executable found. Activate the project environment or set PYTHON_BIN." >&2
    exit 1
fi

"${PYTHON_BIN}" -m pytest "${EXPERIMENT_DIR}/tests" -q
"${PYTHON_BIN}" -m "${MODULE}" --config "${CONFIG}" --phase compare

for result_file in aggregate.csv endpoint_geometry.csv task_metrics.csv comparison.json; do
    test -s "${RESULT_DIR}/${result_file}"
done

sha256sum \
    "${RESULT_DIR}/aggregate.csv" \
    "${RESULT_DIR}/endpoint_geometry.csv" \
    "${RESULT_DIR}/task_metrics.csv" \
    "${RESULT_DIR}/comparison.json"
