#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${P12_REPO_ROOT:-$(cd "${EXPERIMENT_DIR}/../../.." && pwd)}"
RUNS_DIR="${P12_RUNS_ROOT:-${REPO_ROOT}/FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa}"
PYTHON_BIN="${P12_PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
FORMAL_REPO_ROOT="${P12_FORMAL_REPO_ROOT:-}"
if [[ -n "${FORMAL_REPO_ROOT}" ]]; then
  formal_layout_config="${FORMAL_REPO_ROOT}/FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml"
  formal_legacy_config="${FORMAL_REPO_ROOT}/experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml"
  if [[ -f "${formal_layout_config}" ]]; then
    DEFAULT_CONFIG="${formal_layout_config}"
  else
    DEFAULT_CONFIG="${formal_legacy_config}"
  fi
else
  DEFAULT_CONFIG="${EXPERIMENT_DIR}/configs/base_50e.yaml"
fi
CONFIG="${P12_CONFIG:-${DEFAULT_CONFIG}}"
GPU="${P12_MECHANISM_GPU:-0}"
SPLIT="${P12_MECHANISM_SPLIT:-test}"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.mechanism"
BASE_OUTPUT="${P12_MECHANISM_OUTPUT_ROOT:-${RUNS_DIR}/p12_downstream_fa_50e_mechanism}"

usage() {
  cat <<'EOF'
Usage:
  bash .../commands/p12_mechanism_audit.sh pilot
  bash .../commands/p12_mechanism_audit.sh selection
  bash .../commands/p12_mechanism_audit.sh full
  bash .../commands/p12_mechanism_audit.sh full-selection
  bash .../commands/p12_mechanism_audit.sh smoke

pilot (default): Caltech101 + ISIC2016, seed 2026, best endpoints.
selection:       same pilot plus both best and last endpoints.
full:            all three tasks and seeds 2026/2027/2028, best endpoints.
full-selection:  full matrix at both best and last endpoints.
smoke:           pilot identities plus one evaluation batch per state, written
                 to a separate smoke directory; it is not a scientific result.

Environment overrides:
  P12_MECHANISM_GPU=4
  P12_PYTHON_BIN=/home/guest3/miniconda3/envs/xml/bin/python
  P12_REPO_ROOT=/path/to/the/exact/formal-training-worktree
  P12_FORMAL_REPO_ROOT=/path/to/the/exact/formal-training-worktree
  P12_CONFIG=/path/to/configs/base_50e.yaml
  P12_MECHANISM_SPLIT=test
  P12_MECHANISM_OUTPUT_ROOT=/path/to/independent/mechanism/root

This command never trains. It strictly requires completed formal NoFT, BP,
FA-pretrained and FA-random artifacts with matching source/config/code identity.
When the audit code is in a derived worktree, set P12_FORMAL_REPO_ROOT to the
locked training worktree; this restores the original absolute-path digest while
leaving the formal worktree untouched.
EOF
}

require_assets() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "P12 config not found: ${CONFIG}" >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the preflight GPU snapshot." >&2
    exit 1
  fi
}

action="${1:-pilot}"
case "${action}" in
  pilot)
    tasks="caltech101,isic2016"
    seeds="2026"
    endpoints="best"
    output="${BASE_OUTPUT}/pilot_best"
    extra=()
    ;;
  selection)
    tasks="caltech101,isic2016"
    seeds="2026"
    endpoints="best,last"
    output="${BASE_OUTPUT}/pilot_best_last"
    extra=()
    ;;
  full)
    tasks="caltech101,isic2016,lsp"
    seeds="2026,2027,2028"
    endpoints="best"
    output="${BASE_OUTPUT}/full_best"
    extra=()
    ;;
  full-selection)
    tasks="caltech101,isic2016,lsp"
    seeds="2026,2027,2028"
    endpoints="best,last"
    output="${BASE_OUTPUT}/full_best_last"
    extra=()
    ;;
  smoke)
    tasks="caltech101,isic2016"
    seeds="2026"
    endpoints="best"
    output="${BASE_OUTPUT}/smoke_one_batch"
    extra=(--max-eval-batches 1)
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown action: ${action}" >&2
    usage >&2
    exit 2
    ;;
esac

require_assets
formal_repo_args=()
if [[ -n "${FORMAL_REPO_ROOT}" ]]; then
  if [[ ! -d "${FORMAL_REPO_ROOT}" ]]; then
    echo "Formal training repo root not found: ${FORMAL_REPO_ROOT}" >&2
    exit 1
  fi
  formal_repo_args=(--formal-repo-root "${FORMAL_REPO_ROOT}")
fi
echo "Mechanism audit is evaluation-only; preflight GPU snapshot:"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
echo "action=${action} tasks=${tasks} seeds=${seeds} endpoints=${endpoints} GPU=${GPU}"
cd "${REPO_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" -m "${MODULE}" \
  --config "${CONFIG}" \
  "${formal_repo_args[@]}" \
  --tasks "${tasks}" \
  --seeds "${seeds}" \
  --endpoints "${endpoints}" \
  --split "${SPLIT}" \
  --device cuda:0 \
  --output-root "${output}" \
  "${extra[@]}"
