#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ACTION="${IN22K_INDEX_ACTION:-audit}"
INDEX_OUTPUT="${IN22K_INDEX_OUTPUT:?Set IN22K_INDEX_OUTPUT to a repository-independent data index directory}"

case "${ACTION}" in
  build)
    SOURCE_ROOT="${IN22K_SOURCE_ROOT:?Set IN22K_SOURCE_ROOT to the authorized class-folder split}"
    SOURCE_DECLARATION="${IN22K_SOURCE_DECLARATION:?Set IN22K_SOURCE_DECLARATION to the reviewed JSON declaration}"
    "${PYTHON_BIN}" -m "${DATASET_MODULE}" build \
      --source-root "${SOURCE_ROOT}" \
      --declaration "${SOURCE_DECLARATION}" \
      --output "${INDEX_OUTPUT}"
    ;;
  audit)
    "${PYTHON_BIN}" -m "${DATASET_MODULE}" audit --index "${INDEX_OUTPUT}"
    ;;
  fast_audit)
    "${PYTHON_BIN}" -m "${DATASET_MODULE}" audit --index "${INDEX_OUTPUT}" --fast
    ;;
  *)
    echo "IN22K_INDEX_ACTION must be build, audit or fast_audit" >&2
    exit 2
    ;;
esac
