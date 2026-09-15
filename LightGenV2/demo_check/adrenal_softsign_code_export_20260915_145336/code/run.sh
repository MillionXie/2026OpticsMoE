#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/root/autodl-tmp/Figure2b_adrenal_L4_5090/.venv/bin/python
"$PY" -u search_pipeline.py preflight
exec "$PY" -u search_pipeline.py run
