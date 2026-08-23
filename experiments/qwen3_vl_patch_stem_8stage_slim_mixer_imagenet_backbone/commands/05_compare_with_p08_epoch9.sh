#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

p08 = Path("experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/runs/p08_imagenet1k_pretrain_bs96_90e/metrics/history.json")
p09 = Path("experiments/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/runs/p09_imagenet1k_pretrain_bs96_90e/metrics/history.json")
if not p08.is_file() or not p09.is_file():
    raise SystemExit("Both P08 and P09 history.json files are required")
rows08 = {row["epoch"]: row for row in json.load(p08.open(encoding="utf-8"))}
rows09 = {row["epoch"]: row for row in json.load(p09.open(encoding="utf-8"))}
common = sorted(set(rows08) & set(rows09))
if not common:
    raise SystemExit("P09 has not yet completed an epoch shared with P08")
print("epoch\tp08_val_top1\tp09_val_top1\tdelta_pp\tp08_train_top1\tp09_train_top1")
for epoch in common:
    a, b = rows08[epoch], rows09[epoch]
    av, bv = a["validation"]["top1_accuracy"], b["validation"]["top1_accuracy"]
    print(f"{epoch}\t{av:.5f}\t{bv:.5f}\t{100*(bv-av):+.3f}\t{a['train']['top1_accuracy']:.5f}\t{b['train']['top1_accuracy']:.5f}")
PY
