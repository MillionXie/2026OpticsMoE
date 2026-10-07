from __future__ import annotations

import json
from pathlib import Path

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa.summarize import (
    summarize,
)


def test_summarize_complete_result(tmp_path: Path) -> None:
    result_dir = tmp_path / "cifar100" / "bp" / "seed_2026"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "task": "cifar100",
                "category": "natural",
                "method": "bp",
                "seed": 2026,
                "test": {"top1": 0.5, "balanced_accuracy": 0.4},
                "phase": {"mean_absolute_rad": 0.03},
                "wall_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    payload = summarize(tmp_path)
    assert payload["complete_runs"] == 1
    assert payload["groups"][0]["top1_mean"] == 0.5
