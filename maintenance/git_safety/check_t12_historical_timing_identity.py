"""Check the preserved seven-model T12 timing identities without running models.

Weights remain outside Git. This check establishes report/checkpoint identity,
not reproduced performance or a complete historical inference environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


MANIFEST = "maintenance/storage/T12_HISTORICAL_SEVEN_MODEL_IDENTITY_20261004.json"


def check(root: Path, commit: str) -> dict:
    def read(path: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{path}"])

    manifest = json.loads(read(MANIFEST))
    report = manifest["timing_report"]
    # The raw timing asset is deliberately not added to Git.
    asset = root / report["path"]
    raw = asset.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["sha256"], "Timing asset changed"
    timing = json.loads(raw)
    keys = [("baseline", None), ("large", "background"), ("large", "redesign"),
            ("large", "premium"), ("small", "background"), ("small", "redesign"),
            ("small", "premium")]
    assert len(manifest["checkpoints"]) == len(keys) == 7
    assert len({row["path"] for row in manifest["checkpoints"]}) == 7
    for row, (group, task) in zip(manifest["checkpoints"], keys):
        measured = timing[group] if task is None else timing[group][task]
        assert row["timing_name"] == measured["name"]
        assert row["counted_parameters_in_original_report"] == measured["counted_parameters"]
        assert row["measured_bypass_mean_ms"] == measured["measured_optical_bypass"]["mean_ms"]
        assert row["estimated_hardware_mean_ms"] == measured["estimated_hardware_mean_ms"]
        assert row["bytes"] > 0 and len(row["sha256"]) == 64
        assert row["server_sha_verified"] and row["cpu_weights_only_structure_read"]
    assert manifest["historical_full_model_reload_performed"] is False
    assert manifest["timing_rerun_performed"] is False
    assert manifest["frozen_dependencies"]["historical_benchmark_argv_verified"] is False
    assert timing["optical_accounting"].startswith("software FFT bypassed")
    return {"commit": commit, "seven_checkpoint_identities_checked": 7,
            "original_timing_asset_sha_verified": True, "gpu_used": False,
            "timing_rerun": False, "full_model_reproduction_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="main")
    args = parser.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[2], args.commit), indent=2))
