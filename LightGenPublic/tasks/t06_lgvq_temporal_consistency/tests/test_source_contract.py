from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_simulation_help_runs_in_isolated_mode() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(ROOT / "simulate.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "558 videos" in completed.stdout


def test_runtime_has_no_parent_repository_import_path_mutation() -> None:
    forbidden = ("/DATA/", "C:\\\\Users\\", "../LightGenV2", "..\\\\LightGenV2")
    for path in (ROOT / "runtime").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        assert not any(value in text for value in forbidden), path


def test_reference_metrics() -> None:
    release_path = ROOT / "release.json"
    if not release_path.is_file():
        pytest.skip("code checkout intentionally excludes release assets")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    metrics = release["simulation_metrics"]
    assert metrics["count"] == 558
    assert metrics["srcc"] == pytest.approx(0.8022806420367042, abs=1e-12)
    assert release["checkpoint_sha256"] == (
        "5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c"
    )


def test_metric_implementation_handles_ties() -> None:
    try:
        import torch
    except (ImportError, OSError) as error:
        pytest.skip(f"PyTorch runtime is unavailable on this host: {error}")
    sys.path.insert(0, str(ROOT / "runtime"))
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import (
        regression_metrics,
    )

    result = regression_metrics(
        torch.tensor([1.0, 2.0, 2.0, 4.0]),
        torch.tensor([1.0, 3.0, 3.0, 5.0]),
        "temporal",
    )
    assert result["srcc"] == pytest.approx(1.0)
    assert result["count"] == 4
