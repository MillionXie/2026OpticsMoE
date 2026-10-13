"""CPU-only contract tests; no model/data/device is loaded."""
import argparse
import ast
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import evaluate_temporal_baseline_batches as entry


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_cost_excludes_audit_and_uses_16_video_workload(batch):
    cost = entry.system_cost(74.27413415163755 * batch, 50.65795031055901, batch)
    assert cost["latency_16_videos_ms"] == pytest.approx(1188.3861464262009)
    assert cost["energy_16_videos_j"] == pytest.approx(462.1134010767563)
    assert cost["energy_per_video_j"] == pytest.approx(cost["energy_16_videos_j"] / 16)


def test_existing_output_rejected_before_asset_or_gpu_check(tmp_path):
    args = argparse.Namespace(output=tmp_path, manifest=tmp_path / "missing", checkpoint=tmp_path / "missing", baseline="qwen")
    with pytest.raises(FileExistsError):
        entry.preflight_files(args)


def test_output_outside_canonical_task_rejected(tmp_path):
    args = argparse.Namespace(output=tmp_path / "new", baseline="qwen")
    with pytest.raises(ValueError, match="canonical"):
        entry.preflight_files(args)


def test_no_training_or_warmup():
    source = Path(entry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not {"backward", "step", "train"}.intersection(calls)
    assert "explicit_warmup_forwards\": 0" in source
    assert "cpu_inputs, positions = prepare(selected)" in source
    assert "in_cost = start + len(selected) <= 200" in source


def test_invalid_cost_is_not_silently_accepted():
    with pytest.raises(ValueError):
        entry.system_cost(0, 50, 1)
    with pytest.raises(ValueError):
        entry.system_cost(10, 50, 3)
