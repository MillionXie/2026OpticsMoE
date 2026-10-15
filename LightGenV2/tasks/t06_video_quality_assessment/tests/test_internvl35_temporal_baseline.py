"""CPU contracts; real processor/model smoke is an additional server gate."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import internvl35_temporal_baseline as run


def test_original_five_row_training_budget():
    c = run.read_config()
    assert c["epochs"] == 50 and c["quality_output_trainable_parameters"] == 10240
    assert c["frames"] == 4 and c["native_frame_size"] == 448
    assert c["backbone_trainable_parameters"] == 0


def test_source_identity_includes_new_and_original_dependencies():
    s = run.sources()
    assert len(s) == 9
    assert all(len(sha) == 64 for sha in s.values())
    assert any(p.endswith("internvl35_temporal_baseline.py") for p in s)
    assert any(p.endswith("quality_token_train.py") for p in s)


def test_a100_requires_explicit_accuracy_only_concurrency(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", run.frozen.A100_UUID)
    with pytest.raises(ValueError, match="accuracy-only"):
        run.gpu_guard(SimpleNamespace(gpu_uuid=run.frozen.A100_UUID, accuracy_only_concurrent=False))


def test_foreign_process_is_audited_not_killed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", run.frozen.A100_UUID)
    def query(cmd, **kw):
        if any("--query-gpu=" in c for c in cmd):
            return "NVIDIA A100, 250, 25000\n"
        return f"999, {run.frozen.A100_UUID}, 5000\n"
    monkeypatch.setattr(run.subprocess, "check_output", query)
    result = run.gpu_guard(SimpleNamespace(gpu_uuid=run.frozen.A100_UUID, accuracy_only_concurrent=True))
    assert result[0]["pid"] == "999"


def test_insufficient_memory_rejects_admission(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", run.frozen.A100_UUID)
    monkeypatch.setattr(run.subprocess, "check_output", lambda cmd, **kw:
        "NVIDIA A100, 250, 10000\n" if any("--query-gpu=" in c for c in cmd) else "")
    with pytest.raises(RuntimeError, match="Insufficient"):
        run.gpu_guard(SimpleNamespace(gpu_uuid=run.frozen.A100_UUID, accuracy_only_concurrent=True))


def test_3090_never_used_for_formal_inference(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", run.shared.TRAINING_3090_UUID)
    with pytest.raises(ValueError, match="training/smoke"):
        run.gpu_guard(SimpleNamespace(gpu_uuid=run.shared.TRAINING_3090_UUID, phase="evaluate"))


def test_original_training_algorithm_is_not_an_extended_search():
    tree = ast.parse(Path(run.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_head")
    source = ast.unparse(fn)
    assert "range(1, 51)" in source
    assert "lr=0.001, weight_decay=0.0" in source
    assert "cross_entropy" in source
    assert "CosineAnnealingLR(optimizer, T_max=50)" in source
    assert "LoRA" not in source


def test_latest_accuracy_only_suite_is_single_batch_and_bounded():
    tree = ast.parse(Path(run.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "suite")
    source = ast.unparse(fn)
    assert "inference_batches=[1]" in source
    assert "'--batch', '1'" in source
    assert "10800" in source and "7200" in source
    assert "no automatic retry" in source
