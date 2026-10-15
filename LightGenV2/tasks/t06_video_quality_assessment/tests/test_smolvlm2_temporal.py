"""CPU contract tests; importing the launcher does not import Torch or use GPUs."""
import copy
import hashlib
import json

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import smolvlm2_temporal_baseline as smol


def test_fixed_contract():
    c = smol.read_config()
    assert c["hidden_width"] * 5 == c["quality_output_trainable_parameters"] == 10240
    assert c["backbone_trainable_parameters"] == 0
    assert len(c["official_assets"]) == 14


@pytest.mark.parametrize("phase", ["evaluate", "suite", "prepare"])
def test_3090_override_rejected_for_cost_and_suite(phase):
    with pytest.raises(ValueError):
        smol.select_gpu(phase, smol.TRAINING_3090_UUID)


@pytest.mark.parametrize("phase", ["train", "gpu-smoke"])
def test_3090_is_training_only(phase):
    assert smol.select_gpu(phase, smol.TRAINING_3090_UUID) == smol.TRAINING_3090_UUID
    assert smol.select_gpu(phase) == smol.frozen.A100_UUID
    with pytest.raises(ValueError):
        smol.select_gpu(phase, "GPU-other")
    assert smol.select_gpu("evaluate") == smol.frozen.A100_UUID


def test_3090_guard_requires_exact_visible_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    with pytest.raises(RuntimeError, match="explicitly authorized"):
        smol.training_guard(smol.TRAINING_3090_UUID)


@pytest.mark.parametrize("foreign", [False, True])
def test_3090_guard_does_not_ignore_foreign_process(monkeypatch, foreign):
    from types import SimpleNamespace
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", smol.TRAINING_3090_UUID)
    monkeypatch.setattr(smol.subprocess, "check_output", lambda *a, **k: "NVIDIA GeForce RTX 3090\n")
    records = f"{smol.os.getpid()}, {smol.TRAINING_3090_UUID}, python\n"
    records += "12345, GPU-other, other\n"
    if foreign:
        records += f"999999, {smol.TRAINING_3090_UUID}, foreign\n"
    monkeypatch.setattr(smol.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=records))
    if foreign:
        with pytest.raises(RuntimeError, match="occupied"):
            smol.training_guard(smol.TRAINING_3090_UUID)
    else:
        rows = smol.training_guard(smol.TRAINING_3090_UUID)
        assert len(rows) == 1 and rows[0]["pid"] == str(smol.os.getpid())


def test_training_guard_preserves_a100_guard(monkeypatch):
    called = []
    monkeypatch.setattr(smol, "guard", lambda: called.append("a100") or [])
    assert smol.training_guard(smol.frozen.A100_UUID) == []
    assert called == ["a100"]


def test_running_source_change_rejected(monkeypatch):
    monkeypatch.setattr(smol, "sources", lambda: {"x": "changed"})
    with pytest.raises(RuntimeError, match="source changed"):
        smol.run_training_guard({"source_sha256": {"x": "original"}, "gpu_uuid": smol.TRAINING_3090_UUID})


@pytest.mark.parametrize("key,value", [("epochs", 100), ("learning_rate", .01),
    ("frames", 8), ("backbone_trainable_parameters", 1), ("quality_output_trainable_parameters", 20480),
    ("selection", "validation"), ("inference_batches", [1, 2, 4, 8]), ("processor_native_frame_size", 448)])
def test_contract_changes_rejected(key, value):
    c = copy.deepcopy(smol.read_config())
    c[key] = value
    with pytest.raises(ValueError):
        smol.validate_config(c)


def test_existing_output_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        smol.new_output(tmp_path)


def test_broad_output_rejected():
    with pytest.raises((ValueError, FileExistsError)):
        smol.new_output(smol.TASK / "runs/simulation")


def test_git_blob_validation(tmp_path):
    p = tmp_path / "config.json"
    p.write_bytes(b"{}")
    sha = hashlib.sha1(b"blob 2\0{}").hexdigest()
    assert smol.asset_valid(p, {"git_blob_sha1": sha})
    assert not smol.asset_valid(p, {"git_blob_sha1": "0" * 40})


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_cost_and_tail_batch(batch):
    plan = smol.telemetry.inference_plan(558, batch)
    assert sum(end-begin for begin, end in plan) == 558
    assert sum(end-begin for begin, end in plan if end <= 200) == 200
    costs = [[i, end-begin, True, float(end-begin), 0] for i, (begin, end) in enumerate(plan) if end <= 200]
    result = smol.telemetry.workload_cost(costs, 100.)
    assert result["latency_16_videos_ms"] == 16.
    assert result["energy_16_videos_j"] == pytest.approx(7.0112)


def test_prompt_matches_qwen_source():
    import ast
    tree = ast.parse((smol.TASK / "quality_token_common.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPTS" for t in n.targets))
    assert smol.PROMPT == ast.literal_eval(node.value)["temporal"]


def test_only_five_rows_can_update():
    import sys
    if sys.platform == "win32":
        pytest.skip("Run Torch gradient isolation on server; local Torch DLL is unavailable")
    try:
        import torch
    except (ImportError, OSError):
        pytest.skip("Torch unavailable locally; required to pass on server")
    from LightGenV2.tasks.t06_video_quality_assessment.quality_token_train import FiveNativeTokenRows
    native = torch.nn.Linear(2048, 20, bias=False).eval().requires_grad_(False)
    before = native.weight.detach().clone()
    initial = native.weight.detach()[:5].clone()
    head = FiveNativeTokenRows(initial)
    assert head.weight.data_ptr() != initial.data_ptr()
    assert sum(p.numel() for p in head.parameters() if p.requires_grad) == 10240
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
    loss = torch.nn.functional.cross_entropy(head(torch.ones(3, 2048)), torch.tensor([0, 1, 2]))
    loss.backward()
    optimizer.step()
    assert native.weight.grad is None
    assert torch.equal(native.weight, before)
    assert torch.equal(initial, before[:5])
    assert not torch.equal(head.weight, initial)
