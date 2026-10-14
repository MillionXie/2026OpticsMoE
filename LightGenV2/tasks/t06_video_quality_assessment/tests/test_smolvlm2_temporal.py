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
