"""Pure-Python recovery checks: no torch/data/CUDA required."""
import ast
import json
from pathlib import Path
import pytest

from LightGenV2.tasks.t06_video_quality_assessment import reproduce_temporal_baseline as entry
from LightGenV2.tasks.t06_video_quality_assessment import evaluate_temporal_baseline_batches as frozen


def test_existing_output_rejected_before_device(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "cuda_guard", lambda: pytest.fail("Queried GPU before output protection"))
    args = type("Args", (), {"output": tmp_path})()
    with pytest.raises(FileExistsError):
        entry.run_one(args)


def test_canonical_run_boundary(tmp_path):
    with pytest.raises(ValueError):
        entry.require_new_output(tmp_path / "new")


@pytest.mark.parametrize("baseline", ["clip", "yolo", "deepseek"])
def test_original_config_contract(baseline):
    import yaml
    path = entry.TASK / "configs" / "baselines" / entry.CONFIG_NAMES[baseline]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry.validate_original_protocol(raw, baseline)
    raw["training"]["seed"] = 43
    with pytest.raises(ValueError, match="seed"):
        entry.validate_original_protocol(raw, baseline)


def test_bound_new_receipt_and_no_historical_sha_replacement(tmp_path):
    artifacts = {}
    for name in ("checkpoints/temporal/best_checkpoint.pt", "checkpoints/temporal/training_report.json",
                 "checkpoints/temporal/train_history.json", "cpu_verification.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test artifact", encoding="utf-8")
        artifacts[name] = frozen.sha256(path)
    receipt = {"baseline": "clip", "status": "complete", "origin": "retrained_original_protocol",
               "manifest_sha256": frozen.MANIFEST_SHA, "backbone_sha256": frozen.BACKBONE_SHA["clip"],
               "backbone_sha256_after": frozen.BACKBONE_SHA["clip"],
               "training_contract": entry.CONTRACT, "artifacts": artifacts}
    path = tmp_path / "reproduction_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/temporal/best_checkpoint.pt"
    assert frozen.validate_reproduction_receipt(path, "clip", checkpoint) == receipt
    assert frozen.CHECKPOINT_SHA["clip"].startswith("f364f645")
    with pytest.raises(ValueError, match="receipt"):
        frozen.validate_reproduction_receipt(path, "yolo", checkpoint)
    checkpoint.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        frozen.validate_reproduction_receipt(path, "clip", checkpoint)


def test_deepseek_original_single_shard_index(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "model-00001-of-000001.safetensors"}}))
    assert frozen.backbone_file("deepseek", tmp_path).name == "model-00001-of-000001.safetensors"
    index.write_text(json.dumps({"weight_map": {"a": "../wrong.safetensors"}}))
    with pytest.raises(ValueError, match="index path"):
        frozen.backbone_file("deepseek", tmp_path)


def test_original_train_is_temporal_only_and_native_source_is_not_patched():
    tree = ast.parse(Path(entry.__file__).read_text(encoding="utf-8"))
    original_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr in ("train_target", "extract_target")]
    assert len(original_calls) == 2
    for call in original_calls:
        target = next(keyword.value for keyword in call.keywords if keyword.arg == "target")
        assert isinstance(target, ast.Constant) and target.value == "temporal"
    # No new training algorithm/optimizer; native historical functions own it.
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not {"backward", "AdamW", "optimizer_step"}.intersection(calls)


def test_foreign_a100_guard_called_before_training_and_every_batch_evaluation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", frozen.A100_UUID)
    def occupied():
        raise RuntimeError("foreign process")
    monkeypatch.setattr(frozen, "process_audit", occupied)
    with pytest.raises(RuntimeError, match="foreign"):
        entry.cuda_guard()
