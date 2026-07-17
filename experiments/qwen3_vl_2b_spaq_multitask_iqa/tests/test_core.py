from __future__ import annotations

import csv
import json
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_2b_spaq_multitask_iqa import TASK_NAMES
from experiments.qwen3_vl_2b_spaq_multitask_iqa.datasets import load_spaq
from experiments.qwen3_vl_2b_spaq_multitask_iqa.data_prepare import ensure_spaq_dataset
from experiments.qwen3_vl_2b_spaq_multitask_iqa.features import (
    full_multimodal_features,
    load_feature_cache,
    preprocess_image_text,
)
from experiments.qwen3_vl_2b_spaq_multitask_iqa.metrics import multitask_metrics
from experiments.qwen3_vl_2b_spaq_multitask_iqa.modeling import MultitaskRegressionHead
from experiments.qwen3_vl_2b_spaq_multitask_iqa.run import main
from experiments.qwen3_vl_2b_spaq_multitask_iqa.settings import Settings
from experiments.qwen3_vl_2b_spaq_multitask_iqa.training import (
    evaluate_test,
    train_regression_head,
)


def _make_spaq(root: Path, count: int = 20) -> Path:
    image_dir = root / "TestImage"
    image_dir.mkdir(parents=True)
    csv_path = root / "labels.csv"
    rows = []
    for index in range(count):
        name = f"image_{index:03d}.png"
        Image.new("RGB", (12, 10), color=(index, 2 * index, 3 * index)).save(image_dir / name)
        rows.append(
            {
                "Image name": name,
                "MOS": 20 + index,
                "Brightness": 30 + index,
                "Colorfulness": 40 + index,
                "Contrast": 50 + index,
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _settings(tmp_path: Path, data_root: Path) -> Settings:
    settings = Settings(
        data_root=data_root,
        output_dir=tmp_path / "run",
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
        feature_batch_size=2,
        head_batch_size=8,
        num_workers=0,
        epochs=1,
    )
    settings.validate()
    return settings


def test_spaq_discovery_split_and_virtual_tasks(tmp_path: Path) -> None:
    data_root = tmp_path / "SPAQ"
    _make_spaq(data_root)
    settings = _settings(tmp_path, data_root)
    bundle = load_spaq(settings)
    train_names = {record.image_name for record in bundle.train_records}
    test_names = {record.image_name for record in bundle.test_records}
    assert not train_names.intersection(test_names)
    assert len(train_names) == 18
    assert len(test_names) == 2
    assert len(bundle.train) == 4 * len(train_names)
    assert {sample.task for sample in bundle.train.samples} == set(TASK_NAMES)
    saved = json.loads((settings.output_dir / "data_split.json").read_text(encoding="utf-8"))
    assert saved["split_unit"] == "original_image"
    assert saved["seed"] == 42


def test_annotation_failure_lists_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "SPAQ"
    data_root.mkdir()
    (data_root / "wrong.csv").write_text("filename,MOS\na.jpg,50\n", encoding="utf-8")
    settings = _settings(tmp_path, data_root)
    with pytest.raises(RuntimeError, match="Brightness") as error:
        load_spaq(settings)
    assert "wrong.csv" in str(error.value)
    assert "filename" in str(error.value)


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert messages[0]["content"][0]["type"] == "image"
        return "<image>" + messages[0]["content"][1]["text"]

    def __call__(self, text, images, return_tensors, padding):
        batch = len(images)
        return {
            "input_ids": torch.ones(batch, 5, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]] * batch),
            "pixel_values": torch.ones(batch, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * batch),
        }


class _FakeQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.eval()

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs):
        batch, length = input_ids.shape
        hidden = torch.arange(batch * length * 2048, dtype=torch.float32).reshape(batch, length, 2048)
        return SimpleNamespace(hidden_states=(hidden * 0.0, hidden))


def test_full_multimodal_last_token_feature() -> None:
    processor = _FakeProcessor()
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    inputs = preprocess_image_text(processor, images, ["first", "second"])
    features, positions = full_multimodal_features(_FakeQwen(), inputs)
    assert features.shape == (2, 2048)
    assert positions.tolist() == [4, 4]
    assert torch.equal(features[0], torch.arange(4 * 2048, 5 * 2048, dtype=torch.float32))


def test_shared_head_and_metrics() -> None:
    head = MultitaskRegressionHead(2048, 64, 0.1)
    output = head(torch.randn(3, 2048))
    assert output.shape == (3,)
    assert torch.all((output >= 0) & (output <= 1))
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
    rows = []
    for task in TASK_NAMES:
        for value in (10.0, 20.0, 30.0):
            rows.append({"task": task, "image_name": f"{task}_{value}", "true_score": value, "predicted_score": value})
    metrics = multitask_metrics(rows)
    assert metrics["macro_average"]["mae"] == 0.0
    assert metrics["macro_average"]["srcc"] == pytest.approx(1.0)
    assert metrics["macro_average"]["plcc"] == pytest.approx(1.0)


def test_synthetic_cache_train_and_test_outputs(tmp_path: Path) -> None:
    settings = _settings(tmp_path, tmp_path)
    settings.output_dir.mkdir(parents=True)
    samples = 24
    features = torch.randn(samples, 2048)
    normalized = torch.linspace(0.05, 0.95, samples)
    cache = {
        "features": features,
        "normalized_scores": normalized,
        "scores": normalized * 100,
        "task_indices": torch.tensor([index % 4 for index in range(samples)]),
        "sample_indices": torch.arange(samples),
        "image_names": [f"image_{index // 4}.png" for index in range(samples)],
        "image_paths": [f"/fake/image_{index // 4}.png" for index in range(samples)],
        "tasks": [TASK_NAMES[index % 4] for index in range(samples)],
        "metadata": {"schema": "synthetic"},
    }
    cache_path = settings.output_dir / "features" / "train.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(cache, cache_path)
    loaded = load_feature_cache(cache_path, {"schema": "synthetic"})
    head, history = train_regression_head(loaded, settings, torch.device("cpu"))
    rows, metrics = evaluate_test(head, loaded, settings, torch.device("cpu"))
    assert len(history) == 1
    assert len(rows) == samples
    assert set(metrics["tasks"]) == set(TASK_NAMES)
    assert (settings.output_dir / "training_history.csv").is_file()
    assert (settings.output_dir / "checkpoints" / "final_regression_head.pt").is_file()
    assert (settings.output_dir / "test_predictions.csv").is_file()
    assert (settings.output_dir / "test_metrics.json").is_file()


def test_prepare_data_cli_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "SPAQ"
    _make_spaq(data_root, count=10)
    output_dir = tmp_path / "cli_run"
    config = tmp_path / "smoke.json"
    config.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "device": "cpu",
                "dtype": "float32",
                "attn_implementation": "eager",
                "num_workers": 0,
                "epochs": 1,
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config), "--phase", "prepare_data"]) == 0
    assert (output_dir / "resolved_config.json").is_file()
    assert (output_dir / "data_split.json").is_file()
    assert (output_dir / "dataset.json").is_file()


def test_automatic_download_extract_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _make_spaq(source, count=3)
    archive = tmp_path / "source_spaq.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="SPAQ")

    def fake_download(**kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, target)
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    settings = _settings(tmp_path, tmp_path / "downloaded_spaq")
    settings.keep_download_archive = False
    result = ensure_spaq_dataset(settings)
    assert result["action"] == "download"
    assert result["image_count"] == 3
    assert result["has_annotations"] is True
    assert not (settings.data_root / "_downloads").exists()
