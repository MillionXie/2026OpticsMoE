from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.datasets import load_spaq
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.metrics import regression_metrics
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.modeling import (NormalizedLinearRegressionHead,
                                                                              build_head,
                                                                              initialize_student_head,
                                                                              resolve_cached_model_source)
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.optics.geometry import MoEGeometry
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.optics.moe import VisionHomogeneousMoESurrogate
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.sampling import EpochRotatingSampler
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.settings import load_settings
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.teacher_cache import expected_metadata


CONFIG = "experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos_smoke.json"


def _encoder(hidden_size: int = 8) -> VisionHomogeneousMoESurrogate:
    module = VisionHomogeneousMoESurrogate.__new__(VisionHomogeneousMoESurrogate)
    nn.Module.__init__(module)
    module.max_visual_tokens = 120
    module.geometry = MoEGeometry()
    module.input_adapter = nn.Linear(hidden_size, 120)
    module.input_norm = nn.LayerNorm(120)
    module.nonnegative = nn.Softplus()
    return module


def test_grouped_config_and_identical_small_regression_head() -> None:
    settings = load_settings(CONFIG)
    assert settings.dataset == "spaq_mos" and settings.task_name == "MOS"
    assert settings.head_output_activation == "sigmoid"
    assert settings.student_head_output_activation == "linear"
    assert settings.student_head_zero_initialize_regressor is True
    assert settings.student_head_learning_rate == pytest.approx(1e-3)
    assert settings.interlayer_hard_route_mask is True
    assert settings.train_image_limit == 24 and settings.epochs == 1
    head = build_head(settings, 1024)
    output = head(torch.randn(3, 1024))
    assert output.shape == (3,)
    assert torch.all((0.0 <= output) & (output <= 1.0))
    assert sum(parameter.numel() for parameter in head.parameters()) == 3073


def test_student_linear_head_zero_initialization_keeps_teacher_unchanged() -> None:
    teacher = NormalizedLinearRegressionHead(8, "sigmoid")
    teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    student = NormalizedLinearRegressionHead(8, "linear")
    initialize_student_head(student, teacher, zero_initialize_regressor=True)
    values = torch.randn(4, 8)
    assert torch.equal(student.norm.weight, teacher.norm.weight)
    assert torch.equal(student.norm.bias, teacher.norm.bias)
    assert torch.count_nonzero(student.regressor.weight) == 0
    assert torch.count_nonzero(student.regressor.bias) == 0
    assert torch.equal(student(values), torch.zeros(4))
    torch.nn.functional.smooth_l1_loss(student(values), torch.rand(4), beta=0.1).backward()
    assert student.regressor.weight.grad is not None
    assert torch.count_nonzero(student.regressor.weight.grad) > 0
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_before[name])


def test_head_teacher_student_state_is_compatible() -> None:
    teacher = NormalizedLinearRegressionHead(8, "sigmoid")
    student = NormalizedLinearRegressionHead(8, "sigmoid")
    student.load_state_dict(teacher.state_dict())
    values = torch.randn(4, 8)
    assert torch.allclose(teacher(values), student(values))
    torch.nn.functional.smooth_l1_loss(student(values), torch.rand(4), beta=0.1).backward()
    assert student.regressor.weight.grad is not None


def test_cached_model_source_avoids_network_without_changing_model_id(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    snapshot = hub / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    assert resolve_cached_model_source("Qwen/Qwen3-VL-2B-Instruct", None) == str(snapshot.resolve())


def test_spaq_mos_loader_uses_rgb_and_persistent_image_split(tmp_path: Path) -> None:
    root = tmp_path / "SPAQ"; images = root / "images"; images.mkdir(parents=True)
    rows = ["Image name,MOS"]
    for index in range(10):
        name = f"image_{index:02d}.jpg"
        Image.new("RGB", (12, 8), (index * 10, 20, 30)).save(images / name)
        rows.append(f"{name},{10 + index * 8}")
    (root / "annotations.csv").write_text("\n".join(rows), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dataset": "spaq_mos", "task_name": "MOS", "data_root": str(root),
                                  "download": False, "output_dir": str(tmp_path / "run")}), encoding="utf-8")
    settings = load_settings(config); bundle = load_spaq(settings)
    assert len(bundle.train) == 9 and len(bundle.test) == 1
    image, normalized = bundle.train[0]
    assert image.mode == "RGB" and 0.0 <= normalized <= 1.0
    assert (settings.output_dir / "data_split.json").is_file()
    again = load_spaq(settings)
    assert [record.image_name for record in bundle.test_records] == [record.image_name for record in again.test_records]


def test_regression_metrics_perfect_prediction() -> None:
    report = regression_metrics([10, 30, 80], [10, 30, 80])
    assert report["mae"] == 0.0 and report["rmse"] == 0.0
    assert report["srcc"] == pytest.approx(1.0) and report["plcc"] == pytest.approx(1.0)


def test_teacher_cache_schema_contains_regression_targets_only() -> None:
    settings = load_settings(CONFIG); settings.split_digest = "abc"; settings.resolved_annotations_file = "ann.csv"
    metadata = expected_metadata("train", 24, settings, None)
    assert metadata["dataset"] == "spaq_mos" and metadata["task"] == "MOS"
    assert "targets" in metadata["cached_tensors"] and "labels" not in metadata["cached_tensors"]
    assert metadata["target_scale"] == [0.0, 1.0]


def test_rotating_sampler_eventually_changes_epoch_window() -> None:
    sampler = EpochRotatingSampler(20, 7, seed=42)
    first = list(sampler); sampler.set_epoch(2); second = list(sampler)
    assert len(first) == len(second) == 7 and set(first) != set(second)


def test_token_rows_are_nonnegative_zero_padded_without_resize() -> None:
    encoder = _encoder(); field = encoder.encode_groups([torch.randn(60, 8)])
    assert field.shape == (1, 120, 120) and torch.all(field >= 0)
    assert torch.count_nonzero(field[:, 60:]) == 0
    with pytest.raises(RuntimeError, match="visual token count 121"):
        encoder.encode_groups([torch.randn(121, 8)])


def test_full_optical_moe_regression_backward_smoke() -> None:
    settings = load_settings(CONFIG)
    model = VisionHomogeneousMoESurrogate(8, settings)
    hidden = model(torch.randn(3, 8), cu_seqlens=torch.tensor([0, 3], dtype=torch.int32))
    head = NormalizedLinearRegressionHead(8, "sigmoid")
    prediction = head(hidden.mean(0, keepdim=True))
    loss = torch.nn.functional.smooth_l1_loss(prediction, torch.tensor([0.7]), beta=0.1)
    loss.backward()
    assert hidden.shape == (3, 8) and model.last_detector_intensity.shape == (1, 480, 480)
    assert model.input_adapter.weight.grad is not None
    assert model.expert_layers[0].experts[0].raw_phase.grad is not None
    assert model.prompt.router.gate.weight.grad is not None
    assert head.regressor.weight.grad is not None
