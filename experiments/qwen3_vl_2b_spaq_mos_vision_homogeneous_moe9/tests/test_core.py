from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.datasets import load_spaq
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.metrics import regression_metrics
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.modeling import (NormalizedLinearRegressionHead,
                                                                              build_head,
                                                                              resolve_cached_model_source)
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.optics.geometry import MoEGeometry
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.optics.moe import VisionHomogeneousMoESurrogate
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.processor_cache import (ProcessorCacheStore,
                                                                                     build_processor_cache,
                                                                                     validate_processor_cache)
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.sampling import EpochRotatingSampler
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.settings import load_settings
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.teacher_cache import (expected_metadata,
                                                                                   load_teacher_predictions,
                                                                                   write_teacher_predictions)


CONFIG = "experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos_smoke.json"
MAIN_CONFIG = "experiments/qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9/configs/spaq_mos.json"


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
    assert settings.head_output_activation == "linear"
    assert settings.student_head_learning_rate == pytest.approx(1e-3)
    assert load_settings(MAIN_CONFIG).log_interval_batches == 1500
    assert settings.interlayer_hard_route_mask is True
    assert settings.train_image_limit == 24 and settings.epochs == 1
    head = build_head(settings, 1024)
    output = head(torch.randn(3, 1024))
    assert output.shape == (3,)
    assert sum(parameter.numel() for parameter in head.parameters()) == 3073


def test_teacher_and_student_heads_are_fresh_and_independent() -> None:
    teacher = NormalizedLinearRegressionHead(8, "linear")
    student = NormalizedLinearRegressionHead(8, "linear")
    assert teacher.norm.weight.data_ptr() != student.norm.weight.data_ptr()
    assert teacher.regressor.weight.data_ptr() != student.regressor.weight.data_ptr()
    assert not torch.equal(teacher.regressor.weight, student.regressor.weight)
    with torch.no_grad():
        teacher.norm.weight.fill_(2.0)
    assert torch.equal(student.norm.weight, torch.ones_like(student.norm.weight))
    values = torch.randn(4, 8)
    torch.nn.functional.smooth_l1_loss(student(values), torch.rand(4), beta=0.1).backward()
    assert student.regressor.weight.grad is not None
    assert torch.count_nonzero(student.regressor.weight.grad) > 0


def test_head_teacher_student_state_is_compatible() -> None:
    teacher = NormalizedLinearRegressionHead(8, "linear")
    student = NormalizedLinearRegressionHead(8, "linear")
    student.load_state_dict(teacher.state_dict())
    values = torch.randn(4, 8)
    assert torch.allclose(teacher(values), student(values))
    torch.nn.functional.smooth_l1_loss(student(values), torch.rand(4), beta=0.1).backward()
    assert student.regressor.weight.grad is not None


def test_teacher_prediction_cache_rejects_old_activation(tmp_path: Path) -> None:
    output = tmp_path / "run"; (output / "teacher_cache").mkdir(parents=True)
    predictions = torch.tensor([0.2, 0.8]); targets = torch.tensor([0.1, 0.9])
    spec = NormalizedLinearRegressionHead(8, "linear").specification()
    write_teacher_predictions(output, "train", predictions, targets, spec)
    loaded = load_teacher_predictions(output / "teacher_cache" / "train_teacher_predictions.pt", "linear")
    assert torch.allclose(loaded, predictions, atol=1e-3)
    with pytest.raises(RuntimeError, match="Rerun --phase teacher_train"):
        load_teacher_predictions(output / "teacher_cache" / "train_teacher_predictions.pt", "sigmoid")


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


def test_rotating_sampler_keeps_cache_shards_locally_contiguous() -> None:
    sampler = EpochRotatingSampler(40, None, seed=42, shard_size=8)
    order = list(sampler)
    shard_runs: list[int] = []
    for index in order:
        shard = index // 8
        if not shard_runs or shard_runs[-1] != shard:
            shard_runs.append(shard)
    assert len(shard_runs) == 5
    assert sorted(shard_runs) == list(range(5))


def test_processor_cache_persists_qwen_arrays_and_validates_metadata(tmp_path: Path) -> None:
    class FakeImageProcessor:
        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            rows = []
            grids = []
            for number, _image in enumerate(images, start=1):
                rows.append(torch.full((2, 3), float(number)))
                grids.append([1, 1, 2])
            return {"pixel_values": torch.cat(rows), "image_grid_thw": torch.tensor(grids)}

    processor = SimpleNamespace(image_processor=FakeImageProcessor())
    settings = SimpleNamespace(
        output_dir=tmp_path / "run", cache_dtype="float16", teacher_cache_shard_size=2,
        teacher_cache_log_interval_batches=10, feature_batch_size=2, data_root=tmp_path / "SPAQ",
        resolved_annotations_file="annotations.csv", split_digest="digest", model_id="fake-qwen",
        processor_min_pixels=25600, processor_max_pixels=25600,
    )
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    loader = [(images, torch.tensor([0.1, 0.2]), torch.tensor([0, 1]))]
    manifest = build_processor_cache("train", processor, loader, 2, settings)
    store = ProcessorCacheStore(manifest, max_cached_shards=1)
    validate_processor_cache(store, "train", 2, settings)
    first = store.get(0); second = store.get(1)
    assert first["pixel_values"].shape == (2, 3)
    assert first["pixel_values"].dtype == torch.float16
    assert int(first["visual_token_count"]) == 2
    assert torch.equal(second["image_grid_thw"], torch.tensor([1, 1, 2]))
    assert store.stats()["misses"] == 1 and store.stats()["hits"] == 1


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
    head = NormalizedLinearRegressionHead(8, "linear")
    prediction = head(hidden.mean(0, keepdim=True))
    loss = torch.nn.functional.smooth_l1_loss(prediction, torch.tensor([0.7]), beta=0.1)
    loss.backward()
    assert hidden.shape == (3, 8) and model.last_detector_intensity.shape == (1, 480, 480)
    assert model.input_adapter.weight.grad is not None
    assert model.expert_layers[0].experts[0].raw_phase.grad is not None
    assert model.prompt.router.gate.weight.grad is not None
    assert head.regressor.weight.grad is not None
