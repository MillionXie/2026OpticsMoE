from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.datasets import (
    FSSSaliencyDataset,
    prepare_fss1000,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import (
    LightweightSegmentationHead,
    restore_detector_spatial,
    restore_packed_spatial,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.objectives import (
    SegmentationAccumulator,
    boundary_dice_loss,
    dice_loss,
    segmentation_loss,
    soft_iou_loss,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.training import (
    _student_scheduler,
    align_cached_teacher_logits,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.visualization import (
    save_optical_debug_example,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_formal_config_keeps_validated_moe16_geometry() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "fss1000_saliency.yaml")
    assert settings.image_size == 224
    assert settings.processor_min_pixels == 50176
    assert (
        settings.canvas_size,
        settings.active_size,
        settings.expert_size,
        settings.num_experts,
        settings.top_k,
        settings.expert_layers,
        settings.detector_output_size,
    ) == (1026, 986, 224, 16, 4, 1, 224)
    assert settings.mask_kd_weight == 0.0
    assert settings.student_initial_checkpoint is None


def test_final_config_replays_crop_and_flip_for_mask_kd() -> None:
    settings = load_settings(
        EXPERIMENT
        / "configs"
        / "fss1000_saliency_single_layer_from_scratch_100ep.yaml"
    )
    assert settings.mask_kd_align_augmentation is True
    assert settings.augmentation_enabled is True
    assert settings.rotation_degrees == 0.0

    probability = torch.tensor(
        [[[[0.1, 0.2, 0.3, 0.4],
           [0.2, 0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8, 0.9],
           [0.7, 0.8, 0.9, 0.95]]]],
        dtype=torch.float32,
    )
    logits = torch.logit(probability)
    trace = [{
        "crop_box_normalized": [0.0, 0.5, 1.0, 1.0],
        "horizontal_flip": True,
        "rotation_degrees": 0.0,
    }]
    actual = align_cached_teacher_logits(logits, trace).sigmoid()
    expected_crop = torch.flip(probability[:, :, 2:4, :], dims=(-1,))
    expected = torch.nn.functional.interpolate(
        expected_crop, size=(4, 4), mode="bilinear", align_corners=False
    )
    assert torch.allclose(actual, expected, atol=2e-4)


def test_refinement_head_is_zero_initialized_and_lightweight() -> None:
    torch.manual_seed(7)
    base = LightweightSegmentationHead(224, 128, (64, 32, 16), 8)
    refined = LightweightSegmentationHead(
        224, 128, (64, 32, 16), 8, refinement_enabled=True
    )
    source = base.state_dict()
    result = refined.load_state_dict(source, strict=False)
    assert result.unexpected_keys == []
    assert all(name.startswith("refinement.") for name in result.missing_keys)
    features = torch.randn(2, 224, 14, 14)
    assert torch.equal(base(features), refined(features))
    added = sum(p.numel() for p in refined.parameters()) - sum(
        p.numel() for p in base.parameters()
    )
    assert 0 < added < 5_000


def test_progressive_refinement_preserves_initial_logits() -> None:
    torch.manual_seed(11)
    base = LightweightSegmentationHead(224, 128, (64, 32, 16), 8)
    progressive = LightweightSegmentationHead(
        224,
        128,
        (64, 32, 16),
        8,
        progressive_refinement_enabled=True,
    )
    result = progressive.load_state_dict(base.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert all(
        name.startswith(("progressive_refinement.", "progressive_classifier."))
        for name in result.missing_keys
    )
    features = torch.randn(2, 224, 14, 14)
    assert torch.equal(base(features), progressive(features))
    added = sum(p.numel() for p in progressive.parameters()) - sum(
        p.numel() for p in base.parameters()
    )
    assert 90_000 < added < 110_000


def test_detector_input_residual_is_checkpoint_compatible() -> None:
    torch.manual_seed(19)
    base = LightweightSegmentationHead(224, 128, (64, 32, 16), 8)
    residual = LightweightSegmentationHead(
        224,
        128,
        (64, 32, 16),
        8,
        detector_residual_enabled=True,
        detector_input_scale_init=0.0,
    )
    result = residual.load_state_dict(base.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert set(result.missing_keys) == {
        "detector_identity_scale",
        "detector_input_scale",
    }
    detector = torch.randn(2, 224, 14, 14)
    input_feature = torch.randn_like(detector)
    assert torch.equal(base(detector), residual(detector, input_feature))
    residual(detector, input_feature).mean().backward()
    assert residual.detector_input_scale.grad is not None
def test_iou_boundary_objectives_are_differentiable() -> None:
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 5:13] = 1
    perfect_logits = torch.where(target.bool(), 20.0, -20.0)
    assert float(soft_iou_loss(perfect_logits, target)) < 1e-5
    assert float(boundary_dice_loss(perfect_logits, target)) < 1e-5

    logits = torch.randn_like(target, requires_grad=True)
    loss, parts = segmentation_loss(
        logits,
        target,
        bce_weight=1.0,
        dice_weight=1.0,
        soft_iou_weight=0.75,
        boundary_weight=0.25,
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert set(("soft_iou_loss", "boundary_loss")).issubset(parts)


def test_final_from_scratch_profile_is_single_layer_and_reproducible() -> None:
    settings = load_settings(
        EXPERIMENT / "configs"
        / "fss1000_saliency_single_layer_from_scratch_100ep.yaml"
    )
    assert settings.student_epochs == 100
    assert settings.student_batch_size == 16
    assert settings.inference_batch_size == 16
    assert settings.expert_layers == 1
    assert settings.student_initial_checkpoint is None
    assert settings.mask_kd_align_augmentation is True
    assert settings.soft_iou_weight == pytest.approx(0.75)
    assert settings.boundary_weight == pytest.approx(0.25)
    assert settings.student_lr_schedule == "cosine"
    assert settings.student_lr_min_ratio == pytest.approx(0.05)
    assert settings.checkpoint_interval_epochs == 10
    assert settings.student_learning_rate == pytest.approx(1e-3)
    assert settings.phase_learning_rate == pytest.approx(1e-3)
    assert settings.router_learning_rate == pytest.approx(5e-4)
    assert settings.visualization_after_training is True
    assert settings.visualization_optical_sample_count == 4

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = _student_scheduler(optimizer, settings)
    assert scheduler is not None
    initial = optimizer.param_groups[0]["lr"]
    for _ in range(settings.student_epochs - 1):
        optimizer.step()
        scheduler.step()
    final = optimizer.param_groups[0]["lr"]
    assert initial == pytest.approx(1e-3)
    assert final == pytest.approx(1e-3 * 0.05)


def test_optical_debug_writer_saves_all_views(tmp_path: Path) -> None:
    save_optical_debug_example(
        tmp_path,
        input_field=torch.rand(8, 8),
        amplitude_slm=torch.rand(16, 16),
        stage_fields=[torch.complex(torch.rand(16, 16), torch.rand(16, 16))],
        detector_intensity=torch.rand(12, 12),
        detector_readout=torch.rand(8, 8),
        routing_weights=torch.tensor([0.6, 0.0, 0.4, 0.0]),
        selected_mask=torch.tensor([True, False, True, False]),
        grid_rows=2,
        grid_cols=2,
    )
    expected = {
        "01_optical_input_field.png",
        "02_amplitude_slm_canvas.png",
        "03_expert_stage_01_intensity.png",
        "04_detector_intensity.png",
        "05_detector_readout_224.png",
        "06_routing_weights.png",
    }
    assert expected == {path.name for path in tmp_path.glob("*.png")}


def test_restore_teacher_tokens_uses_runtime_grid() -> None:
    grid = torch.tensor([[1, 2, 3], [1, 2, 3]])
    packed = torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5)
    spatial = restore_packed_spatial(packed, grid)
    assert spatial.shape == (2, 5, 2, 3)
    assert torch.equal(spatial[0].permute(1, 2, 0).reshape(6, 5), packed[:6])


def test_restore_detector_reads_only_valid_rows() -> None:
    grid = torch.tensor([[1, 2, 3], [1, 2, 3]])
    detector = torch.randn(2, 224, 224)
    spatial = restore_detector_spatial(detector, grid, [6, 6])
    assert spatial.shape == (2, 224, 2, 3)
    assert torch.equal(
        spatial[1].permute(1, 2, 0).reshape(6, 224),
        detector[1, :6],
    )


def test_token_grid_mismatch_raises_instead_of_crop_or_reshape() -> None:
    grid = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="does not match"):
        restore_packed_spatial(torch.randn(5, 8), grid)
    with pytest.raises(RuntimeError, match="optical token count"):
        restore_detector_spatial(torch.randn(1, 224, 224), grid, [5])


def test_lightweight_heads_return_224_logits_and_are_small() -> None:
    teacher_head = LightweightSegmentationHead(1024, 128, (64, 32, 16), 8)
    student_head = LightweightSegmentationHead(224, 128, (64, 32, 16), 8)
    teacher_logits = teacher_head(torch.randn(2, 1024, 14, 14))
    student_logits = student_head(torch.randn(2, 224, 14, 14))
    assert teacher_logits.shape == student_logits.shape == (2, 1, 224, 224)
    assert sum(p.numel() for p in teacher_head.parameters()) < 500_000
    assert sum(p.numel() for p in student_head.parameters()) < 500_000


def test_bce_dice_and_metrics_are_finite() -> None:
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1
    logits = torch.where(target.bool(), torch.tensor(8.0), torch.tensor(-8.0))
    loss, parts = segmentation_loss(
        logits, target, bce_weight=1.0, dice_weight=1.0
    )
    assert torch.isfinite(loss)
    assert parts["bce"] < 0.001
    assert parts["dice_loss"] < 0.001
    accumulator = SegmentationAccumulator()
    accumulator.update(logits, target, loss=loss)
    metrics = accumulator.compute()
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_mask_kd_uses_only_final_teacher_mask() -> None:
    target = torch.zeros(1, 1, 4, 4)
    student = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn_like(target)
    loss, parts = segmentation_loss(
        student,
        target,
        bce_weight=1.0,
        dice_weight=1.0,
        teacher_logits=teacher,
        mask_kd_weight=0.5,
    )
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(parts["mask_kd"])


def test_official_test_classes_are_disjoint(tmp_path: Path) -> None:
    # The loader deliberately requires the complete 240-class official test
    # profile even in a synthetic unit test.
    test_classes = [f"test_{index:03d}" for index in range(240)]
    (tmp_path / "fss_test_set.txt").write_text(
        "\n".join(test_classes) + "\n", encoding="utf-8"
    )
    for class_name in ["train_only", *test_classes]:
        folder = tmp_path / "fewshot_data" / class_name
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), (120, 30, 10)).save(folder / "1.jpg")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(folder / "1.png")
    settings = SimpleNamespace(
        data_root=tmp_path,
        output_dir=tmp_path / "run",
        download=False,
        download_source="auto",
        huggingface_dataset_id="unused",
        huggingface_endpoint="unused",
        download_file_id="unused",
        official_test_list_url="unused",
        merge_official_validation_into_train=True,
        train_class_limit=None,
        test_class_limit=None,
        images_per_class_limit=None,
    )
    bundle = prepare_fss1000(settings)
    assert bundle.train_classes == ("train_only",)
    assert len(bundle.test_classes) == 240
    assert not (set(bundle.train_classes) & set(bundle.test_classes))


def test_mask_resize_remains_binary(tmp_path: Path) -> None:
    folder = tmp_path / "class"
    folder.mkdir()
    Image.new("RGB", (13, 17), "white").save(folder / "1.jpg")
    mask = np.zeros((17, 13), dtype=np.uint8)
    mask[3:11, 4:9] = 255
    Image.fromarray(mask).save(folder / "1.png")
    record = SimpleNamespace(
        sample_index=0,
        sample_id="class/1",
        class_name="class",
        image_path=folder / "1.jpg",
        mask_path=folder / "1.png",
    )
    settings = SimpleNamespace(
        image_size=224,
        augmentation_enabled=False,
        crop_scale_min=0.9,
        horizontal_flip_probability=0.5,
        rotation_degrees=5.0,
        brightness_jitter=0.1,
        contrast_jitter=0.1,
    )
    item = FSSSaliencyDataset((record,), settings, training=False)[0]
    assert item["mask"].shape == (1, 224, 224)
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}


def test_prepare_quarantines_source_geometry_mismatch(tmp_path: Path) -> None:
    test_classes = [f"test_{index:03d}" for index in range(240)]
    (tmp_path / "fss_test_set.txt").write_text(
        "\n".join(test_classes) + "\n", encoding="utf-8"
    )
    for class_name in ["train_only", *test_classes]:
        folder = tmp_path / "fewshot_data" / class_name
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(folder / "1.jpg")
        Image.new("L", (8, 8), 255).save(folder / "1.png")
    broken = tmp_path / "fewshot_data" / test_classes[0]
    Image.new("RGB", (8, 8), "white").save(broken / "2.jpg")
    Image.new("L", (16, 4), 255).save(broken / "2.png")
    settings = SimpleNamespace(
        data_root=tmp_path,
        output_dir=tmp_path / "run",
        download=False,
        download_source="auto",
        huggingface_dataset_id="unused",
        huggingface_endpoint="unused",
        download_file_id="unused",
        official_test_list_url="unused",
        merge_official_validation_into_train=True,
        train_class_limit=None,
        test_class_limit=None,
        images_per_class_limit=None,
    )
    bundle = prepare_fss1000(settings)
    assert len(bundle.test_records) == 240
    assert bundle.metadata["ignored_geometry_mismatch"] == 1
    ignored = (settings.output_dir / "manifests" / "ignored_samples.csv").read_text(
        encoding="utf-8"
    )
    assert "source_image_mask_geometry_mismatch" in ignored
    assert f"{test_classes[0]}/2" in ignored
