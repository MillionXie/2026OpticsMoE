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
    dice_loss,
    segmentation_loss,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.training import (
    align_cached_teacher_logits,
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


def test_mask_kd_finetune_config_uses_fresh_optimizer_initialization() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "fss1000_saliency_mask_kd_finetune.yaml"
    )
    assert settings.mask_kd_weight == pytest.approx(0.5)
    assert settings.student_initial_checkpoint is not None
    assert settings.student_learning_rate == pytest.approx(3e-4)
    assert settings.phase_learning_rate == pytest.approx(1e-4)
    assert settings.router_learning_rate == pytest.approx(1e-4)
    assert settings.augmentation_enabled is False


def test_augmented_mask_kd_config_replays_crop_and_flip() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "fss1000_saliency_mask_kd_augmented_finetune.yaml"
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


def test_augmented_mask_kd_batch16_profile() -> None:
    settings = load_settings(
        EXPERIMENT / "configs"
        / "fss1000_saliency_mask_kd_augmented_finetune_batch16.yaml"
    )
    assert settings.student_batch_size == 16
    assert settings.inference_batch_size == 16
    assert settings.mask_kd_align_augmentation is True


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
