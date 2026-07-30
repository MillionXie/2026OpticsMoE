from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.datasets import (
    DIRECTORIES,
    ISICSegmentationDataset,
    paired_transform,
    prepare_isic2016,
)
from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.metrics import (
    ISICSegmentationAccumulator,
    per_sample_metrics,
)
from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.settings import (
    EXPERIMENT_DIR,
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.training import (
    _build_optimizer,
    _configure_trainability,
    initialize_model,
)


CONFIG_DIR = EXPERIMENT_DIR / "configs"


def test_formal_configs_keep_identical_optical_architecture() -> None:
    scratch = load_settings(CONFIG_DIR / "isic2016_scratch.yaml")
    pretrained = load_settings(
        CONFIG_DIR / "isic2016_coco_duts_pretrained.yaml"
    )
    fields = (
        "image_size",
        "canvas_size",
        "active_size",
        "expert_size",
        "expert_pitch",
        "num_experts",
        "expert_layers",
        "top_k",
        "detector_output_size",
    )
    assert {field: getattr(scratch, field) for field in fields} == {
        field: getattr(pretrained, field) for field in fields
    }
    assert scratch.initialization_mode == "scratch_end_to_end"
    assert pretrained.initialization_mode == "coco_duts_pretrained"
    assert scratch.head_warmup_epochs == 0
    assert scratch.output_dir.parent == EXPERIMENT_DIR / "runs"
    assert pretrained.output_dir.parent == EXPERIMENT_DIR / "runs"


def test_official_urls_and_counts_are_configured() -> None:
    settings = load_settings(CONFIG_DIR / "isic2016_scratch.yaml")
    assert settings.expected_train_samples == 900
    assert settings.expected_test_samples == 379
    assert "ISBI2016_ISIC_Part1_Training_Data.zip" in settings.train_image_url
    assert (
        "ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
        in settings.train_mask_url
    )
    assert "ISBI2016_ISIC_Part1_Test_Data.zip" in settings.test_image_url
    assert (
        "ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
        in settings.test_mask_url
    )


def test_prepare_pairs_official_image_and_segmentation_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ISIC2016"
    directories = {key: root / name for key, name in DIRECTORIES.items()}
    for directory in directories.values():
        directory.mkdir(parents=True)
    _save_pair(directories, "train", "ISIC_0000001")
    _save_pair(directories, "train", "ISIC_0000002")
    _save_pair(directories, "test", "ISIC_9000001")
    settings = _dataset_settings(root, tmp_path / "run")
    bundle = prepare_isic2016(settings, persist=True)
    assert len(bundle.train_records) == 2
    assert len(bundle.test_records) == 1
    assert bundle.train_records[0].sample_id == "ISIC_0000001"
    assert bundle.test_records[0].split == "test"
    assert bundle.metadata["train_test_overlap"] == 0
    assert (settings.output_dir / "manifests" / "samples.csv").is_file()


def test_missing_mask_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ISIC2016"
    directories = {key: root / name for key, name in DIRECTORIES.items()}
    for directory in directories.values():
        directory.mkdir(parents=True)
    _save_pair(directories, "train", "ISIC_0000001")
    _save_pair(directories, "test", "ISIC_9000001")
    Image.new("RGB", (12, 8)).save(
        directories["train_images"] / "ISIC_0000002.jpg"
    )
    settings = _dataset_settings(root, tmp_path / "run")
    with pytest.raises(RuntimeError, match="missing masks"):
        prepare_isic2016(settings, persist=False)


def test_paired_transform_keeps_binary_nearest_mask() -> None:
    image = Image.new("RGB", (17, 11), color=(80, 120, 160))
    mask_array = np.zeros((11, 17), dtype=np.uint8)
    mask_array[2:9, 5:15] = 255
    mask = Image.fromarray(mask_array, mode="L")
    settings = SimpleNamespace(
        image_size=224,
        augmentation_enabled=False,
        crop_scale_min=1.0,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotation_degrees=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    transformed_image, transformed_mask, _ = paired_transform(
        image,
        mask,
        settings,
        training=False,
    )
    assert transformed_image.size == (224, 224)
    assert transformed_mask.size == (224, 224)
    assert set(np.unique(np.asarray(transformed_mask))).issubset({0, 255})


def test_dataset_returns_rgb_and_binary_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "ISIC_1.jpg"
    mask_path = tmp_path / "ISIC_1_Segmentation.png"
    Image.new("RGB", (20, 12), color=(20, 30, 40)).save(image_path)
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[:, 5:15] = 255
    Image.fromarray(mask).save(mask_path)
    record = SimpleNamespace(
        sample_index=0,
        sample_id="ISIC_1",
        split="train",
        image_path=image_path,
        mask_path=mask_path,
    )
    settings = SimpleNamespace(
        image_size=224,
        augmentation_enabled=False,
        crop_scale_min=1.0,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotation_degrees=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    item = ISICSegmentationDataset(
        (record,),
        settings,
        training=False,
    )[0]
    assert item["image"].mode == "RGB"
    assert item["mask"].shape == (1, 224, 224)
    assert set(torch.unique(item["mask"]).tolist()) == {0.0, 1.0}


def test_perfect_prediction_metrics() -> None:
    target = torch.tensor(
        [[[[0.0, 1.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    logits = torch.where(target > 0.5, 20.0, -20.0)
    accumulator = ISICSegmentationAccumulator()
    accumulator.update(logits, target, loss=0.0)
    metrics = accumulator.compute()
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    rows = per_sample_metrics(logits, target)
    assert rows[0]["mean_iou"] == pytest.approx(1.0)
    assert rows[0]["mean_dice"] == pytest.approx(1.0)


def test_scratch_initialization_loads_no_checkpoint() -> None:
    result = initialize_model(
        object(),
        SimpleNamespace(initialization_mode="scratch_end_to_end"),
    )
    assert result["loaded_components"] == []
    assert result["all_task_modules_train_from_epoch_1"] is True


def test_pretrained_initialization_requires_checkpoint(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        initialization_mode="coco_duts_pretrained",
        source_checkpoint=tmp_path / "missing.pt",
    )
    with pytest.raises(FileNotFoundError, match="Initialization checkpoint"):
        initialize_model(object(), settings)


def test_optical_modules_remain_trainable_after_freezing_qwen_visual() -> None:
    class Core(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_adapter = nn.Linear(4, 4)
            self.router = nn.Linear(4, 2)

    class Capture(nn.Module):
        def __init__(self, core: nn.Module, recombiner: nn.Module) -> None:
            super().__init__()
            self.core = core
            self.recombiner = recombiner

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = Core()
            self.recombiner = nn.Linear(4, 4)
            self.visual = nn.Module()
            self.visual.native_qwen = nn.Linear(4, 4)
            self.visual.blocks = nn.ModuleList(
                [Capture(self.core, self.recombiner)]
            )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Backbone()
            self.head = nn.Linear(4, 1)

    model = Model()
    _configure_trainability(model, warmup=False)
    assert all(
        parameter.requires_grad for parameter in model.backbone.core.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.backbone.recombiner.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in model.backbone.visual.native_qwen.parameters()
    )
    optimizer = _build_optimizer(
        model,
        SimpleNamespace(
            optical_learning_rate=1e-4,
            router_learning_rate=1e-4,
            recombiner_learning_rate=1e-4,
            head_learning_rate=1e-3,
            weight_decay=0.0,
        ),
        warmup=False,
    )
    counts = {
        group["name"]: sum(parameter.numel() for parameter in group["params"])
        for group in optimizer.param_groups
    }
    assert all(counts[name] > 0 for name in ("optical", "router", "recombiner", "head"))


def _save_pair(
    directories: dict[str, Path],
    split: str,
    sample_id: str,
) -> None:
    image = Image.new("RGB", (12, 8), color=(90, 100, 110))
    mask = Image.new("L", (12, 8), color=0)
    key = "train" if split == "train" else "test"
    image.save(directories[f"{key}_images"] / f"{sample_id}.jpg")
    mask.save(
        directories[f"{key}_masks"] / f"{sample_id}_Segmentation.png"
    )


def _dataset_settings(root: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=root,
        output_dir=output_dir,
        auto_download=False,
        expected_train_samples=2,
        expected_test_samples=1,
        train_limit=None,
        test_limit=None,
        random_seed=42,
        image_size=224,
        train_image_url="https://example/train.zip",
        train_mask_url="https://example/train-mask.zip",
        test_image_url="https://example/test.zip",
        test_mask_url="https://example/test-mask.zip",
    )
