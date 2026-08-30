from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa import (
    datasets as p12data,
)


def _config(task: p12data.TaskName, root: Path, **kwargs: object) -> p12data.DownstreamDataConfig:
    return p12data.DownstreamDataConfig(task=task, data_root=root, **kwargs)


def test_clip_normalization_has_the_exact_p11_tensor_contract() -> None:
    image = Image.new("RGB", (224, 224), (255, 128, 0))
    value = p12data.pil_to_clip_tensor(image)
    assert value.shape == (3, 224, 224)
    assert value.dtype == torch.float32
    raw = torch.tensor([1.0, 128.0 / 255.0, 0.0])
    expected = (raw - torch.tensor(p12data.CLIP_MEAN)) / torch.tensor(
        p12data.CLIP_STD
    )
    assert torch.allclose(value[:, 17, 31], expected, atol=1.0e-6)


def test_caltech101_split_is_25_train_5_val_rest_test_per_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / "101_ObjectCategories"
    available = {
        f"class_{index:03d}": [
            image_root / f"class_{index:03d}" / f"image_{sample:04d}.jpg"
            for sample in range(31)
        ]
        for index in range(101)
    }
    monkeypatch.setattr(p12data, "find_caltech_image_root", lambda _: image_root)
    monkeypatch.setattr(p12data, "discover_caltech_classes", lambda _: available)
    config = _config(
        "caltech101", tmp_path, output_dir=tmp_path / "run", random_seed=2026
    )

    first = p12data.prepare_caltech101(config)
    second = p12data.prepare_caltech101(config)

    assert first.metadata["full_counts"] == {
        "train": 2525,
        "val": 505,
        "test": 101,
    }
    assert Counter(record.class_name for record in first.train) == {
        name: 25 for name in available
    }
    assert Counter(record.class_name for record in first.val) == {
        name: 5 for name in available
    }
    assert Counter(record.class_name for record in first.test) == {
        name: 1 for name in available
    }
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.train == second.train
    assert (tmp_path / "run/manifests/caltech101_splits.csv").is_file()
    assert (tmp_path / "run/manifests/caltech101_splits.json").is_file()


def test_smoke_limits_are_applied_after_the_formal_caltech_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / "objects"
    available = {
        f"c{index:03d}": [
            image_root / f"c{index:03d}" / f"{sample:03d}.jpg"
            for sample in range(32)
        ]
        for index in range(101)
    }
    monkeypatch.setattr(p12data, "find_caltech_image_root", lambda _: image_root)
    monkeypatch.setattr(p12data, "discover_caltech_classes", lambda _: available)
    bundle = p12data.prepare_caltech101(
        _config(
            "caltech101",
            tmp_path,
            train_limit=7,
            val_limit=5,
            test_limit=3,
        )
    )
    assert (len(bundle.train), len(bundle.val), len(bundle.test)) == (7, 5, 3)
    assert bundle.metadata["full_counts"] == {
        "train": 2525,
        "val": 505,
        "test": 202,
    }


def _isic_record(root: Path, index: int, split: str) -> p12data.ISICRecord:
    return p12data.ISICRecord(
        sample_index=index,
        sample_id=f"{split}_{index:04d}",
        split=split,
        image_path=root / split / f"image_{index:04d}.jpg",
        mask_path=root / split / f"image_{index:04d}_Segmentation.png",
    )


def test_isic_uses_seeded_720_180_and_preserves_official_379_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = p12data.SourceISICBundle(
        train_records=tuple(_isic_record(tmp_path, index, "official_train") for index in range(900)),
        test_records=tuple(_isic_record(tmp_path, index, "official_test") for index in range(379)),
        metadata={"official_split": True},
    )
    monkeypatch.setattr(
        p12data, "prepare_validated_isic2016", lambda _settings, persist=False: source
    )
    config = _config("isic2016", tmp_path, random_seed=2026)
    first = p12data.prepare_isic2016(config)
    second = p12data.prepare_isic2016(config)

    assert (len(first.train), len(first.val), len(first.test)) == (720, 180, 379)
    assert first.manifest_sha256 == second.manifest_sha256
    assert {record.sample_id for record in first.test} == {
        record.sample_id for record in source.test_records
    }
    assert {record.sample_id for record in first.train}.isdisjoint(
        record.sample_id for record in first.val
    )


def test_isic_geometric_augmentation_is_synchronized_with_binary_mask(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "image_Segmentation.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16, 0] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:, :16] = 255
    Image.fromarray(image).save(image_path, quality=100, subsampling=0)
    Image.fromarray(mask).save(mask_path)
    record = p12data.ISICRecord(0, "sample", "train", image_path, mask_path)
    config = _config(
        "isic2016",
        tmp_path,
        isic_crop_scale_min=1.0,
        isic_horizontal_flip_probability=1.0,
        isic_vertical_flip_probability=0.0,
        isic_brightness_jitter=0.0,
        isic_contrast_jitter=0.0,
        isic_rotation_degrees=0.0,
    )
    item = p12data.P12ISIC2016Dataset([record], config, training=True)[0]
    red = (
        item["image"][0] * p12data.CLIP_STD[0] + p12data.CLIP_MEAN[0]
    )
    target = item["mask"][0]

    assert item["image"].shape == (3, 224, 224)
    assert item["mask"].shape == (1, 224, 224)
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
    assert red[:, 140:].mean() > red[:, :84].mean() + 0.75
    assert target[:, 140:].mean() > target[:, :84].mean() + 0.95
    assert item["geometry_transform"]["horizontal_flip"] is True


def _pose_record(root: Path, source: str, index: int, split: str) -> p12data.PoseRecord:
    points = np.stack(
        (
            np.linspace(12.0, 48.0, 14, dtype=np.float32),
            np.linspace(10.0, 52.0, 14, dtype=np.float32),
        ),
        axis=1,
    )
    return p12data.PoseRecord(
        sample_id=f"{source}_{index:05d}",
        source=source,
        split=split,
        source_index=index,
        image_path=root / source / f"im{index:05d}.jpg",
        keypoints=points,
        raw_visibility=np.zeros(14, dtype=np.float32),
    )


def test_lsp_reuses_official_test_and_adds_seeded_source_stratified_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = p12data.SourceLSPBundle(
        train=[
            *(_pose_record(tmp_path, "lspet", index, "train") for index in range(100)),
            *(_pose_record(tmp_path, "lsp", index, "train") for index in range(20)),
        ],
        test=[
            _pose_record(tmp_path, "lsp", 1000 + index, "test")
            for index in range(1000)
        ],
        metadata={"protocol": "validated_standard_protocol"},
    )
    monkeypatch.setattr(
        p12data, "prepare_validated_lsp", lambda _settings, persist=False: source
    )
    config = _config("lsp", tmp_path, lsp_val_fraction=0.10, random_seed=2026)
    first = p12data.prepare_lsp(config)
    second = p12data.prepare_lsp(config)

    assert (len(first.train), len(first.val), len(first.test)) == (108, 12, 1000)
    assert Counter(record.source for record in first.val) == {"lspet": 10, "lsp": 2}
    assert first.manifest_sha256 == second.manifest_sha256
    assert {record.sample_id for record in first.test} == {
        record.sample_id for record in source.test
    }


def test_lsp_dataset_returns_heatmaps_keypoints_visibility_and_scales(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "lsp" / "im00001.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), (128, 96, 64)).save(image_path)
    record = _pose_record(tmp_path, "lsp", 1, "val")
    config = _config("lsp", tmp_path, augmentation_enabled=False)
    item = p12data.P12LSPDataset([record], config, training=False)[0]

    assert item["image"].shape == (3, 224, 224)
    assert item["heatmaps"].shape == (14, 56, 56)
    assert item["keypoints"].shape == (14, 2)
    assert item["visible"].shape == (14,)
    assert item["visible"].dtype == torch.bool
    assert item["torso_scale"].ndim == 0
    assert item["head_scale"].ndim == 0
    assert item["split"] == "val"
    assert torch.isfinite(item["heatmaps"]).all()


def test_config_rejects_geometry_or_formal_split_changes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="224x224"):
        _config("caltech101", tmp_path, image_size=256)
    with pytest.raises(ValueError, match="25 train / 5 val"):
        _config("caltech101", tmp_path, caltech_train_per_class=20)
