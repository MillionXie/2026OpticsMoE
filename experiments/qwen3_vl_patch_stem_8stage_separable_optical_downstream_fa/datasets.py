from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.prepare_caltech101_retrieval_subset import (
    BACKGROUND_CLASS,
    Caltech101RetrievalDataset,
    Caltech101Sample,
    _discover_classes as discover_caltech_classes,
    _find_image_root as find_caltech_image_root,
)
from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.datasets import (
    DatasetBundle as SourceISICBundle,
    ISICRecord,
    ISICSegmentationDataset,
    prepare_isic2016 as prepare_validated_isic2016,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    DatasetBundle as SourceLSPBundle,
    LSPPoseDataset,
    PoseRecord,
    prepare_lsp as prepare_validated_lsp,
)


TaskName = Literal["caltech101", "isic2016", "lsp"]
SplitName = Literal["train", "val", "test"]

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ISIC_URLS = {
    "train_images": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Training_Data.zip"
    ),
    "train_masks": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
    ),
    "test_images": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Test_Data.zip"
    ),
    "test_masks": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
    ),
}

LSP_URLS = (
    "hf://LiuRunky/Leeds_Sports_Pose",
    "https://sam.johnson.io/research/lsp_dataset.zip",
    "http://sam.johnson.io/research/lsp_dataset.zip",
)
LSPET_URLS = (
    "https://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip",
    "https://sam.johnson.io/research/lspet_dataset.zip",
    "http://sam.johnson.io/research/lspet_dataset.zip",
)


@dataclass(frozen=True)
class DownstreamDataConfig:
    task: TaskName
    data_root: Path
    output_dir: Path | None = None
    image_size: int = 224
    random_seed: int = 2026
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    augmentation_enabled: bool = True
    auto_download: bool = False

    # Caltech-101 is deliberately class balanced for train/validation. All
    # remaining images are held out for the final classification test.
    caltech_train_per_class: int = 25
    caltech_val_per_class: int = 5
    caltech_crop_scale_min: float = 0.75
    caltech_brightness_jitter: float = 0.20
    caltech_contrast_jitter: float = 0.20
    caltech_rotation_degrees: float = 12.0
    caltech_horizontal_flip_probability: float = 0.5

    # ISIC augmentation is paired: every geometric operation is applied to
    # the RGB image and binary mask together.
    isic_val_fraction: float = 0.20
    isic_crop_scale_min: float = 0.92
    isic_horizontal_flip_probability: float = 0.5
    isic_vertical_flip_probability: float = 0.5
    isic_brightness_jitter: float = 0.08
    isic_contrast_jitter: float = 0.08
    isic_rotation_degrees: float = 10.0

    # The validated LSP loader defines the official LSP/LSPET protocol. P12
    # only adds a deterministic source-stratified validation split.
    lsp_val_fraction: float = 0.10
    lspet_expected_count: int = 9428
    lsp_visibility_policy: str = "coordinates_in_image"
    lsp_crop_margin: float = 1.25
    lsp_scale_jitter: float = 0.15
    lsp_center_jitter: float = 0.05
    lsp_horizontal_flip_probability: float = 0.5
    lsp_brightness_jitter: float = 0.10
    lsp_contrast_jitter: float = 0.10
    heatmap_size: int = 56
    heatmap_sigma: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser().resolve())
        if self.output_dir is not None:
            object.__setattr__(
                self, "output_dir", Path(self.output_dir).expanduser().resolve()
            )
        if self.task not in {"caltech101", "isic2016", "lsp"}:
            raise ValueError(f"Unsupported downstream task: {self.task}")
        if self.image_size != 224:
            raise ValueError("P12 uses the frozen Qwen 224x224 patch stem")
        for name in ("train_limit", "val_limit", "test_limit"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive or null")
        for name in ("train_batch_size", "eval_batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.caltech_train_per_class != 25 or self.caltech_val_per_class != 5:
            raise ValueError("The formal P12 Caltech split is locked to 25 train / 5 val")
        if not 0.0 < self.isic_val_fraction < 1.0:
            raise ValueError("isic_val_fraction must be in (0,1)")
        if not 0.0 < self.lsp_val_fraction < 1.0:
            raise ValueError("lsp_val_fraction must be in (0,1)")


@dataclass(frozen=True)
class DownstreamBundle:
    task: TaskName
    train: tuple[Any, ...]
    val: tuple[Any, ...]
    test: tuple[Any, ...]
    metadata: dict[str, Any]
    manifest_sha256: str

    @property
    def train_records(self) -> tuple[Any, ...]:
        return self.train

    @property
    def val_records(self) -> tuple[Any, ...]:
        return self.val

    @property
    def test_records(self) -> tuple[Any, ...]:
        return self.test

    def records(self, split: SplitName) -> tuple[Any, ...]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return getattr(self, split)


def pil_to_clip_tensor(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Convert an already geometrically transformed RGB image for P11."""

    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (int(image_size), int(image_size)):
        image = image.resize(
            (int(image_size), int(image_size)), Image.Resampling.BICUBIC
        )
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(CLIP_STD).view(3, 1, 1)
    return (tensor - mean) / std


def prepare_caltech101(config: DownstreamDataConfig) -> DownstreamBundle:
    if config.task != "caltech101":
        raise ValueError("prepare_caltech101 requires task='caltech101'")
    image_root = find_caltech_image_root(config.data_root)
    if image_root is None:
        raise FileNotFoundError(
            "Caltech-101 101_ObjectCategories was not found below "
            f"{config.data_root}; the P12 runner does not download datasets implicitly"
        )
    available = discover_caltech_classes(image_root)
    class_names = tuple(sorted(name for name in available if name != BACKGROUND_CLASS))
    if len(class_names) != 101:
        raise RuntimeError(
            f"Expected 101 object categories after excluding {BACKGROUND_CLASS}, "
            f"found {len(class_names)}"
        )

    splits: dict[str, list[Caltech101Sample]] = {
        "train": [], "val": [], "test": []
    }
    for class_index, class_name in enumerate(class_names):
        paths = sorted(
            available[class_name],
            key=lambda path: _stable_key(
                "p12_caltech101_25_5_v1",
                config.random_seed,
                class_name,
                str(path.relative_to(image_root)),
            ),
        )
        required = config.caltech_train_per_class + config.caltech_val_per_class + 1
        if len(paths) < required:
            raise RuntimeError(
                f"Caltech class {class_name!r} has {len(paths)} images; "
                f"at least {required} are required"
            )
        cut_train = config.caltech_train_per_class
        cut_val = cut_train + config.caltech_val_per_class
        assignments = {
            "train": paths[:cut_train],
            "val": paths[cut_train:cut_val],
            "test": paths[cut_val:],
        }
        for split, selected in assignments.items():
            splits[split].extend(
                Caltech101Sample(
                    sample_id=(
                        f"caltech101:{class_name}:"
                        f"{path.relative_to(image_root).as_posix()}"
                    ),
                    image_path=path.resolve(),
                    class_id=class_index + 1,
                    class_name=class_name,
                    class_index=class_index,
                    split=split,
                    source_split=f"p12_{split}",
                    is_gallery=False,
                )
                for path in selected
            )

    full_counts = {name: len(values) for name, values in splits.items()}
    selected = _limit_splits(splits, config)
    metadata = {
        "dataset": "Caltech-101",
        "task": "101-class object classification",
        "data_root": str(config.data_root),
        "image_root": str(image_root),
        "class_names": list(class_names),
        "num_classes": 101,
        "excluded_category": BACKGROUND_CLASS,
        "split_policy": (
            "per-class SHA256 ordering; exactly 25 train, 5 validation, "
            "all remaining images test"
        ),
        "full_counts": full_counts,
        "random_seed": config.random_seed,
    }
    return _finish_bundle("caltech101", selected, metadata, config)


def prepare_isic2016(config: DownstreamDataConfig) -> DownstreamBundle:
    if config.task != "isic2016":
        raise ValueError("prepare_isic2016 requires task='isic2016'")
    source_settings = _isic_settings(config, output_dir=config.output_dir)
    source: SourceISICBundle = prepare_validated_isic2016(
        source_settings, persist=False
    )
    if len(source.train_records) != 900 or len(source.test_records) != 379:
        raise RuntimeError("P12 requires the complete official ISIC 900/379 split")

    ranked = sorted(
        source.train_records,
        key=lambda record: _stable_key(
            "p12_isic2016_train_val_v1", config.random_seed, record.sample_id
        ),
    )
    val_count = int(round(len(ranked) * config.isic_val_fraction))
    val_ids = {record.sample_id for record in ranked[:val_count]}
    train = tuple(
        replace(record, split="train")
        for record in source.train_records
        if record.sample_id not in val_ids
    )
    val = tuple(
        replace(record, split="val")
        for record in source.train_records
        if record.sample_id in val_ids
    )
    test = tuple(replace(record, split="test") for record in source.test_records)
    splits = _limit_splits({"train": train, "val": val, "test": test}, config)
    metadata = {
        "dataset": "ISBI2016_ISIC_Task1",
        "task": "binary skin-lesion segmentation",
        "data_root": str(config.data_root),
        "split_policy": (
            "official 900 training pairs split by seeded SHA256 into 80% train / "
            "20% validation; official 379 test pairs remain sealed"
        ),
        "official_train_pairs": 900,
        "official_test_pairs": 379,
        "full_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "patient_group_ids_available": False,
        "mask_resize_interpolation": "nearest",
        "random_seed": config.random_seed,
    }
    return _finish_bundle("isic2016", splits, metadata, config)


def prepare_lsp(config: DownstreamDataConfig) -> DownstreamBundle:
    if config.task != "lsp":
        raise ValueError("prepare_lsp requires task='lsp'")
    settings = _lsp_settings(config)
    source: SourceLSPBundle = prepare_validated_lsp(settings, persist=False)
    if len(source.test) != 1000 or any(record.source != "lsp" for record in source.test):
        raise RuntimeError("Validated LSP loader did not return the official final-1000 test")

    train, val = _source_stratified_train_val(
        source.train,
        fraction=config.lsp_val_fraction,
        seed=config.random_seed,
    )
    train = tuple(replace(record, split="train") for record in train)
    val = tuple(replace(record, split="val") for record in val)
    test = tuple(replace(record, split="test") for record in source.test)
    splits = _limit_splits({"train": train, "val": val, "test": test}, config)
    metadata = {
        **source.metadata,
        "task": "14-joint human-pose heatmap localization",
        "data_root": str(config.data_root),
        "split_policy_p12": (
            "validated HR-LSPET + first-1000 LSP training pool; seeded "
            "source-stratified 10% validation; official final-1000 LSP test sealed"
        ),
        "full_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "random_seed": config.random_seed,
    }
    return _finish_bundle("lsp", splits, metadata, config)


def prepare_bundle(config: DownstreamDataConfig) -> DownstreamBundle:
    if config.task == "caltech101":
        return prepare_caltech101(config)
    if config.task == "isic2016":
        return prepare_isic2016(config)
    return prepare_lsp(config)


class P12Caltech101Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Caltech101Sample],
        config: DownstreamDataConfig,
        *,
        training: bool,
    ) -> None:
        self.source = Caltech101RetrievalDataset(
            records,
            config.image_size,
            augment=training and config.augmentation_enabled,
            crop_scale_min=config.caltech_crop_scale_min,
            brightness_jitter=config.caltech_brightness_jitter,
            contrast_jitter=config.caltech_contrast_jitter,
            rotation_degrees=config.caltech_rotation_degrees,
            horizontal_flip_probability=config.caltech_horizontal_flip_probability,
        )
        self.image_size = config.image_size

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self.source[index]
        sample: Caltech101Sample = value["sample"]
        return {
            "task": "caltech101",
            "image": pil_to_clip_tensor(value["image"], self.image_size),
            "label": torch.tensor(sample.class_index, dtype=torch.long),
            "sample_id": sample.sample_id,
            "split": sample.split,
            "image_path": str(sample.image_path),
            "class_name": sample.class_name,
            "dataset_index": int(value["dataset_index"]),
        }


class P12ISIC2016Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[ISICRecord],
        config: DownstreamDataConfig,
        *,
        training: bool,
    ) -> None:
        self.source = ISICSegmentationDataset(
            tuple(records), _isic_settings(config), training=training
        )
        self.image_size = config.image_size

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self.source[index]
        return {
            "task": "isic2016",
            "image": pil_to_clip_tensor(value["image"], self.image_size),
            "mask": value["mask"].float(),
            "sample_id": value["sample_id"],
            "split": value["split"],
            "image_path": value["image_path"],
            "mask_path": value["mask_path"],
            "geometry_transform": value["geometry_transform"],
            "dataset_index": int(value["sample_index"]),
        }


class P12LSPDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[PoseRecord],
        config: DownstreamDataConfig,
        *,
        training: bool,
    ) -> None:
        self.records = tuple(records)
        self.source = LSPPoseDataset(
            list(self.records), _lsp_settings(config), training=training
        )
        self.image_size = config.image_size

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self.source[index]
        return {
            "task": "lsp",
            "image": pil_to_clip_tensor(value["image"], self.image_size),
            "heatmaps": value["heatmaps"].float(),
            "keypoints": value["keypoints"].float(),
            "visible": value["visible"].bool(),
            "torso_scale": value["torso_scale"].float(),
            "head_scale": value["head_scale"].float(),
            "sample_id": value["sample_id"],
            "split": self.records[index].split,
            "source": value["source"],
            "image_path": value["image_path"],
            "crop_box": value["crop_box"],
            "flipped": bool(value["flipped"]),
            "dataset_index": int(value["index"]),
        }


def build_datasets(
    bundle: DownstreamBundle,
    config: DownstreamDataConfig,
) -> dict[SplitName, Dataset[dict[str, Any]]]:
    if bundle.task != config.task:
        raise ValueError(f"Bundle task {bundle.task} != config task {config.task}")
    cls: type[Dataset]
    if config.task == "caltech101":
        cls = P12Caltech101Dataset
    elif config.task == "isic2016":
        cls = P12ISIC2016Dataset
    else:
        cls = P12LSPDataset
    return {
        split: cls(bundle.records(split), config, training=split == "train")
        for split in ("train", "val", "test")
    }


def build_loaders(
    bundle: DownstreamBundle,
    config: DownstreamDataConfig,
) -> dict[SplitName, DataLoader]:
    datasets = build_datasets(bundle, config)
    loaders: dict[SplitName, DataLoader] = {}
    for offset, split in enumerate(("train", "val", "test")):
        generator = torch.Generator().manual_seed(config.random_seed + offset)
        kwargs: dict[str, Any] = {
            "batch_size": (
                config.train_batch_size if split == "train" else config.eval_batch_size
            ),
            "shuffle": split == "train",
            "drop_last": False,
            "num_workers": config.num_workers,
            "pin_memory": bool(config.pin_memory),
            "persistent_workers": bool(config.persistent_workers and config.num_workers > 0),
            "worker_init_fn": _seed_worker,
            "generator": generator,
        }
        if config.num_workers > 0:
            kwargs["prefetch_factor"] = int(config.prefetch_factor)
        loaders[split] = DataLoader(datasets[split], **kwargs)
    return loaders


def _isic_settings(
    config: DownstreamDataConfig,
    *,
    output_dir: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=config.data_root,
        output_dir=output_dir or config.output_dir or config.data_root / ".p12_unused",
        auto_download=config.auto_download,
        remove_archives_after_extract=False,
        expected_train_samples=900,
        expected_test_samples=379,
        train_limit=None,
        test_limit=None,
        random_seed=config.random_seed,
        image_size=config.image_size,
        augmentation_enabled=config.augmentation_enabled,
        crop_scale_min=config.isic_crop_scale_min,
        horizontal_flip_probability=config.isic_horizontal_flip_probability,
        vertical_flip_probability=config.isic_vertical_flip_probability,
        brightness_jitter=config.isic_brightness_jitter,
        contrast_jitter=config.isic_contrast_jitter,
        rotation_degrees=config.isic_rotation_degrees,
        student_batch_size=config.train_batch_size,
        inference_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        train_image_url=ISIC_URLS["train_images"],
        train_mask_url=ISIC_URLS["train_masks"],
        test_image_url=ISIC_URLS["test_images"],
        test_mask_url=ISIC_URLS["test_masks"],
    )


def _lsp_settings(config: DownstreamDataConfig) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=config.data_root,
        output_dir=config.output_dir or config.data_root / ".p12_unused",
        download=config.auto_download,
        lsp_urls=LSP_URLS,
        lspet_urls=LSPET_URLS,
        lspet_expected_count=config.lspet_expected_count,
        visibility_policy=config.lsp_visibility_policy,
        strict_dataset_counts=True,
        train_limit=None,
        test_limit=None,
        random_seed=config.random_seed,
        image_size=config.image_size,
        heatmap_size=config.heatmap_size,
        heatmap_sigma=config.heatmap_sigma,
        augmentation_enabled=config.augmentation_enabled,
        crop_margin=config.lsp_crop_margin,
        crop_scale_jitter=config.lsp_scale_jitter,
        crop_center_jitter=config.lsp_center_jitter,
        horizontal_flip_probability=config.lsp_horizontal_flip_probability,
        brightness_jitter=config.lsp_brightness_jitter,
        contrast_jitter=config.lsp_contrast_jitter,
    )


def _source_stratified_train_val(
    records: Sequence[PoseRecord],
    *,
    fraction: float,
    seed: int,
) -> tuple[tuple[PoseRecord, ...], tuple[PoseRecord, ...]]:
    by_source: dict[str, list[PoseRecord]] = {}
    for record in records:
        by_source.setdefault(record.source, []).append(record)
    val_ids: set[str] = set()
    for source, values in sorted(by_source.items()):
        ranked = sorted(
            values,
            key=lambda record: _stable_key(
                "p12_lsp_train_val_v1", seed, source, record.sample_id
            ),
        )
        count = max(1, int(round(len(ranked) * fraction)))
        val_ids.update(record.sample_id for record in ranked[:count])
    train = tuple(record for record in records if record.sample_id not in val_ids)
    val = tuple(record for record in records if record.sample_id in val_ids)
    return train, val


def _limit_splits(
    splits: Mapping[str, Sequence[Any]],
    config: DownstreamDataConfig,
) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for offset, split in enumerate(("train", "val", "test")):
        records = tuple(splits[split])
        limit = getattr(config, f"{split}_limit")
        if limit is not None and int(limit) < len(records):
            records = tuple(
                sorted(
                    records,
                    key=lambda record: _stable_key(
                        "p12_smoke_limit_v1",
                        config.random_seed + offset,
                        _record_id(record),
                    ),
                )[: int(limit)]
            )
        result[split] = records
    return result


def _finish_bundle(
    task: TaskName,
    splits: Mapping[str, Sequence[Any]],
    metadata: dict[str, Any],
    config: DownstreamDataConfig,
) -> DownstreamBundle:
    normalized = {name: tuple(splits[name]) for name in ("train", "val", "test")}
    _assert_no_leakage(normalized)
    rows = _manifest_rows(task, normalized)
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = {
        **metadata,
        "counts": {name: len(values) for name, values in normalized.items()},
        "manifest_sha256": digest,
        "image_size": config.image_size,
        "clip_mean": list(CLIP_MEAN),
        "clip_std": list(CLIP_STD),
        "limits": {
            "train": config.train_limit,
            "val": config.val_limit,
            "test": config.test_limit,
        },
        "split_leakage": 0,
    }
    if config.output_dir is not None:
        _write_manifest(config.output_dir, task, rows, metadata)
    return DownstreamBundle(
        task=task,
        train=normalized["train"],
        val=normalized["val"],
        test=normalized["test"],
        metadata=metadata,
        manifest_sha256=digest,
    )


def _assert_no_leakage(splits: Mapping[str, Sequence[Any]]) -> None:
    ids = {name: {_record_id(record) for record in values} for name, values in splits.items()}
    paths = {
        name: {str(Path(record.image_path).resolve()) for record in values}
        for name, values in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        id_overlap = ids[left] & ids[right]
        path_overlap = paths[left] & paths[right]
        if id_overlap or path_overlap:
            raise RuntimeError(
                f"Dataset leakage between {left}/{right}: "
                f"ids={sorted(id_overlap)[:3]}, paths={sorted(path_overlap)[:3]}"
            )


def _manifest_rows(
    task: TaskName,
    splits: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for record in splits[split]:
            row: dict[str, Any] = {
                "task": task,
                "split": split,
                "sample_id": _record_id(record),
                "image_path": str(Path(record.image_path).resolve()),
                "target_path": "",
                "label": "",
                "class_name": "",
                "source": str(getattr(record, "source", "")),
            }
            if isinstance(record, Caltech101Sample):
                row["label"] = int(record.class_index)
                row["class_name"] = record.class_name
            elif isinstance(record, ISICRecord):
                row["target_path"] = str(Path(record.mask_path).resolve())
            rows.append(row)
    return sorted(rows, key=lambda row: (row["split"], row["sample_id"]))


def _write_manifest(
    output_dir: Path,
    task: TaskName,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    root = Path(output_dir) / "manifests"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{task}_splits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / f"{task}_splits.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_id(record: Any) -> str:
    value = getattr(record, "sample_id", None)
    if value is None:
        raise TypeError(f"Dataset record has no sample_id: {type(record).__name__}")
    return str(value)


def _stable_key(namespace: str, seed: int, *parts: str) -> str:
    payload = "|".join((namespace, str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "DownstreamBundle",
    "DownstreamDataConfig",
    "P12Caltech101Dataset",
    "P12ISIC2016Dataset",
    "P12LSPDataset",
    "build_datasets",
    "build_loaders",
    "pil_to_clip_tensor",
    "prepare_bundle",
    "prepare_caltech101",
    "prepare_isic2016",
    "prepare_lsp",
]
