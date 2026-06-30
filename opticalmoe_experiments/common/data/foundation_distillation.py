import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .datasets import DataBundle, _labels_for_dataset, create_dataloaders
from .loader_utils import dataloader_kwargs


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
DINOV2_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV2_IMAGE_STD = (0.229, 0.224, 0.225)


class IndexedGrayDataset(Dataset):
    """Expose deterministic split-local indices alongside grayscale samples."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image = torch.as_tensor(image, dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            image = image.mean(dim=0, keepdim=True)
        return image, int(label), int(index)


class CachedTeacherFeatureDataset(Dataset):
    """Join an indexed student dataset with an offline teacher feature cache."""

    def __init__(self, dataset: IndexedGrayDataset, cache_payload: Mapping) -> None:
        self.dataset = dataset
        self.features = torch.as_tensor(cache_payload["features"], dtype=torch.float32).cpu()
        self.labels = torch.as_tensor(cache_payload["labels"], dtype=torch.long).cpu()
        indices = torch.as_tensor(cache_payload["indices"], dtype=torch.long).cpu()
        if len(indices) != len(dataset) or len(self.features) != len(dataset) or len(self.labels) != len(dataset):
            raise ValueError("Teacher cache length does not match the current dataset split.")
        if len(torch.unique(indices)) != len(indices):
            raise ValueError("Teacher cache contains duplicate sample indices.")
        self.position_by_index = {int(value): position for position, value in enumerate(indices.tolist())}
        expected = set(range(len(dataset)))
        if set(self.position_by_index) != expected:
            raise ValueError("Teacher cache indices do not match the deterministic split-local indices.")
        try:
            expected_labels = _labels_for_dataset(dataset.dataset)
        except ValueError:
            if hasattr(dataset.dataset, "tensors") and len(dataset.dataset.tensors) >= 2:
                expected_labels = torch.as_tensor(dataset.dataset.tensors[1], dtype=torch.long)
            else:
                expected_labels = torch.as_tensor([dataset[index][1] for index in range(len(dataset))], dtype=torch.long)
        ordered_cache_labels = torch.empty_like(self.labels)
        for sample_index, position in self.position_by_index.items():
            ordered_cache_labels[sample_index] = self.labels[position]
        if not torch.equal(expected_labels.cpu(), ordered_cache_labels.cpu()):
            raise ValueError("Teacher cache labels do not match the current dataset split.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label, sample_index = self.dataset[index]
        position = self.position_by_index[int(sample_index)]
        cached_label = int(self.labels[position].item())
        if int(label) != cached_label:
            raise ValueError(
                f"Teacher cache label mismatch at split index {sample_index}: dataset={label}, cache={cached_label}."
            )
        return image, int(label), self.features[position], int(sample_index)


@dataclass
class DistillationDatasetBundle:
    train_dataset: IndexedGrayDataset
    val_dataset: IndexedGrayDataset
    test_dataset: IndexedGrayDataset
    num_classes: int
    class_names: List[str]


@dataclass
class CachedDistillationBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: List[str]
    teacher_feature_dim: int
    teacher_metadata: Dict


def create_distillation_datasets(dataset_cfg: Dict, seed: int = 7) -> DistillationDatasetBundle:
    if dataset_cfg.get("grayscale", True) is not True:
        raise ValueError("foundation_distillation v1 requires dataset.grayscale=true for both teacher and student.")
    cfg = dict(dataset_cfg)
    cfg["grayscale"] = True
    bundle: DataBundle = create_dataloaders(cfg, seed=seed)
    return DistillationDatasetBundle(
        train_dataset=IndexedGrayDataset(bundle.train_loader.dataset),
        val_dataset=IndexedGrayDataset(bundle.val_loader.dataset),
        test_dataset=IndexedGrayDataset(bundle.test_loader.dataset),
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
    )


def teacher_input_from_student_gray(
    images: torch.Tensor,
    teacher_cfg: Optional[Mapping] = None,
    image_size: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Build teacher input from exactly the same grayscale student information."""
    if isinstance(teacher_cfg, (int, float)):
        image_size = int(teacher_cfg)
        teacher_cfg = None
    cfg = dict(teacher_cfg or {})
    teacher_type = str(cfg.get("type", cfg.get("teacher_type", "clip_image_encoder"))).lower()
    if image_size is None:
        image_size = int(cfg.get("teacher_image_size", 224))
    if teacher_type == "dinov2_image_encoder":
        default_mean, default_std = DINOV2_IMAGE_MEAN, DINOV2_IMAGE_STD
    else:
        default_mean, default_std = CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    image_mean = cfg.get("teacher_image_mean", default_mean)
    image_std = cfg.get("teacher_image_std", default_std)
    images = torch.as_tensor(images, dtype=torch.float32)
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.shape[1] != 1:
        images = images.mean(dim=1, keepdim=True)
    images = F.interpolate(images, size=(int(image_size), int(image_size)), mode="bicubic", align_corners=False)
    images = images.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    if normalize:
        mean = images.new_tensor(image_mean).view(1, 3, 1, 1)
        std = images.new_tensor(image_std).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def dataset_config_hash(dataset_cfg: Dict, teacher_cfg: Dict, seed: int) -> str:
    split_keys = (
        "name",
        "root",
        "input_size",
        "grayscale",
        "val_split",
        "download",
        "smoke_test",
        "smoke_train_size",
        "smoke_test_size",
        "sampling_protocol",
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
    )
    relevant = {
        "dataset": {key: dataset_cfg.get(key) for key in split_keys},
        "teacher": {
            "type": teacher_cfg.get("type"),
            "model_name": teacher_cfg.get("model_name"),
            "input_mode": teacher_cfg.get("input_mode"),
        },
        "seed": int(seed),
    }
    if teacher_cfg.get("type") == "dinov2_image_encoder":
        relevant["teacher"].update(
            {
                "backend": teacher_cfg.get("backend", "transformers"),
                "feature_type": teacher_cfg.get("feature_type", "cls"),
            }
        )
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cache_payloads(cache_dir: Path) -> Dict[str, Dict]:
    cache_dir = Path(cache_dir)
    payloads = {}
    for split in ("train", "val", "test"):
        path = cache_dir / f"{split}_features.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Teacher cache is missing {path}. Run build_teacher_feature_cache.py before training."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("split") != split:
            raise ValueError(f"Teacher cache {path} declares split={payload.get('split')!r}, expected {split!r}.")
        payloads[split] = payload
    return payloads


def validate_cache(
    cache_dir: Path,
    datasets: DistillationDatasetBundle,
    dataset_cfg: Dict,
    teacher_cfg: Dict,
    seed: int,
    require_metadata_match: bool = True,
) -> Dict:
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Teacher cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_lengths = {
        "num_train": len(datasets.train_dataset),
        "num_val": len(datasets.val_dataset),
        "num_test": len(datasets.test_dataset),
    }
    for key, expected in expected_lengths.items():
        if int(metadata.get(key, -1)) != int(expected):
            raise ValueError(f"Teacher cache {key}={metadata.get(key)} does not match current split size {expected}.")
    if require_metadata_match:
        expected_hash = dataset_config_hash(dataset_cfg, teacher_cfg, seed)
        if metadata.get("config_hash_or_summary") != expected_hash:
            raise ValueError("Teacher cache metadata does not match the current dataset/teacher configuration.")
    return metadata


def create_cached_distillation_loaders(
    dataset_cfg: Dict,
    teacher_cfg: Dict,
    cache_cfg: Dict,
    seed: int = 7,
) -> CachedDistillationBundle:
    datasets = create_distillation_datasets(dataset_cfg, seed=seed)
    cache_dir = Path(cache_cfg["cache_dir"])
    metadata = validate_cache(
        cache_dir,
        datasets,
        dataset_cfg,
        teacher_cfg,
        seed,
        require_metadata_match=bool(cache_cfg.get("require_metadata_match", True)),
    )
    payloads = load_cache_payloads(cache_dir)
    train_dataset = CachedTeacherFeatureDataset(datasets.train_dataset, payloads["train"])
    val_dataset = CachedTeacherFeatureDataset(datasets.val_dataset, payloads["val"])
    test_dataset = CachedTeacherFeatureDataset(datasets.test_dataset, payloads["test"])
    return CachedDistillationBundle(
        train_loader=DataLoader(train_dataset, **dataloader_kwargs(dataset_cfg, shuffle=True, seed=seed + 300_000)),
        val_loader=DataLoader(val_dataset, **dataloader_kwargs(dataset_cfg, shuffle=False)),
        test_loader=DataLoader(test_dataset, **dataloader_kwargs(dataset_cfg, shuffle=False)),
        num_classes=datasets.num_classes,
        class_names=datasets.class_names,
        teacher_feature_dim=int(metadata["teacher_feature_dim"]),
        teacher_metadata=dict(metadata),
    )
