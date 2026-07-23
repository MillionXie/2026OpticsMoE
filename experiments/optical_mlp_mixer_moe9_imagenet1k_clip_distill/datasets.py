from __future__ import annotations

import hashlib
import io
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from .settings import ExperimentSettings


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TRANSFORM_SCHEMA_VERSION = 1


@contextmanager
def _deterministic_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


def view_seed(base_seed: int, sample_index: int, view_index: int) -> int:
    payload = f"optical-mixer-view-v1:{base_seed}:{sample_index}:{view_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


class DeterministicImageTransform:
    """Reproduce the exact same augmented view for CLIP caching and student training."""

    def __init__(self, settings: ExperimentSettings, train: bool) -> None:
        try:
            from torchvision import transforms
        except Exception as error:
            raise RuntimeError("torchvision is required for ImageNet preprocessing") from error
        self.train = bool(train)
        self.image_size = settings.model.image_size
        if train:
            operations: list = [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=settings.clip.random_resized_crop_scale,
                    ratio=settings.clip.random_resized_crop_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(settings.clip.horizontal_flip_probability),
            ]
            if settings.clip.randaugment_enabled:
                operations.append(
                    transforms.RandAugment(
                        num_ops=settings.clip.randaugment_num_ops,
                        magnitude=settings.clip.randaugment_magnitude,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                )
            operations.extend([transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        else:
            resize_size = int(round(self.image_size / 0.875))
            operations = [
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        self.transform = transforms.Compose(operations)

    def __call__(self, image, seed: int) -> torch.Tensor:
        image = image.convert("RGB")
        if not self.train:
            return self.transform(image)
        with _deterministic_rng(seed):
            return self.transform(image)


class ImageNetViewDataset(Dataset):
    """ImageFolder exposed as deterministic (image, view) pairs.

    The logical dataset has ``base_samples * views`` entries. Training samples
    only one view per base image per epoch through :class:`EpochViewSampler`.
    """

    def __init__(
        self,
        directory: Path,
        settings: ExperimentSettings,
        *,
        train: bool,
        limit: int | None,
        expected_classes: list[str] | None = None,
    ) -> None:
        try:
            from torchvision.datasets import ImageFolder
        except Exception as error:
            raise RuntimeError("torchvision.datasets.ImageFolder is required for ImageNet-1K") from error
        if not directory.is_dir():
            raise FileNotFoundError(
                f"ImageNet split directory is missing: {directory}\n"
                "ImageNet-1K cannot be downloaded anonymously by this experiment. "
                "Place the licensed ILSVRC2012 data under data_root/{train,val}/<class>/image.JPEG."
            )
        folder = ImageFolder(str(directory))
        if expected_classes is not None and folder.classes != expected_classes:
            raise RuntimeError("Train and validation ImageFolder class order differs")
        base_indices = list(range(len(folder.samples)))
        if limit is not None:
            if limit <= 0:
                raise ValueError("Dataset limit must be positive")
            base_indices = _stratified_limit(folder.targets, min(limit, len(folder)), settings.dataset.seed)
        self.directory = directory
        self.dataset_source = "imagefolder"
        self.fingerprint: str | None = None
        self.source_split = directory.name
        self.loader = folder.loader
        self.classes = list(folder.classes)
        self.class_to_idx = dict(folder.class_to_idx)
        self.samples = [folder.samples[index] for index in base_indices]
        self.targets = [int(target) for _, target in self.samples]
        self.train = bool(train)
        self.views = settings.clip.views_per_train_image if train else 1
        self.base_seed = int(settings.dataset.seed)
        self.transform = DeterministicImageTransform(settings, train=train)

    @property
    def base_sample_count(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples) * self.views

    def decode_index(self, composite_index: int) -> tuple[int, int]:
        if composite_index < 0 or composite_index >= len(self):
            raise IndexError(composite_index)
        return composite_index // self.views, composite_index % self.views

    def __getitem__(self, composite_index: int) -> dict:
        sample_index, view_index = self.decode_index(int(composite_index))
        path, label = self.samples[sample_index]
        image = self.loader(path)
        seed = view_seed(self.base_seed, sample_index, view_index)
        tensor = self.transform(image, seed)
        return {
            "image": tensor,
            "label": int(label),
            "sample_index": sample_index,
            "view_index": view_index,
            "path": path,
            "view_seed": seed,
        }


class HuggingFaceImageNetViewDataset(Dataset):
    """Deterministic views backed directly by the gated HF ImageNet parquet data.

    No second ImageFolder copy is materialized: the Hugging Face cache owns the
    downloaded image bytes and records are decoded to PIL images on demand.
    """

    def __init__(
        self,
        raw_dataset: Any,
        settings: ExperimentSettings,
        *,
        split: str,
        train: bool,
        limit: int | None,
        expected_classes: list[str] | None = None,
    ) -> None:
        features = getattr(raw_dataset, "features", None)
        label_feature = features.get("label") if features is not None else None
        names = list(getattr(label_feature, "names", []) or [])
        if len(names) != settings.model.num_classes:
            raise RuntimeError(
                "Hugging Face ImageNet label metadata is invalid: expected "
                f"{settings.model.num_classes} ClassLabel names, found {len(names)}. "
                "Refusing to guess the label order."
            )
        if expected_classes is not None and names != expected_classes:
            raise RuntimeError("Hugging Face train/validation label order differs")
        full_targets = [int(value) for value in raw_dataset["label"]]
        base_indices = list(range(len(full_targets)))
        if limit is not None:
            if limit <= 0:
                raise ValueError("Dataset limit must be positive")
            selected = _stratified_limit(
                full_targets,
                min(limit, len(full_targets)),
                settings.dataset.seed,
            )
            base_indices = selected
        self.raw_dataset = raw_dataset
        self.raw_sample_count = len(full_targets)
        self.base_indices = base_indices
        self.targets = [full_targets[index] for index in base_indices]
        self.classes = names
        self.class_to_idx = {name: index for index, name in enumerate(names)}
        self.train = bool(train)
        self.views = settings.clip.views_per_train_image if train else 1
        self.base_seed = int(settings.dataset.seed)
        self.transform = DeterministicImageTransform(settings, train=train)
        self.dataset_source = "huggingface"
        self.dataset_id = settings.dataset.hf_dataset_id
        self.revision = settings.dataset.hf_revision
        self.source_split = str(split)
        self.directory = settings.dataset.root
        self.fingerprint = str(getattr(raw_dataset, "_fingerprint", "unknown"))

    @property
    def base_sample_count(self) -> int:
        return len(self.base_indices)

    def __len__(self) -> int:
        return self.base_sample_count * self.views

    def decode_index(self, composite_index: int) -> tuple[int, int]:
        if composite_index < 0 or composite_index >= len(self):
            raise IndexError(composite_index)
        return composite_index // self.views, composite_index % self.views

    @staticmethod
    def _decode_image(value: Any):
        if hasattr(value, "convert"):
            return value
        if isinstance(value, dict):
            try:
                from PIL import Image
            except Exception as error:
                raise RuntimeError("Pillow is required to decode ImageNet images") from error
            if value.get("bytes") is not None:
                return Image.open(io.BytesIO(value["bytes"]))
            if value.get("path"):
                return Image.open(value["path"])
        raise RuntimeError(
            f"Unsupported Hugging Face ImageNet image value: {type(value).__name__}"
        )

    def __getitem__(self, composite_index: int) -> dict:
        sample_index, view_index = self.decode_index(int(composite_index))
        source_index = self.base_indices[sample_index]
        record = self.raw_dataset[source_index]
        label = int(record["label"])
        if label != self.targets[sample_index]:
            raise RuntimeError(
                f"Hugging Face label changed at source sample {source_index}"
            )
        image = self._decode_image(record["image"])
        filename = str(getattr(image, "filename", "") or "")
        path = filename or (
            f"hf://{self.dataset_id}/{self.source_split}/{source_index:08d}"
        )
        seed = view_seed(self.base_seed, sample_index, view_index)
        tensor = self.transform(image, seed)
        return {
            "image": tensor,
            "label": label,
            "sample_index": sample_index,
            "source_sample_index": source_index,
            "view_index": view_index,
            "path": path,
            "view_seed": seed,
        }


class EpochViewSampler(Sampler[int]):
    """One deterministic cached view per ImageNet image and epoch.

    Shuffling operates on base image indices, while the selected view cycles
    through all cached views. In DDP, the standard equal-length padding rule
    repeats at most ``world_size - 1`` samples so every rank executes the same
    number of optimizer steps; no original image is dropped.
    """

    def __init__(
        self,
        dataset: ImageNetViewDataset,
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid rank/world_size")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        count = self.dataset.base_sample_count
        return math.ceil(count / self.world_size)

    def __iter__(self):
        count = self.dataset.base_sample_count
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(count, generator=generator).tolist()
        else:
            indices = list(range(count))
        total_size = math.ceil(count / self.world_size) * self.world_size
        if total_size > count:
            indices.extend(indices[: total_size - count])
        indices = indices[self.rank::self.world_size]
        views = self.dataset.views
        for sample_index in indices:
            hashed_offset = view_seed(self.seed, sample_index, 0) % views
            view_index = (hashed_offset + self.epoch) % views
            yield sample_index * views + view_index


@dataclass
class ImageNetBundle:
    train: ImageNetViewDataset | HuggingFaceImageNetViewDataset
    validation: ImageNetViewDataset | HuggingFaceImageNetViewDataset
    class_names: list[str]
    folder_classes: list[str]
    digest: str


def load_imagenet(settings: ExperimentSettings) -> ImageNetBundle:
    root = settings.dataset.root
    if settings.dataset.source == "huggingface":
        raw_train = _load_huggingface_split(settings, settings.dataset.train_split)
        train = HuggingFaceImageNetViewDataset(
            raw_train,
            settings,
            split=settings.dataset.train_split,
            train=True,
            limit=settings.dataset.train_limit,
        )
        raw_validation = _load_huggingface_split(
            settings, settings.dataset.validation_split
        )
        validation = HuggingFaceImageNetViewDataset(
            raw_validation,
            settings,
            split=settings.dataset.validation_split,
            train=False,
            limit=settings.dataset.validation_limit,
            expected_classes=train.classes,
        )
    else:
        train = ImageNetViewDataset(
            root / settings.dataset.train_split,
            settings,
            train=True,
            limit=settings.dataset.train_limit,
        )
        validation = ImageNetViewDataset(
            root / settings.dataset.validation_split,
            settings,
            train=False,
            limit=settings.dataset.validation_limit,
            expected_classes=train.classes,
        )
    if settings.dataset.strict_standard_counts and settings.dataset.train_limit is None:
        if train.base_sample_count != settings.dataset.expected_train_samples:
            raise RuntimeError(
                f"Expected standard ImageNet-1K train count {settings.dataset.expected_train_samples:,}, "
                f"found {train.base_sample_count:,} under {root / settings.dataset.train_split}"
            )
    if settings.dataset.strict_standard_counts and settings.dataset.validation_limit is None:
        if validation.base_sample_count != settings.dataset.expected_validation_samples:
            raise RuntimeError(
                f"Expected standard ImageNet-1K validation count "
                f"{settings.dataset.expected_validation_samples:,}, found "
                f"{validation.base_sample_count:,} under {root / settings.dataset.validation_split}"
            )
    if len(train.classes) != settings.model.num_classes:
        raise RuntimeError(
            f"Expected {settings.model.num_classes} ImageNet classes, found {len(train.classes)}"
        )
    class_names = imagenet_class_names(len(train.classes))
    digest = dataset_digest(train, validation)
    return ImageNetBundle(train, validation, class_names, train.classes, digest)


def imagenet_class_names(expected_count: int = 1000) -> list[str]:
    """Use torchvision's packaged category metadata without downloading weights."""

    try:
        from torchvision.models import ResNet50_Weights
        categories = list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    except Exception as error:
        raise RuntimeError(
            "Could not load ImageNet human-readable class names from torchvision metadata. "
            "Install a current torchvision build; silent use of synset folder IDs is forbidden "
            "because it would make CLIP text prototypes invalid."
        ) from error
    if len(categories) != expected_count:
        raise RuntimeError(f"Expected {expected_count} ImageNet names, found {len(categories)}")
    return categories


def dataset_digest(train: ImageNetViewDataset, validation: ImageNetViewDataset) -> str:
    digest = hashlib.sha256()
    digest.update(b"imagenet1k-optical-mixer-dataset-v1\n")
    for split, dataset in (("train", train), ("validation", validation)):
        digest.update(f"{split}:{dataset.base_sample_count}:{dataset.views}\n".encode())
        if dataset.dataset_source == "huggingface":
            digest.update(
                (
                    f"{dataset.dataset_id}:{dataset.revision}:"
                    f"{dataset.source_split}:{dataset.fingerprint}\n"
                ).encode()
            )
            if dataset.base_sample_count != dataset.raw_sample_count:
                for source_index, target in zip(
                    dataset.base_indices, dataset.targets
                ):
                    digest.update(f"{source_index}:{target}\n".encode())
        else:
            for path, target in dataset.samples:
                try:
                    relative = Path(path).resolve().relative_to(
                        dataset.directory.resolve()
                    )
                except ValueError:
                    relative = Path(path).name
                digest.update(f"{relative.as_posix()}:{target}\n".encode())
    return digest.hexdigest()


def dataset_report(bundle: ImageNetBundle, settings: ExperimentSettings) -> dict:
    def counts(dataset: ImageNetViewDataset) -> dict[str, int]:
        values = torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes))
        return {bundle.class_names[index]: int(value) for index, value in enumerate(values)}

    return {
        "dataset": "imagenet1k",
        "source": settings.dataset.source,
        "data_root": str(settings.dataset.root),
        "download": settings.dataset.download,
        "hf_dataset_id": (
            settings.dataset.hf_dataset_id
            if settings.dataset.source == "huggingface"
            else None
        ),
        "hf_revision": (
            settings.dataset.hf_revision
            if settings.dataset.source == "huggingface"
            else None
        ),
        "hf_cache_dir": (
            str(settings.dataset.hf_cache_dir)
            if settings.dataset.source == "huggingface"
            else None
        ),
        "train_source_fingerprint": getattr(bundle.train, "fingerprint", None),
        "validation_source_fingerprint": getattr(
            bundle.validation, "fingerprint", None
        ),
        "dataset_digest": bundle.digest,
        "train_samples": bundle.train.base_sample_count,
        "validation_samples": bundle.validation.base_sample_count,
        "train_cached_views_per_image": bundle.train.views,
        "validation_cached_views_per_image": 1,
        "class_count": len(bundle.class_names),
        "folder_classes": bundle.folder_classes,
        "class_names": bundle.class_names,
        "per_class_train_counts": counts(bundle.train),
        "per_class_validation_counts": counts(bundle.validation),
        "train_limit": settings.dataset.train_limit,
        "validation_limit": settings.dataset.validation_limit,
        "transform_schema_version": TRANSFORM_SCHEMA_VERSION,
        "train_transform_version": settings.clip.train_transform_version,
        "seed": settings.dataset.seed,
    }


def write_dataset_report(bundle: ImageNetBundle, settings: ExperimentSettings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset_report(bundle, settings), indent=2) + "\n", encoding="utf-8")


def _stratified_limit(targets: list[int], limit: int, seed: int) -> list[int]:
    if limit >= len(targets):
        return list(range(len(targets)))
    classes: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        classes.setdefault(int(target), []).append(index)
    generator = random.Random(seed)
    for values in classes.values():
        generator.shuffle(values)
    result: list[int] = []
    class_ids = sorted(classes)
    cursor = 0
    while len(result) < limit:
        class_id = class_ids[cursor % len(class_ids)]
        values = classes[class_id]
        offset = cursor // len(class_ids)
        if offset < len(values):
            result.append(values[offset])
        cursor += 1
        if cursor > limit * max(2, len(class_ids)):
            remaining = [index for values in classes.values() for index in values if index not in set(result)]
            result.extend(remaining[: limit - len(result)])
            break
    return sorted(result[:limit])


def _load_huggingface_split(settings: ExperimentSettings, split: str):
    try:
        from datasets import DownloadConfig, load_dataset
        from huggingface_hub import get_token
    except Exception as error:
        raise RuntimeError(
            "The Hugging Face ImageNet backend requires `datasets`, "
            "`huggingface_hub`, and `hf_xet`."
        ) from error
    token = get_token()
    if not token:
        raise RuntimeError(
            "ImageNet-1K authorization is missing. First accept the terms at "
            "https://huggingface.co/datasets/ILSVRC/imagenet-1k, then run "
            "`hf auth login` in this server account. The token is read from the "
            "Hugging Face credential store and is never written to experiment JSON."
        )
    cache_dir = settings.dataset.hf_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_config = DownloadConfig(
        cache_dir=str(cache_dir / "downloads"),
        local_files_only=not settings.dataset.download,
    )
    try:
        return load_dataset(
            settings.dataset.hf_dataset_id,
            split=split,
            revision=settings.dataset.hf_revision,
            cache_dir=str(cache_dir),
            token=token,
            download_config=download_config,
        )
    except Exception as error:
        mode = "download/reuse" if settings.dataset.download else "offline reuse"
        raise RuntimeError(
            f"Could not {mode} gated ImageNet split {split!r} from "
            f"{settings.dataset.hf_dataset_id}@{settings.dataset.hf_revision}. "
            "Confirm that the same Hugging Face account accepted the ImageNet "
            "terms and that its read token is active. Cache directory: "
            f"{cache_dir}. Original error: {type(error).__name__}: {error}"
        ) from error


def denormalize_clip_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.new_tensor(CLIP_MEAN)[:, None, None]
    std = tensor.new_tensor(CLIP_STD)[:, None, None]
    return (tensor * std + mean).clamp(0, 1)
