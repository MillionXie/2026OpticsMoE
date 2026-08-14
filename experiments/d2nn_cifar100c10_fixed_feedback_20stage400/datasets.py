from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .settings import Settings


def _stable_seed(*values: object) -> int:
    text = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**31)


def _gray_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(np.asarray(image) if not isinstance(image, torch.Tensor) else image)
    if value.ndim == 3 and value.shape[-1] == 3:
        value = value.permute(2, 0, 1)
    value = value.float() / 255.0 if value.max() > 1.0 else value.float()
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.shape[0] != 3:
        raise ValueError(f"Expected RGB image, got shape {tuple(value.shape)}")
    return (0.299 * value[0] + 0.587 * value[1] + 0.114 * value[2]).unsqueeze(0)


def _augment(image: torch.Tensor, *, seed: int, padding: int, horizontal_flip: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    if padding > 0:
        padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        max_offset = 2 * padding
        y0 = int(torch.randint(0, max_offset + 1, (), generator=generator))
        x0 = int(torch.randint(0, max_offset + 1, (), generator=generator))
        image = padded[:, y0 : y0 + 32, x0 : x0 + 32]
    if horizontal_flip and bool(torch.rand((), generator=generator) < 0.5):
        image = torch.flip(image, dims=(-1,))
    return image.contiguous()


class CIFAR100View(Dataset):
    def __init__(
        self,
        base: Dataset,
        indices: Iterable[int],
        *,
        training: bool,
        seed: int,
        crop_padding: int,
        horizontal_flip: bool,
    ) -> None:
        self.base = base
        self.indices = tuple(int(i) for i in indices)
        self.training = bool(training)
        self.seed = int(seed)
        self.crop_padding = int(crop_padding)
        self.horizontal_flip = bool(horizontal_flip)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        base_index = self.indices[index]
        image, target = self.base[base_index]
        image = _gray_tensor(np.asarray(image))
        if self.training:
            image = _augment(
                image,
                seed=_stable_seed(self.seed, self.epoch, base_index),
                padding=self.crop_padding,
                horizontal_flip=self.horizontal_flip,
            )
        return {
            "image": image,
            "target": int(target),
            "local_target": int(target),
            "sample_id": f"cifar100_train_{base_index:05d}",
            "base_index": int(base_index),
            "corruption": "clean",
        }


@dataclass(frozen=True)
class CorruptionEntry:
    corruption: str
    absolute_index: int
    base_index: int
    global_target: int
    local_target: int


class CIFAR100CorruptionView(Dataset):
    def __init__(
        self,
        root: Path,
        entries: Iterable[CorruptionEntry],
        *,
        training: bool,
        seed: int,
        crop_padding: int,
        horizontal_flip: bool,
    ) -> None:
        self.root = Path(root)
        self.entries = tuple(entries)
        self.training = bool(training)
        self.seed = int(seed)
        self.crop_padding = int(crop_padding)
        self.horizontal_flip = bool(horizontal_flip)
        self.epoch = 0
        self._arrays: dict[str, np.ndarray] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def _array(self, corruption: str) -> np.ndarray:
        if corruption not in self._arrays:
            self._arrays[corruption] = np.load(self.root / f"{corruption}.npy", mmap_mode="r")
        return self._arrays[corruption]

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        image = _gray_tensor(self._array(entry.corruption)[entry.absolute_index])
        if self.training:
            image = _augment(
                image,
                seed=_stable_seed(self.seed, self.epoch, entry.base_index, entry.corruption),
                padding=self.crop_padding,
                horizontal_flip=self.horizontal_flip,
            )
        return {
            "image": image,
            "target": entry.global_target,
            "local_target": entry.local_target,
            "sample_id": f"{entry.corruption}_base{entry.base_index:05d}",
            "base_index": entry.base_index,
            "corruption": entry.corruption,
        }


@dataclass(frozen=True)
class DatasetBundle:
    class_names: tuple[str, ...]
    selected_class_indices: tuple[int, ...]
    pretrain_train: CIFAR100View
    pretrain_validation: CIFAR100View
    finetune_train: CIFAR100CorruptionView
    finetune_validation: CIFAR100CorruptionView
    finetune_test: CIFAR100CorruptionView


CIFAR100C_ARCHIVE_BYTES = 2_918_473_216


def _archive_is_complete(path: Path, *, expected_bytes: int) -> bool:
    return (
        path.exists()
        and path.stat().st_size == expected_bytes
        and tarfile.is_tarfile(path)
    )


def _download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_bytes: int = CIFAR100C_ARCHIVE_BYTES,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _archive_is_complete(destination, expected_bytes=expected_bytes):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and not temporary.exists():
        destination.replace(temporary)
    wget = shutil.which("wget")
    if wget is not None:
        subprocess.run([wget, "-c", url, "-O", str(temporary)], check=True)
        if temporary.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"CIFAR-100-C download is incomplete: {temporary.stat().st_size} "
                f"of {expected_bytes} bytes. Re-run prepare_data to resume it."
            )
        temporary.replace(destination)
        return
    headers: dict[str, str] = {}
    mode = "wb"
    if temporary.exists():
        headers["Range"] = f"bytes={temporary.stat().st_size}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        # Some mirrors ignore Range and return the complete object. In that
        # case overwrite the partial archive instead of appending a duplicate.
        response_mode = mode if getattr(response, "status", None) == 206 else "wb"
        with temporary.open(response_mode) as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"CIFAR-100-C download is incomplete: {temporary.stat().st_size} "
            f"of {expected_bytes} bytes. Re-run prepare_data to resume it."
        )
    temporary.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def prepare_data(settings: Settings) -> dict[str, object]:
    try:
        from torchvision.datasets import CIFAR100
    except Exception as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("torchvision is required for CIFAR-100") from exc

    settings.data.torchvision_root.mkdir(parents=True, exist_ok=True)
    train = CIFAR100(root=settings.data.torchvision_root, train=True, download=True)
    test = CIFAR100(root=settings.data.torchvision_root, train=False, download=True)
    expected_files = [settings.data.cifar100c_root / "labels.npy"] + [
        settings.data.cifar100c_root / f"{name}.npy" for name in settings.data.corruptions
    ]
    if not all(path.exists() for path in expected_files):
        archive = settings.data.root / "CIFAR-100-C.tar"
        _download_with_resume(settings.data.cifar100c_url, archive)
        _safe_extract(archive, settings.data.root)
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"CIFAR-100-C extraction is incomplete: {missing}")
    metadata = {
        "cifar100_train": len(train),
        "cifar100_test": len(test),
        "class_names": list(train.classes),
        "cifar100c_root": str(settings.data.cifar100c_root),
        "corruptions": list(settings.data.corruptions),
        "severity": settings.data.severity,
        "selected_classes": list(settings.data.selected_classes),
        "source_url": settings.data.cifar100c_url,
    }
    settings.data.root.mkdir(parents=True, exist_ok=True)
    (settings.data.root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _stratified_pretrain_indices(targets: list[int], seed: int) -> tuple[list[int], list[int]]:
    generator = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    targets_array = np.asarray(targets)
    for class_index in range(100):
        indices = np.flatnonzero(targets_array == class_index)
        generator.shuffle(indices)
        validation_indices.extend(int(v) for v in indices[:50])
        train_indices.extend(int(v) for v in indices[50:])
    return train_indices, validation_indices


def _downstream_entries(
    settings: Settings,
    class_names: tuple[str, ...],
) -> tuple[list[CorruptionEntry], list[CorruptionEntry], list[CorruptionEntry], tuple[int, ...]]:
    selected_indices = tuple(class_names.index(name) for name in settings.data.selected_classes)
    labels = np.load(settings.data.cifar100c_root / "labels.npy", mmap_mode="r")
    severity_offset = (settings.data.severity - 1) * 10_000
    severity_labels = np.asarray(labels[severity_offset : severity_offset + 10_000])
    generator = np.random.default_rng(settings.data.split_seed)
    split_base: dict[str, list[tuple[int, int, int]]] = {"train": [], "validation": [], "test": []}
    for local_target, global_target in enumerate(selected_indices):
        base_indices = np.flatnonzero(severity_labels == global_target)
        if len(base_indices) != 100:
            raise RuntimeError(
                f"Expected 100 CIFAR-100-C base images for {class_names[global_target]}, got {len(base_indices)}"
            )
        generator.shuffle(base_indices)
        n_train = settings.data.train_per_class
        n_val = settings.data.validation_per_class
        n_test = settings.data.test_per_class
        pieces = {
            "train": base_indices[:n_train],
            "validation": base_indices[n_train : n_train + n_val],
            "test": base_indices[n_train + n_val : n_train + n_val + n_test],
        }
        for split, indices in pieces.items():
            split_base[split].extend((int(i), int(global_target), int(local_target)) for i in indices)

    def expand(split: str) -> list[CorruptionEntry]:
        result: list[CorruptionEntry] = []
        for corruption in settings.data.corruptions:
            for base_index, global_target, local_target in split_base[split]:
                result.append(
                    CorruptionEntry(
                        corruption=corruption,
                        absolute_index=severity_offset + base_index,
                        base_index=base_index,
                        global_target=global_target,
                        local_target=local_target,
                    )
                )
        return result

    train, validation, test = expand("train"), expand("validation"), expand("test")
    train_ids = {entry.base_index for entry in train}
    validation_ids = {entry.base_index for entry in validation}
    test_ids = {entry.base_index for entry in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Base-image leakage detected in downstream split")
    return train, validation, test, selected_indices


def load_datasets(settings: Settings, *, prepare: bool = True) -> DatasetBundle:
    if prepare:
        prepare_data(settings)
    from torchvision.datasets import CIFAR100

    clean_train = CIFAR100(root=settings.data.torchvision_root, train=True, download=False)
    class_names = tuple(str(value) for value in clean_train.classes)
    pretrain_train_indices, pretrain_validation_indices = _stratified_pretrain_indices(
        list(clean_train.targets), settings.training.pretrain_seed
    )
    downstream_train, downstream_validation, downstream_test, selected_indices = _downstream_entries(
        settings, class_names
    )
    bundle = DatasetBundle(
        class_names=class_names,
        selected_class_indices=selected_indices,
        pretrain_train=CIFAR100View(
            clean_train,
            pretrain_train_indices,
            training=True,
            seed=settings.training.pretrain_seed,
            crop_padding=settings.data.pretrain_crop_padding,
            horizontal_flip=settings.data.pretrain_horizontal_flip,
        ),
        pretrain_validation=CIFAR100View(
            clean_train,
            pretrain_validation_indices,
            training=False,
            seed=settings.training.pretrain_seed,
            crop_padding=0,
            horizontal_flip=False,
        ),
        finetune_train=CIFAR100CorruptionView(
            settings.data.cifar100c_root,
            downstream_train,
            training=True,
            seed=settings.data.split_seed,
            crop_padding=settings.data.finetune_crop_padding,
            horizontal_flip=settings.data.finetune_horizontal_flip,
        ),
        finetune_validation=CIFAR100CorruptionView(
            settings.data.cifar100c_root,
            downstream_validation,
            training=False,
            seed=settings.data.split_seed,
            crop_padding=0,
            horizontal_flip=False,
        ),
        finetune_test=CIFAR100CorruptionView(
            settings.data.cifar100c_root,
            downstream_test,
            training=False,
            seed=settings.data.split_seed,
            crop_padding=0,
            horizontal_flip=False,
        ),
    )
    manifest = {
        "selected_classes": list(settings.data.selected_classes),
        "selected_class_indices": list(selected_indices),
        "corruptions": list(settings.data.corruptions),
        "severity": settings.data.severity,
        "pretrain_train_samples": len(bundle.pretrain_train),
        "pretrain_validation_samples": len(bundle.pretrain_validation),
        "finetune_train_samples": len(bundle.finetune_train),
        "finetune_validation_samples": len(bundle.finetune_validation),
        "finetune_test_samples": len(bundle.finetune_test),
        "base_image_split": {
            "train_per_class": settings.data.train_per_class,
            "validation_per_class": settings.data.validation_per_class,
            "test_per_class": settings.data.test_per_class,
        },
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle


def epoch_order(length: int, *, epoch: int, seed: int, limit: int | None) -> list[int]:
    if length <= 0:
        return []
    generator = torch.Generator().manual_seed(int(seed))
    base = torch.randperm(length, generator=generator).tolist()
    if limit is None or limit >= length:
        epoch_generator = torch.Generator().manual_seed(_stable_seed(seed, epoch, "order"))
        return torch.randperm(length, generator=epoch_generator).tolist()
    start = ((int(epoch) - 1) * int(limit)) % length
    repeats = (start + int(limit) + length - 1) // length + 1
    tiled = (base * repeats)[start : start + int(limit)]
    return [int(value) for value in tiled]


def order_digest(indices: Iterable[int]) -> str:
    array = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()
