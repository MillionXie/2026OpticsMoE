from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor


SplitName = Literal["train800", "val200", "train800val200", "test"]
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    directory: str
    category: str
    num_classes: int
    test_samples: int


TASK_SPECS: dict[str, TaskSpec] = {
    "cifar100": TaskSpec("cifar100", "cifar", "natural", 100, 10_000),
    "flowers102": TaskSpec("flowers102", "oxford_flowers102", "natural", 102, 6_149),
    "eurosat": TaskSpec("eurosat", "eurosat", "specialized", 10, 5_400),
    "patch_camelyon": TaskSpec("patch_camelyon", "patch_camelyon", "specialized", 2, 32_768),
    "dsprites_orientation": TaskSpec(
        "dsprites_orientation", "dsprites_ori", "structured", 16, 73_728
    ),
    "smallnorb_azimuth": TaskSpec(
        "smallnorb_azimuth", "smallnorb_azi", "structured", 18, 12_150
    ),
}


def image_transform() -> Compose:
    """Deterministic VTAB resize with the normalization required by the Qwen stem."""

    return Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


class Vtab1kClassificationDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path, task: str, split: SplitName) -> None:
        if task not in TASK_SPECS:
            raise ValueError(f"Unsupported VTAB task: {task}")
        if split not in {"train800", "val200", "train800val200", "test"}:
            raise ValueError(f"Unsupported VTAB split: {split}")
        self.spec = TASK_SPECS[task]
        self.split = split
        self.root = Path(root).expanduser().resolve() / self.spec.directory
        self.list_path = self.root / f"{split}.txt"
        if not self.list_path.is_file():
            raise FileNotFoundError(f"Missing standard VTAB split: {self.list_path}")
        self.samples: list[tuple[Path, int]] = []
        for line_number, raw_line in enumerate(
            self.list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                relative, label_text = line.rsplit(" ", 1)
                label = int(label_text)
            except Exception as error:
                raise ValueError(
                    f"Malformed {self.list_path}:{line_number}: {raw_line!r}"
                ) from error
            image_path = (self.root / relative).resolve()
            if self.root not in image_path.parents:
                raise ValueError(f"VTAB sample escapes its dataset root: {relative}")
            self.samples.append((image_path, label))
        expected = {
            "train800": 800,
            "val200": 200,
            "train800val200": 1_000,
            "test": self.spec.test_samples,
        }[split]
        if len(self.samples) != expected:
            raise RuntimeError(
                f"{task}/{split} has {len(self.samples)} samples; expected {expected}"
            )
        labels = {label for _, label in self.samples}
        if min(labels) != 0 or max(labels) != self.spec.num_classes - 1:
            raise RuntimeError(
                f"{task}/{split} labels span [{min(labels)},{max(labels)}], "
                f"expected [0,{self.spec.num_classes - 1}]"
            )
        self.transform = image_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, label = self.samples[index]
        if not path.is_file():
            raise FileNotFoundError(f"Missing VTAB image: {path}")
        image: Image.Image = default_loader(path)
        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": f"{self.split}:{path.name}",
        }

    def manifest_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.spec.name.encode("utf-8"))
        digest.update(self.split.encode("utf-8"))
        digest.update(self.list_path.read_bytes())
        return digest.hexdigest()


def build_loaders(
    root: str | Path,
    task: str,
    *,
    train_batch_size: int,
    evaluation_batch_size: int,
    num_workers: int,
    seed: int,
    smoke_samples: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    datasets = {
        split: Vtab1kClassificationDataset(root, task, split)
        for split in ("train800", "val200", "train800val200", "test")
    }
    if smoke_samples is not None:
        limit = max(1, int(smoke_samples))
        datasets = {
            split: torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
            for split, dataset in datasets.items()
        }
    loaders = {
        "train800": DataLoader(
            datasets["train800"], batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,
        ),
        "val200": DataLoader(
            datasets["val200"], batch_size=evaluation_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,
        ),
        "train800val200": DataLoader(
            datasets["train800val200"], batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=evaluation_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,
        ),
    }
    spec = TASK_SPECS[task]
    metadata = {
        "protocol": "VTAB-1k train800/val200/train800val200/test",
        "task": task,
        "dataset_directory": spec.directory,
        "category": spec.category,
        "num_classes": spec.num_classes,
        "counts": {key: len(value) for key, value in datasets.items()},
        "split_sha256": {
            split: Vtab1kClassificationDataset(root, task, split).manifest_sha256()
            for split in ("train800", "val200", "train800val200", "test")
        },
        "preprocessing": "resize224_bicubic_clip_normalization_no_augmentation",
    }
    return loaders, metadata


__all__ = [
    "TASK_SPECS",
    "TaskSpec",
    "Vtab1kClassificationDataset",
    "build_loaders",
]
