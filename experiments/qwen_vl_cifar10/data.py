from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class PILRGBDataset(Dataset[tuple[Image.Image, int]]):
    """Keep CIFAR images as PIL RGB and only resize when explicitly requested."""

    def __init__(self, dataset: Dataset[Any], resize_to: int | None = None) -> None:
        self.dataset = dataset
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image, label = self.dataset[index]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image from CIFAR10, received {type(image).__name__}.")
        image = image.convert("RGB")
        if self.resize_to is not None:
            image = image.resize((self.resize_to, self.resize_to), Image.Resampling.BICUBIC)
        return image, int(label)


@dataclass(frozen=True)
class CIFAR10Data:
    train_dataset: Dataset[tuple[Image.Image, int]]
    test_dataset: Dataset[tuple[Image.Image, int]]
    class_names: list[str]


def load_cifar10(
    data_root: Path,
    image_size: int,
    resize_to: int | None,
    train_limit: int | None,
    test_limit: int | None,
    download: bool = True,
) -> CIFAR10Data:
    try:
        from torchvision.datasets import CIFAR10
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "torchvision is required for CIFAR-10. Install compatible torch/torchvision builds "
            "using experiments/qwen_vl_cifar10/requirements.txt."
        ) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    train_base = CIFAR10(root=str(data_root), train=True, download=download, transform=None)
    test_base = CIFAR10(root=str(data_root), train=False, download=download, transform=None)

    actual_size = train_base[0][0].size
    if actual_size != (image_size, image_size):
        raise ValueError(
            f"--image-size describes the unmodified dataset image size, but CIFAR-10 returned "
            f"{actual_size[0]}x{actual_size[1]} while --image-size={image_size}. Use "
            "--image-size 32; use --resize-to only for an explicit preprocessing ablation."
        )

    train: Dataset[tuple[Image.Image, int]] = PILRGBDataset(train_base, resize_to)
    test: Dataset[tuple[Image.Image, int]] = PILRGBDataset(test_base, resize_to)
    train = _limit_dataset(train, train_limit)
    test = _limit_dataset(test, test_limit)
    return CIFAR10Data(train, test, list(train_base.classes))


def _limit_dataset(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("Dataset limits must be positive.")
    return Subset(dataset, range(min(limit, len(dataset))))


def pil_collate(batch: Sequence[tuple[Image.Image, int]]) -> tuple[list[Image.Image], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def make_image_loader(
    dataset: Dataset[tuple[Image.Image, int]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[tuple[list[Image.Image], torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pil_collate,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
