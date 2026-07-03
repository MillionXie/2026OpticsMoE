from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset[tuple[Image.Image, int]]
    test: Dataset[tuple[Image.Image, int]]
    class_names: list[str]
    metadata: dict[str, Any]


class RGBDataset(Dataset[tuple[Image.Image, int]]):
    def __init__(self, base: Dataset[Any], resize_to: int | None = None) -> None:
        self.base = base
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image, label = self.base[index]
        if not isinstance(image, Image.Image):
            try:
                from torchvision.transforms.functional import to_pil_image

                image = to_pil_image(image)
            except Exception as exc:
                raise TypeError(f"Cannot convert {type(image).__name__} to PIL.Image") from exc
        image = image.convert("RGB")
        if self.resize_to is not None:
            image = image.resize((self.resize_to, self.resize_to), Image.Resampling.BICUBIC)
        return image, int(label)


def load_dataset(
    name: str,
    root: Path,
    download: bool,
    resize_to: int | None,
    train_limit: int | None,
    test_limit: int | None,
    imagefolder_train: str = "train",
    imagefolder_test: str = "test",
) -> DatasetBundle:
    try:
        from torchvision import datasets
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc

    root.mkdir(parents=True, exist_ok=True)
    if name == "cifar10":
        train_base = datasets.CIFAR10(str(root), train=True, download=download)
        test_base = datasets.CIFAR10(str(root), train=False, download=download)
        class_names = list(train_base.classes)
    elif name == "cifar100":
        train_base = datasets.CIFAR100(str(root), train=True, download=download)
        test_base = datasets.CIFAR100(str(root), train=False, download=download)
        class_names = list(train_base.classes)
    elif name == "stl10":
        train_base = datasets.STL10(str(root), split="train", download=download)
        test_base = datasets.STL10(str(root), split="test", download=download)
        class_names = list(getattr(train_base, "classes", _stl10_classes()))
    elif name == "svhn":
        train_base = datasets.SVHN(str(root), split="train", download=download)
        test_base = datasets.SVHN(str(root), split="test", download=download)
        class_names = [str(index) for index in range(10)]
    elif name == "fashionmnist":
        train_base = datasets.FashionMNIST(str(root), train=True, download=download)
        test_base = datasets.FashionMNIST(str(root), train=False, download=download)
        class_names = list(train_base.classes)
    elif name == "imagefolder":
        train_dir = root / imagefolder_train
        test_dir = root / imagefolder_test
        train_base = datasets.ImageFolder(str(train_dir))
        test_base = datasets.ImageFolder(str(test_dir))
        if train_base.class_to_idx != test_base.class_to_idx:
            raise ValueError("ImageFolder train/test class mappings do not match")
        class_names = list(train_base.classes)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train = _limit(RGBDataset(train_base, resize_to), train_limit)
    test = _limit(RGBDataset(test_base, resize_to), test_limit)
    first_image, _ = train[0]
    return DatasetBundle(
        train=train,
        test=test,
        class_names=class_names,
        metadata={
            "name": name,
            "train_samples": len(train),
            "test_samples": len(test),
            "num_classes": len(class_names),
            "class_names": class_names,
            "first_image_size": list(first_image.size),
            "resize_to": resize_to,
        },
    )


def make_loader(
    dataset: Dataset[tuple[Image.Image, int]],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pil_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def pil_collate(batch: Sequence[tuple[Image.Image, int]]) -> tuple[list[Image.Image], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def _limit(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def _stl10_classes() -> list[str]:
    return ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

