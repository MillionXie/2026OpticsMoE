from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    sha256_records,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    GrocerySample,
)

from .settings import CIFAR10_CLASSES, CIFAR10Settings


def prepare_cifar10_subset(
    settings: CIFAR10Settings, *, persist: bool = True
) -> GroceryRetrievalBundle:
    try:
        from torchvision.datasets import CIFAR10
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc

    official_train = CIFAR10(
        str(settings.dataset_root), train=True, download=settings.download
    )
    official_test = CIFAR10(
        str(settings.dataset_root), train=False, download=settings.download
    )
    if tuple(official_train.classes) != CIFAR10_CLASSES:
        raise RuntimeError("Installed CIFAR-10 labels do not match the official order")

    train: list[GrocerySample] = []
    test: list[GrocerySample] = []
    gallery: list[GrocerySample] = []
    for class_index, class_name in enumerate(CIFAR10_CLASSES):
        train_indices = [
            index
            for index, label in enumerate(official_train.targets)
            if int(label) == class_index
        ]
        random.Random(f"{settings.random_seed}:train:{class_name}").shuffle(
            train_indices
        )
        gallery_indices = train_indices[: settings.gallery_images_per_sku]
        retained_train = train_indices[settings.gallery_images_per_sku :]
        if settings.train_limit_per_sku is not None:
            retained_train = retained_train[: int(settings.train_limit_per_sku)]

        test_indices = [
            index
            for index, label in enumerate(official_test.targets)
            if int(label) == class_index
        ]
        random.Random(f"{settings.random_seed}:test:{class_name}").shuffle(
            test_indices
        )
        if settings.test_limit_per_sku is not None:
            test_indices = test_indices[: int(settings.test_limit_per_sku)]
        if not retained_train or not test_indices:
            raise RuntimeError(f"CIFAR-10 class {class_name} has an empty split")

        gallery.extend(
            _materialize_samples(
                official_train,
                gallery_indices,
                class_index,
                class_name,
                "gallery",
                "official_train",
                settings.materialized_image_root,
                True,
            )
        )
        train.extend(
            _materialize_samples(
                official_train,
                retained_train,
                class_index,
                class_name,
                "train",
                "official_train",
                settings.materialized_image_root,
                False,
            )
        )
        test.extend(
            _materialize_samples(
                official_test,
                test_indices,
                class_index,
                class_name,
                "test",
                "official_test",
                settings.materialized_image_root,
                False,
            )
        )

    records = [
        sample.manifest_record()
        for sample in sorted(
            gallery + train + test,
            key=lambda item: (item.split, item.sku_index, item.sample_id),
        )
    ]
    digest = sha256_records(records)
    metadata = {
        "dataset": "CIFAR-10",
        "dataset_source": "torchvision.datasets.CIFAR10",
        "dataset_root": str(settings.dataset_root),
        "selected_categories": list(CIFAR10_CLASSES),
        "split_policy": (
            "official train: seeded 3-image gallery per class then remaining images "
            "for training; official test used only for evaluation"
        ),
        "seed": settings.random_seed,
        "manifest_sha256": digest,
        "counts": {
            "train": len(train),
            "test": len(test),
            "gallery": len(gallery),
        },
        "per_category_counts": {
            split: dict(sorted(Counter(item.sku_name for item in values).items()))
            for split, values in (
                ("train", train),
                ("test", test),
                ("gallery", gallery),
            )
        },
    }
    bundle = GroceryRetrievalBundle(
        tuple(train),
        tuple(test),
        tuple(gallery),
        CIFAR10_CLASSES,
        digest,
        metadata,
    )
    if persist:
        write_csv(
            settings.subset_manifest_path,
            [sample.manifest_record() for sample in bundle.all_samples()],
            [
                "sample_id",
                "image_path",
                "sku_id",
                "sku_name",
                "sku_index",
                "split",
                "source_split",
                "is_gallery",
            ],
        )
        write_json(settings.output_dir / "dataset.json", metadata)
        write_json(
            settings.output_dir
            / "manifests"
            / f"{settings.dataset_variant}_subset_metadata.json",
            metadata,
        )
    return bundle


def _materialize_samples(
    dataset,
    indices: list[int],
    class_index: int,
    class_name: str,
    split: str,
    source_split: str,
    root: Path,
    is_gallery: bool,
) -> list[GrocerySample]:
    directory = root / source_split / class_name
    directory.mkdir(parents=True, exist_ok=True)
    samples: list[GrocerySample] = []
    for index in indices:
        path = directory / f"{index:05d}.png"
        if not path.is_file():
            image, label = dataset[index]
            if int(label) != class_index:
                raise RuntimeError("CIFAR-10 label changed during materialization")
            image.convert("RGB").save(path, format="PNG")
        samples.append(
            GrocerySample(
                sample_id=f"cifar10:{source_split}:{index:05d}",
                image_path=path,
                sku_id=class_index,
                sku_name=class_name,
                sku_index=class_index,
                split=split,
                source_split=source_split,
                is_gallery=is_gallery,
            )
        )
    return samples
