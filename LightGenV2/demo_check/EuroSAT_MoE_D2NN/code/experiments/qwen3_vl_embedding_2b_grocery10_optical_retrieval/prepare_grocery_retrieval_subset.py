from __future__ import annotations

import argparse
import csv
import random
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .io_utils import sha256_records, write_csv, write_json
from .settings import Settings, load_settings


@dataclass(frozen=True)
class GrocerySample:
    sample_id: str
    image_path: Path
    sku_id: int
    sku_name: str
    sku_index: int
    split: str
    source_split: str
    is_gallery: bool

    def manifest_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_path"] = str(self.image_path.resolve())
        return value


@dataclass(frozen=True)
class GroceryRetrievalBundle:
    train_samples: tuple[GrocerySample, ...]
    test_samples: tuple[GrocerySample, ...]
    gallery_samples: tuple[GrocerySample, ...]
    class_names: tuple[str, ...]
    manifest_digest: str
    metadata: dict[str, Any]

    def all_samples(self) -> tuple[GrocerySample, ...]:
        return self.gallery_samples + self.train_samples + self.test_samples


class GroceryRetrievalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[GrocerySample],
        image_size: int,
        *,
        augment: bool = False,
        crop_scale_min: float = 0.9,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
        rotation_degrees: float = 5.0,
    ) -> None:
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.crop_scale_min = float(crop_scale_min)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.rotation_degrees = float(rotation_degrees)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        image = self._transform(image)
        return {"image": image, "sample": sample, "dataset_index": index}

    def _transform(self, image: Image.Image) -> Image.Image:
        if not self.augment:
            return ImageOps.fit(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        # Light packaging-safe augmentation: no mirror, erase, blur, MixUp, or CutMix.
        scale = random.uniform(self.crop_scale_min, 1.0)
        crop_w = max(1, round(image.width * scale))
        crop_h = max(1, round(image.height * scale))
        left = random.randint(0, max(0, image.width - crop_w))
        top = random.randint(0, max(0, image.height - crop_h))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if self.rotation_degrees:
            image = image.rotate(
                random.uniform(-self.rotation_degrees, self.rotation_degrees),
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0, 0, 0),
            )
        if self.brightness_jitter:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            )
        if self.contrast_jitter:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            )
        return image


def collate_grocery(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in items],
        "samples": [item["sample"] for item in items],
        "dataset_indices": [int(item["dataset_index"]) for item in items],
    }


def prepare_grocery_subset(
    settings: Settings, *, persist: bool = True
) -> GroceryRetrievalBundle:
    dataset_dir = _ensure_dataset(settings)
    classes_path = dataset_dir / "classes.csv"
    class_rows = list(csv.DictReader(classes_path.open("r", encoding="utf-8-sig", newline="")))
    if not class_rows:
        raise RuntimeError(f"No class rows found in {classes_path}")
    name_column = _column(class_rows[0], "Class Name (str)", "class_name", "name")
    id_column = _column(class_rows[0], "Class ID (int)", "class_id", "id")
    iconic_column = _column(
        class_rows[0], "Iconic Image Path (str)", "iconic_image_path", "iconic"
    )
    by_name = {str(row[name_column]).strip(): row for row in class_rows}
    missing_skus = [name for name in settings.selected_skus if name not in by_name]
    if missing_skus:
        raise RuntimeError(
            f"Configured SKUs are absent from Grocery Store classes.csv: {missing_skus}. "
            f"Available packaged candidates include: {list(by_name)[28:59]}"
        )

    selected_ids: dict[int, tuple[int, str]] = {}
    galleries: list[GrocerySample] = []
    for sku_index, sku_name in enumerate(settings.selected_skus):
        row = by_name[sku_name]
        original_id = int(str(row[id_column]).strip())
        selected_ids[original_id] = (sku_index, sku_name)
        iconic_path = _resolve_dataset_path(dataset_dir, str(row[iconic_column]))
        if not iconic_path.is_file():
            raise FileNotFoundError(f"Iconic gallery image is missing for {sku_name}: {iconic_path}")
        galleries.append(
            GrocerySample(
                sample_id=f"gallery:{sku_name}:0",
                image_path=iconic_path,
                sku_id=original_id,
                sku_name=sku_name,
                sku_index=sku_index,
                split="gallery",
                source_split="iconic",
                is_gallery=True,
            )
        )
    if settings.gallery_images_per_sku != 1:
        raise RuntimeError(
            "The official Grocery Store Dataset supplies one iconic image per fine SKU. "
            "gallery_images_per_sku must remain 1 unless additional standard gallery images "
            "are explicitly added to the subset loader."
        )

    official: dict[str, list[GrocerySample]] = {}
    for source_split in ("train", "val", "test"):
        official[source_split] = _read_split(
            dataset_dir, source_split, selected_ids, settings.selected_skus
        )
    train_pool = list(official["train"])
    if settings.merge_official_validation_into_train:
        train_pool.extend(official["val"])
    test_pool = list(official["test"])
    train = _limit_per_sku(
        train_pool, settings.train_limit_per_sku, settings.random_seed, "train"
    )
    test = _limit_per_sku(
        test_pool, settings.test_limit_per_sku, settings.random_seed + 1, "test"
    )
    _validate_partition(train, test, galleries, settings.selected_skus)

    records = [
        sample.manifest_record()
        for sample in sorted(
            galleries + train + test, key=lambda item: (item.split, item.sku_index, item.sample_id)
        )
    ]
    digest = sha256_records(records)
    metadata = {
        "dataset": "GroceryStoreDataset",
        "dataset_source": "https://github.com/marcusklasson/GroceryStoreDataset",
        "dataset_root": str(settings.dataset_root),
        "selected_skus": list(settings.selected_skus),
        "split_policy": (
            "official train+val -> train; official test -> test; official iconic -> gallery"
            if settings.merge_official_validation_into_train
            else "official train -> train; official test -> test; official iconic -> gallery"
        ),
        "validation_split_created": False,
        "seed": settings.random_seed,
        "manifest_sha256": digest,
        "counts": {
            "train": len(train),
            "test": len(test),
            "gallery": len(galleries),
        },
        "per_sku_counts": {
            split: dict(sorted(Counter(sample.sku_name for sample in values).items()))
            for split, values in (
                ("train", train),
                ("test", test),
                ("gallery", galleries),
            )
        },
        "excluded_categories": ["fruit", "vegetables", "loose produce"],
    }
    bundle = GroceryRetrievalBundle(
        tuple(train), tuple(test), tuple(galleries), settings.selected_skus, digest, metadata
    )
    if persist:
        rows = []
        for sample in bundle.all_samples():
            rows.append(sample.manifest_record())
        write_csv(
            settings.subset_manifest_path,
            rows,
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


def _ensure_dataset(settings: Settings) -> Path:
    candidates = [
        settings.dataset_root / "dataset",
        settings.dataset_root,
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("classes.csv", "train.txt", "test.txt")):
            return candidate
    if not settings.download:
        raise FileNotFoundError(
            "Grocery Store Dataset not found. Checked: "
            + ", ".join(str(path) for path in candidates)
        )
    settings.dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=settings.dataset_root.parent) as temporary_dir:
        archive_path = Path(temporary_dir) / "grocery_store_dataset.zip"
        request = urllib.request.Request(
            settings.download_url, headers={"User-Agent": "2026OpticsMoE-GroceryRetrieval/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, archive_path.open(
                "wb"
            ) as target:
                shutil.copyfileobj(response, target)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Grocery Store Dataset from {settings.download_url}: {exc}"
            ) from exc
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary_dir)
        roots = [
            path
            for path in Path(temporary_dir).iterdir()
            if path.is_dir() and (path / "dataset" / "classes.csv").is_file()
        ]
        if len(roots) != 1:
            raise RuntimeError(
                f"Downloaded archive layout was unexpected; candidate roots: {roots}"
            )
        if settings.dataset_root.exists():
            # A partial directory is preserved only if it is empty.
            if any(settings.dataset_root.iterdir()):
                raise RuntimeError(
                    f"Refusing to overwrite incomplete non-empty dataset_root {settings.dataset_root}"
                )
            settings.dataset_root.rmdir()
        shutil.copytree(roots[0], settings.dataset_root)
    return settings.dataset_root / "dataset"


def _read_split(
    dataset_dir: Path,
    source_split: str,
    selected_ids: dict[int, tuple[int, str]],
    selected_skus: Sequence[str],
) -> list[GrocerySample]:
    path = dataset_dir / f"{source_split}.txt"
    if not path.is_file():
        if source_split == "val":
            return []
        raise FileNotFoundError(f"Official Grocery Store split file is missing: {path}")
    samples: list[GrocerySample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for row_index, row in enumerate(reader):
            if len(row) < 2:
                raise RuntimeError(f"Malformed {path.name} line {row_index + 1}: {row}")
            original_id = int(row[1].strip())
            if original_id not in selected_ids:
                continue
            sku_index, sku_name = selected_ids[original_id]
            image_path = _resolve_dataset_path(dataset_dir, row[0].strip())
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Listed {source_split} image is missing for {sku_name}: {image_path}"
                )
            samples.append(
                GrocerySample(
                    sample_id=f"{source_split}:{sku_name}:{image_path.stem}",
                    image_path=image_path,
                    sku_id=original_id,
                    sku_name=sku_name,
                    sku_index=sku_index,
                    split="train" if source_split in {"train", "val"} else "test",
                    source_split=source_split,
                    is_gallery=False,
                )
            )
    present = {sample.sku_name for sample in samples}
    missing = sorted(set(selected_skus) - present)
    if missing and source_split != "val":
        raise RuntimeError(f"Official {source_split} split has no images for selected SKUs: {missing}")
    return samples


def _limit_per_sku(
    samples: Sequence[GrocerySample], limit: int | None, seed: int, split: str
) -> list[GrocerySample]:
    grouped: dict[int, list[GrocerySample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sku_index].append(sample)
    output: list[GrocerySample] = []
    for sku_index in sorted(grouped):
        values = sorted(grouped[sku_index], key=lambda item: item.sample_id)
        random.Random(seed + sku_index * 1009).shuffle(values)
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"{split}_limit_per_sku must be positive or null")
            values = values[:limit]
        output.extend(values)
    return output


def _validate_partition(
    train: Sequence[GrocerySample],
    test: Sequence[GrocerySample],
    galleries: Sequence[GrocerySample],
    selected_skus: Sequence[str],
) -> None:
    for name, values in (("train", train), ("test", test), ("gallery", galleries)):
        present = {sample.sku_name for sample in values}
        missing = sorted(set(selected_skus) - present)
        if missing:
            raise RuntimeError(f"{name} split has no samples for SKUs: {missing}")
    train_paths = {sample.image_path.resolve() for sample in train}
    test_paths = {sample.image_path.resolve() for sample in test}
    gallery_paths = {sample.image_path.resolve() for sample in galleries}
    overlaps = {
        "train_test": train_paths & test_paths,
        "train_gallery": train_paths & gallery_paths,
        "test_gallery": test_paths & gallery_paths,
    }
    leaking = {name: sorted(map(str, paths)) for name, paths in overlaps.items() if paths}
    if leaking:
        raise RuntimeError(f"Dataset leakage detected across retrieval partitions: {leaking}")


def _resolve_dataset_path(dataset_dir: Path, value: str) -> Path:
    relative = Path(value.lstrip("/\\"))
    attempts = [dataset_dir / relative, dataset_dir.parent / relative]
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    return attempts[0].resolve()


def _column(row: dict[str, str], *candidates: str) -> str:
    normalized = {name.strip().lower(): name for name in row}
    for candidate in candidates:
        if candidate.strip().lower() in normalized:
            return normalized[candidate.strip().lower()]
    raise RuntimeError(
        f"Required classes.csv column not found. Tried {list(candidates)}; found {list(row)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_grocery_subset(settings, persist=True)
    print(
        f"Prepared Grocery-{len(bundle.class_names)}: train={len(bundle.train_samples)} "
        f"test={len(bundle.test_samples)} gallery={len(bundle.gallery_samples)} "
        f"manifest={bundle.manifest_digest[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
