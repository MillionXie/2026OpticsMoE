from __future__ import annotations

import json
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from .io_utils import write_csv, write_json


@dataclass(frozen=True)
class SALICONRecord:
    sample_index: int
    split: str
    image_id: int
    image_path: Path
    density_path: Path
    fixation_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.split}/{self.image_id:012d}"


@dataclass(frozen=True)
class SALICONBundle:
    train_records: tuple[SALICONRecord, ...]
    validation_records: tuple[SALICONRecord, ...]
    metadata: dict[str, Any]


def prepare_salicon(settings: Any, *, persist: bool = True) -> SALICONBundle:
    _ensure_sources(settings)
    records: list[SALICONRecord] = []
    split_metadata: dict[str, Any] = {}
    sample_index = 0
    for split, limit in (
        ("train", settings.train_limit),
        ("validation", settings.validation_limit),
    ):
        annotation_path = _annotation_path(settings.data_root, split)
        images = list(_iter_json_array(annotation_path, "images"))
        if not images:
            raise RuntimeError(
                f"{annotation_path} has no COCO-style images array"
            )
        ordered_images = sorted(images, key=lambda value: int(value["id"]))
        if limit is not None:
            ordered_images = ordered_images[: int(limit)]
        annotation_rows = _ensure_prepared_maps(
            annotation_path=annotation_path,
            image_rows=ordered_images,
            cache_root=settings.artifact_cache_dir,
            split=split,
            output_size=settings.image_size,
            sigma_px=settings.density_sigma_px,
            enabled=settings.materialize_density_maps,
        )
        for image_info in ordered_images:
            image_id = int(image_info["id"])
            file_name = str(image_info["file_name"])
            image_path = _find_image(settings.data_root, split, file_name)
            density_path, fixation_path = _map_paths(
                settings.artifact_cache_dir, split, image_id
            )
            if not density_path.is_file() or not fixation_path.is_file():
                raise FileNotFoundError(
                    f"Prepared SALICON maps are missing for {split}/{image_id}. "
                    "Enable dataset.materialize_density_maps."
                )
            records.append(
                SALICONRecord(
                    sample_index,
                    split,
                    image_id,
                    image_path,
                    density_path,
                    fixation_path,
                )
            )
            sample_index += 1
        split_metadata[split] = {
            "images": len(ordered_images),
            "annotation_images": len(images),
            "annotation_rows": annotation_rows,
            "images_with_fixations": len(ordered_images),
            "annotation_path": str(annotation_path),
        }
    train = tuple(record for record in records if record.split == "train")
    validation = tuple(
        record for record in records if record.split == "validation"
    )
    if not train or not validation:
        raise RuntimeError("SALICON train/validation split is empty")
    if {row.image_id for row in train} & {row.image_id for row in validation}:
        raise RuntimeError("SALICON train/validation image leakage detected")
    metadata = {
        "dataset": "SALICON 2015r1",
        "task": "human_fixation_density_prediction",
        "official_split_policy": (
            "official train2014 trains the model; public val2014 ground truth is "
            "held out for checkpoint selection/evaluation; test annotations are private"
        ),
        "train_images": len(train),
        "validation_images": len(validation),
        "image_id_disjoint": True,
        "image_size": settings.image_size,
        "density_sigma_px_at_source": settings.density_sigma_px,
        "density_construction": (
            "fixation coordinates projected to 224x224, then Gaussian blurred "
            "with separate x/y source-sigma scaling"
        ),
        "density_resize_interpolation": "bilinear during aligned augmentation",
        "fixation_resize_interpolation": "coordinate projection / nearest for augmentation",
        "splits": split_metadata,
    }
    if persist:
        write_json(settings.output_dir / "dataset.json", metadata)
        write_csv(
            settings.output_dir / "manifests" / "samples.csv",
            [
                {
                    "sample_index": row.sample_index,
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "image_id": row.image_id,
                    "image_path": str(row.image_path),
                    "density_path": str(row.density_path),
                    "fixation_path": str(row.fixation_path),
                }
                for row in records
            ],
            [
                "sample_index",
                "sample_id",
                "split",
                "image_id",
                "image_path",
                "density_path",
                "fixation_path",
            ],
        )
    return SALICONBundle(train, validation, metadata)


class SALICONSaliencyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[SALICONRecord],
        settings: Any,
        *,
        training: bool,
    ) -> None:
        self.records = tuple(records)
        self.settings = settings
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        with Image.open(row.image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(row.density_path) as handle:
            density = handle.convert("L")
        with Image.open(row.fixation_path) as handle:
            fixation = handle.convert("L")
        image, density, fixation = _aligned_transform(
            image,
            density,
            fixation,
            self.settings,
            training=self.training,
        )
        density_array = np.asarray(density, dtype=np.float32) / 255.0
        density_sum = float(density_array.sum())
        if density_sum <= 0:
            raise RuntimeError(f"Empty density target after transform: {row.sample_id}")
        density_array /= density_sum
        fixation_array = (np.asarray(fixation, dtype=np.uint8) > 0).astype(
            np.float32
        )
        if fixation_array.sum() <= 0:
            # Extremely tight random crops can remove every discrete point even
            # while the smoothed density has support. Fall back to its peak.
            y, x = np.unravel_index(density_array.argmax(), density_array.shape)
            fixation_array[y, x] = 1.0
        return {
            "image": image,
            "density": torch.from_numpy(density_array).unsqueeze(0),
            "fixation": torch.from_numpy(fixation_array).unsqueeze(0),
            "sample_index": row.sample_index,
            "sample_id": row.sample_id,
            "image_id": row.image_id,
            "image_path": str(row.image_path),
        }


def collate_salicon(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [row["image"] for row in rows],
        "density": torch.stack([row["density"] for row in rows]),
        "fixation": torch.stack([row["fixation"] for row in rows]),
        "sample_indices": torch.tensor(
            [row["sample_index"] for row in rows], dtype=torch.long
        ),
        "sample_ids": [row["sample_id"] for row in rows],
        "image_ids": [row["image_id"] for row in rows],
        "image_paths": [row["image_path"] for row in rows],
    }


def _aligned_transform(
    image: Image.Image,
    density: Image.Image,
    fixation: Image.Image,
    settings: Any,
    *,
    training: bool,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    if training and settings.augmentation_enabled:
        scale = random.uniform(settings.crop_scale_min, 1.0)
        left = random.uniform(0.0, 1.0 - scale)
        top = random.uniform(0.0, 1.0 - scale)
        image = _normalized_crop(image, left, top, scale)
        density = _normalized_crop(density, left, top, scale)
        fixation = _normalized_crop(fixation, left, top, scale)
        if random.random() < settings.horizontal_flip_probability:
            image = ImageOps.mirror(image)
            density = ImageOps.mirror(density)
            fixation = ImageOps.mirror(fixation)
        if settings.brightness_jitter:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(
                    1.0 - settings.brightness_jitter,
                    1.0 + settings.brightness_jitter,
                )
            )
        if settings.contrast_jitter:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(
                    1.0 - settings.contrast_jitter,
                    1.0 + settings.contrast_jitter,
                )
            )
    size = (settings.image_size, settings.image_size)
    return (
        image.resize(size, Image.Resampling.BICUBIC),
        density.resize(size, Image.Resampling.BILINEAR),
        fixation.resize(size, Image.Resampling.NEAREST),
    )


def _normalized_crop(
    image: Image.Image, left: float, top: float, scale: float
) -> Image.Image:
    x0 = round(left * image.width)
    y0 = round(top * image.height)
    x1 = max(x0 + 1, round((left + scale) * image.width))
    y1 = max(y0 + 1, round((top + scale) * image.height))
    return image.crop((x0, y0, min(image.width, x1), min(image.height, y1)))


def _materialize_maps(
    *,
    width: int,
    height: int,
    points: Sequence[tuple[int, int]],
    density_path: Path,
    fixation_path: Path,
    output_size: int,
    sigma_px: float,
) -> None:
    if not points:
        raise RuntimeError(
            f"SALICON annotation contains no fixation points for {density_path.stem}"
        )
    source = np.zeros((height, width), dtype=np.uint8)
    target_fixation = np.zeros((output_size, output_size), dtype=np.uint8)
    for row, column in points:
        y = max(0, min(height - 1, row - 1))
        x = max(0, min(width - 1, column - 1))
        source[y, x] = 255
        target_y = max(0, min(output_size - 1, round(y * (output_size - 1) / max(1, height - 1))))
        target_x = max(0, min(output_size - 1, round(x * (output_size - 1) / max(1, width - 1))))
        target_fixation[target_y, target_x] = 255
    density = Image.fromarray(source, mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(sigma_px))
    )
    density = density.resize(
        (output_size, output_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(density, dtype=np.float32)
    maximum = float(array.max())
    if maximum <= 0:
        raise RuntimeError(f"Gaussian density is empty for {density_path.stem}")
    array = np.round(array / maximum * 255.0).astype(np.uint8)
    density_path.parent.mkdir(parents=True, exist_ok=True)
    fixation_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(density_path)
    Image.fromarray(target_fixation, mode="L").save(fixation_path)


def _ensure_prepared_maps(
    *,
    annotation_path: Path,
    image_rows: Sequence[dict[str, Any]],
    cache_root: Path,
    split: str,
    output_size: int,
    sigma_px: float,
    enabled: bool,
) -> int | None:
    paths = [
        (*_map_paths(cache_root, split, int(row["id"])), row)
        for row in image_rows
    ]
    missing = [
        (density, fixation, row)
        for density, fixation, row in paths
        if not density.is_file() or not fixation.is_file()
    ]
    metadata_path = cache_root / "prepared_maps" / split / "metadata.json"
    cached_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    cache_identity_matches = (
        cached_metadata.get("annotation_size_bytes") == annotation_path.stat().st_size
        and cached_metadata.get("output_size") == output_size
        and float(cached_metadata.get("sigma_px_at_source", -1.0))
        == float(sigma_px)
    )
    if not cache_identity_matches:
        # A different sigma/output resolution/annotation cannot reuse old maps.
        missing = [(density, fixation, row) for density, fixation, row in paths]
    if not missing:
        return int(cached_metadata.get("annotation_rows", 0))
    if not enabled:
        raise FileNotFoundError(
            f"Prepared SALICON maps are missing or incompatible for {split}. "
            "Enable dataset.materialize_density_maps or use a cache with the "
            "same annotation file, image_size, and density_sigma_px."
        )

    # The official fixation JSON is very large.  Keeping Python tuples for all
    # mouse samples can expand it to tens of GB, so stream the annotations and
    # aggregate only the selected images into an on-disk uint8 memmap.
    temporary = cache_root / "prepared_maps" / split / "_fixations.tmp.npy"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    fixation_maps = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(len(missing), output_size, output_size),
    )
    lookup = {
        int(row["id"]): (index, row)
        for index, (_, _, row) in enumerate(missing)
    }
    annotation_rows = 0
    try:
        for annotation in _iter_json_array(annotation_path, "annotations"):
            annotation_rows += 1
            image_id = int(annotation["image_id"])
            selected = lookup.get(image_id)
            if selected is None:
                continue
            index, image_row = selected
            width = int(image_row["width"])
            height = int(image_row["height"])
            fixations = annotation.get("fixations", [])
            if not isinstance(fixations, list):
                raise RuntimeError(
                    f"Invalid fixations for image_id={image_id}: "
                    f"{type(fixations)}"
                )
            for point in fixations:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                source_y = max(0, min(height - 1, int(point[0]) - 1))
                source_x = max(0, min(width - 1, int(point[1]) - 1))
                target_y = max(
                    0,
                    min(
                        output_size - 1,
                        round(
                            source_y
                            * (output_size - 1)
                            / max(1, height - 1)
                        ),
                    ),
                )
                target_x = max(
                    0,
                    min(
                        output_size - 1,
                        round(
                            source_x
                            * (output_size - 1)
                            / max(1, width - 1)
                        ),
                    ),
                )
                fixation_maps[index, target_y, target_x] = 255

        fixation_maps.flush()
        for index, (density_path, fixation_path, row) in enumerate(missing):
            fixation_array = np.asarray(fixation_maps[index])
            if not np.any(fixation_array):
                raise RuntimeError(
                    f"SALICON annotation contains no fixation points for "
                    f"{split}/{int(row['id'])}"
                )
            width = int(row["width"])
            height = int(row["height"])
            sigma_y = max(
                0.5,
                float(sigma_px) * (output_size - 1) / max(1, height - 1),
            )
            sigma_x = max(
                0.5,
                float(sigma_px) * (output_size - 1) / max(1, width - 1),
            )
            density_array = gaussian_filter(
                (fixation_array > 0).astype(np.float32),
                sigma=(sigma_y, sigma_x),
                mode="constant",
            )
            maximum = float(density_array.max())
            if maximum <= 0:
                raise RuntimeError(
                    f"Gaussian density is empty for {split}/{int(row['id'])}"
                )
            density_array = np.round(
                density_array / maximum * 255.0
            ).astype(np.uint8)
            density_path.parent.mkdir(parents=True, exist_ok=True)
            fixation_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(density_array, mode="L").save(density_path)
            Image.fromarray(fixation_array, mode="L").save(fixation_path)
        write_json(
            metadata_path,
            {
                "annotation_path": str(annotation_path),
                "annotation_size_bytes": annotation_path.stat().st_size,
                "annotation_rows": annotation_rows,
                "image_count": len(image_rows),
                "output_size": output_size,
                "sigma_px_at_source": sigma_px,
            },
        )
    finally:
        del fixation_maps
        temporary.unlink(missing_ok=True)
    return annotation_rows


def _iter_json_array(path: Path, key: str):
    """Yield objects from a top-level JSON array without loading the file."""
    marker = f'"{key}"'
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise RuntimeError(f"JSON key {key!r} not found in {path}")
            buffer += chunk
            marker_index = buffer.find(marker)
            if marker_index < 0:
                buffer = buffer[-len(marker) :]
                continue
            array_index = buffer.find("[", marker_index + len(marker))
            while array_index < 0:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise RuntimeError(
                        f"JSON array for key {key!r} is incomplete in {path}"
                    )
                buffer += chunk
                array_index = buffer.find("[", marker_index + len(marker))
            position = array_index + 1
            break

        eof = False
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise RuntimeError(
                        f"Unterminated JSON array {key!r} in {path}"
                    )
                buffer = ""
                position = 0
                chunk = handle.read(chunk_size)
                eof = not chunk
                buffer += chunk
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                buffer = buffer[position:]
                position = 0
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise RuntimeError(
                        f"Invalid/incomplete JSON array {key!r} in {path}"
                    )
                buffer += chunk
                continue
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Expected object in JSON array {key!r}, got {type(value)}"
                )
            yield value
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def _ensure_sources(settings: Any) -> None:
    missing = []
    for split in ("train", "validation"):
        if _find_image_directory(settings.data_root, split) is None:
            missing.append(f"{split} images")
        if not _annotation_path(settings.data_root, split, required=False).is_file():
            missing.append(f"{split} annotations")
    if not missing:
        return
    if not settings.download:
        raise FileNotFoundError(
            f"SALICON sources are incomplete under {settings.data_root}: {missing}. "
            "Enable dataset.download or place official train/val images and fixation JSON."
        )
    settings.data_root.mkdir(parents=True, exist_ok=True)
    downloads = settings.data_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for split, name, url in (
        ("train", "train.zip", settings.train_images_url),
        ("validation", "val.zip", settings.validation_images_url),
    ):
        if _find_image_directory(settings.data_root, split) is not None:
            continue
        archive = downloads / name
        _download(url, archive)
        _safe_extract_zip(archive, settings.data_root / "images")
    annotations = settings.data_root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    train_annotations = annotations / "fixations_train2014.json"
    validation_annotations = annotations / "fixations_val2014.json"
    if not train_annotations.is_file():
        _download(settings.train_annotations_url, train_annotations)
    if not validation_annotations.is_file():
        _download(settings.validation_annotations_url, validation_annotations)
    unresolved = []
    for split in ("train", "validation"):
        if _find_image_directory(settings.data_root, split) is None:
            unresolved.append(f"{split} images")
        if not _annotation_path(
            settings.data_root, split, required=False
        ).is_file():
            unresolved.append(f"{split} annotations")
    if unresolved:
        discovered = [
            str(path.relative_to(settings.data_root))
            for path in settings.data_root.rglob("*")
            if path.is_dir()
        ][:50]
        raise RuntimeError(
            "SALICON download/extraction completed but required sources are "
            f"still unavailable: {unresolved}. Discovered directories: "
            f"{discovered}. Remove an incomplete archive under "
            f"{downloads} and retry."
        )


def _download(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    existing = temporary.stat().st_size if temporary.is_file() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request, timeout=120)
    resumed = existing > 0 and getattr(response, "status", None) == 206
    mode = "ab" if resumed else "wb"
    if existing and not resumed:
        print(
            f"Server ignored HTTP Range for {url}; restarting download.",
            flush=True,
        )
    with response, temporary.open(mode) as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(path)


def _safe_extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in {path}: {member.filename}")
        archive.extractall(destination)


def _annotation_path(
    root: Path, split: str, *, required: bool = True
) -> Path:
    name = (
        "fixations_train2014.json"
        if split == "train"
        else "fixations_val2014.json"
    )
    candidates = (
        root / "annotations" / name,
        root / name,
    )
    for path in candidates:
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(
            f"SALICON {split} annotation JSON not found. Tried: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def _find_image_directory(root: Path, split: str) -> Path | None:
    # The SALICON 2015r1 S3 archives expand to images/train and images/val,
    # while COCO-style installations commonly use train2014 and val2014.
    # The annotation filenames remain COCO_train2014_*.jpg in both layouts.
    folders = (
        ("train2014", "train")
        if split == "train"
        else ("val2014", "val", "validation")
    )
    for folder in folders:
        for candidate in (root / "images" / folder, root / folder):
            if candidate.is_dir() and any(candidate.glob("*.jpg")):
                return candidate
    if root.exists():
        for folder in folders:
            for candidate in root.rglob(folder):
                if candidate.is_dir() and any(candidate.glob("*.jpg")):
                    return candidate
    return None


def _find_image(root: Path, split: str, file_name: str) -> Path:
    directory = _find_image_directory(root, split)
    if directory is None:
        raise FileNotFoundError(f"SALICON {split} image directory is missing")
    candidates = (directory / file_name, root / file_name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"SALICON image {file_name!r} not found. Tried "
        f"{[str(path) for path in candidates]}"
    )


def _map_paths(
    cache_root: Path, split: str, image_id: int
) -> tuple[Path, Path]:
    folder = cache_root / "prepared_maps" / split
    stem = f"{image_id:012d}"
    return folder / f"{stem}_density.png", folder / f"{stem}_fixation.png"
