from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


JOINT_NAMES = (
    "right_ankle", "right_knee", "right_hip", "left_hip", "left_knee",
    "left_ankle", "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist", "neck", "head_top",
)
FLIP_PERMUTATION = (5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13)
SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (9, 12), (12, 13), (2, 8), (3, 9),
)


@dataclass(frozen=True)
class PoseRecord:
    sample_id: str
    source: str
    split: str
    source_index: int
    image_path: Path
    keypoints: np.ndarray
    raw_visibility: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    train: list[PoseRecord]
    test: list[PoseRecord]
    metadata: dict[str, Any]


def normalize_joints_array(value: np.ndarray) -> np.ndarray:
    """Normalize common LSP/LSPET MATLAB layouts to ``[N,14,3]``.

    The low-resolution LSP archive stores ``[3,14,N]`` while the public
    HR-LSPET mirror stores only x/y coordinates as ``[14,2,N]``.  A zero
    third channel is added for the latter; ``coordinates_in_image`` remains
    the authoritative visibility policy used by the formal configuration.
    """
    value = np.asarray(value)
    if value.ndim != 3:
        raise RuntimeError(f"joints.mat must contain a rank-3 array, got {value.shape}")
    if value.shape[0] in (2, 3) and value.shape[1] == 14:
        value = value.transpose(2, 1, 0)
    elif value.shape[0] == 14 and value.shape[1] in (2, 3):
        value = value.transpose(2, 0, 1)
    elif value.shape[1] == 14 and value.shape[2] in (2, 3):
        pass
    elif value.shape[1] in (2, 3) and value.shape[2] == 14:
        value = value.transpose(0, 2, 1)
    else:
        raise RuntimeError(
            f"Cannot identify joint axes in {value.shape}; expected 2 or 3 "
            "coordinate channels and 14 joints"
        )
    if value.shape[-1] == 2:
        value = np.concatenate(
            [value, np.zeros((*value.shape[:-1], 1), dtype=value.dtype)], axis=-1
        )
    return np.asarray(value, dtype=np.float32)


def prepare_lsp(settings: Any, *, persist: bool = True) -> DatasetBundle:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    roots = _discover_roots(settings.data_root)
    if roots is None:
        if not settings.download:
            raise FileNotFoundError(_missing_message(settings.data_root))
        _download_and_extract(
            settings.lsp_urls, settings.data_root, "lsp_dataset.zip", "lsp_dataset"
        )
        _download_and_extract(
            settings.lspet_urls, settings.data_root, "lspet_dataset.zip", "lspet_dataset"
        )
        roots = _discover_roots(settings.data_root)
    if roots is None:
        raise FileNotFoundError(_missing_message(settings.data_root))
    lsp_root, lspet_root = roots
    lsp = _read_source(lsp_root, "lsp", settings.visibility_policy)
    lspet = _read_source(lspet_root, "lspet", settings.visibility_policy)
    if settings.strict_dataset_counts and (
        len(lsp) != 2000 or len(lspet) != settings.lspet_expected_count
    ):
        raise RuntimeError(
            f"Unexpected official dataset sizes: LSP={len(lsp)} (expected 2000), "
            f"LSPET={len(lspet)} (expected {settings.lspet_expected_count}). "
            "The configured HR-LSPET mirror contains the documented 9,428-image "
            "re-annotated subset; the old low-resolution archive contained 10,000. "
            "Set lspet_expected_count to match a deliberately supplied archive."
        )
    # Standard protocol: LSPET extends training, while original LSP keeps its
    # canonical first-1000 train / last-1000 test split.
    train, test = split_standard_protocol(lsp, lspet)
    train = _stable_limit(train, settings.train_limit, settings.random_seed)
    test = _stable_limit(test, settings.test_limit, settings.random_seed + 1)
    train_paths = {r.image_path.resolve() for r in train}
    test_paths = {r.image_path.resolve() for r in test}
    overlap = train_paths & test_paths
    if overlap:
        raise RuntimeError(f"Train/test image leakage detected: {next(iter(overlap))}")
    metadata = {
        "dataset": "HR-LSPET + LSP" if len(lspet) == 9428 else "LSPET + LSP",
        "protocol": (
            f"LSPET_{len(lspet)}_all_plus_LSP_first1000_train__LSP_last1000_test"
        ),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_lspet": sum(r.source == "lspet" for r in train),
        "train_lsp": sum(r.source == "lsp" for r in train),
        "test_lsp": sum(r.source == "lsp" for r in test),
        "lspet_expected_count": settings.lspet_expected_count,
        "joint_names": list(JOINT_NAMES),
        "visibility_policy": settings.visibility_policy,
        "random_seed": settings.random_seed,
        "lsp_root": str(lsp_root),
        "lspet_root": str(lspet_root),
        "annotation_sha256": {
            "lsp": _sha256(_find_joints_mat(lsp_root)),
            "lspet": _sha256(_find_joints_mat(lspet_root)),
        },
    }
    bundle = DatasetBundle(train=train, test=test, metadata=metadata)
    if persist:
        _persist(bundle, settings.output_dir)
    return bundle


def split_standard_protocol(
    lsp: list[PoseRecord], lspet: list[PoseRecord],
) -> tuple[list[PoseRecord], list[PoseRecord]]:
    """LSPET is training-only; original LSP uses its canonical 1000/1000 split."""
    train = [*(_with_split(r, "train") for r in lspet),
             *(_with_split(r, "train") for r in lsp[:1000])]
    test = [_with_split(r, "test") for r in lsp[1000:]]
    return train, test


def _with_split(record: PoseRecord, split: str) -> PoseRecord:
    return PoseRecord(
        sample_id=record.sample_id,
        source=record.source,
        split=split,
        source_index=record.source_index,
        image_path=record.image_path,
        keypoints=record.keypoints,
        raw_visibility=record.raw_visibility,
    )


def _read_source(root: Path, source: str, visibility_policy: str) -> list[PoseRecord]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("Reading official LSP annotations requires scipy") from exc
    joints_path = _find_joints_mat(root)
    raw = loadmat(joints_path)
    candidates = [(k, v) for k, v in raw.items() if not k.startswith("__") and np.asarray(v).ndim == 3]
    if not candidates:
        raise RuntimeError(f"No rank-3 joint array found in {joints_path}; keys={list(raw)}")
    key, array = next(((k, v) for k, v in candidates if k.lower() == "joints"), candidates[0])
    joints = normalize_joints_array(array)
    image_dir = _find_image_dir(root)
    images = sorted(
        (p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=_numeric_sort_key,
    )
    if len(images) != len(joints):
        raise RuntimeError(
            f"{source}: annotation array {key!r} contains {len(joints)} samples but "
            f"{image_dir} contains {len(images)} images"
        )
    records = []
    for index, (image_path, sample) in enumerate(zip(images, joints)):
        coordinates = sample[:, :2].copy()
        # Official .mat coordinates are one-based MATLAB pixels.
        coordinates -= 1.0
        raw_visibility = sample[:, 2].copy()
        if visibility_policy == "third_channel_zero_visible":
            valid = raw_visibility <= 0.5
        elif visibility_policy == "third_channel_one_visible":
            valid = raw_visibility > 0.5
        else:
            valid = np.ones(14, dtype=bool)
        valid &= np.isfinite(coordinates).all(axis=1)
        coordinates[~valid] = np.nan
        records.append(PoseRecord(
            sample_id=f"{source}_{index + 1:05d}",
            source=source,
            split="unassigned",
            source_index=index,
            image_path=image_path.resolve(),
            keypoints=coordinates,
            raw_visibility=raw_visibility,
        ))
    return records


class LSPPoseDataset(Dataset):
    def __init__(self, records: list[PoseRecord], settings: Any, *, training: bool) -> None:
        self.records = records
        self.settings = settings
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        transformed, keypoints, visible, crop = transform_person(
            image,
            record.keypoints,
            image_size=self.settings.image_size,
            crop_margin=self.settings.crop_margin,
            training=self.training and self.settings.augmentation_enabled,
            scale_jitter=self.settings.crop_scale_jitter,
            center_jitter=self.settings.crop_center_jitter,
            flip_probability=self.settings.horizontal_flip_probability,
            brightness_jitter=self.settings.brightness_jitter,
            contrast_jitter=self.settings.contrast_jitter,
        )
        heatmaps = make_heatmaps(
            keypoints, visible, self.settings.image_size,
            self.settings.heatmap_size, self.settings.heatmap_sigma,
        )
        torso_scale, head_scale = pose_scales(keypoints, visible)
        return {
            "image": transformed,
            "heatmaps": heatmaps,
            "keypoints": torch.from_numpy(keypoints),
            "visible": torch.from_numpy(visible),
            "torso_scale": torch.tensor(torso_scale, dtype=torch.float32),
            "head_scale": torch.tensor(head_scale, dtype=torch.float32),
            "sample_id": record.sample_id,
            "source": record.source,
            "image_path": str(record.image_path),
            "crop_box": crop,
        }


def pose_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in items],
        "heatmaps": torch.stack([item["heatmaps"] for item in items]),
        "keypoints": torch.stack([item["keypoints"] for item in items]),
        "visible": torch.stack([item["visible"] for item in items]),
        "torso_scale": torch.stack([item["torso_scale"] for item in items]),
        "head_scale": torch.stack([item["head_scale"] for item in items]),
        "sample_id": [item["sample_id"] for item in items],
        "source": [item["source"] for item in items],
        "image_path": [item["image_path"] for item in items],
        "crop_box": [item["crop_box"] for item in items],
    }


def transform_person(
    image: Image.Image,
    keypoints: np.ndarray,
    *,
    image_size: int,
    crop_margin: float,
    training: bool,
    scale_jitter: float,
    center_jitter: float,
    flip_probability: float,
    brightness_jitter: float,
    contrast_jitter: float,
) -> tuple[Image.Image, np.ndarray, np.ndarray, list[float]]:
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    finite = np.isfinite(keypoints).all(axis=1)
    if not finite.any():
        raise RuntimeError("Pose sample contains no finite joint coordinates")
    xy = keypoints[finite]
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    side = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 32.0) * crop_margin
    if training:
        side *= random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        center += np.asarray([
            random.uniform(-center_jitter, center_jitter) * side,
            random.uniform(-center_jitter, center_jitter) * side,
        ], dtype=np.float32)
    left = float(center[0] - side * 0.5)
    top = float(center[1] - side * 0.5)
    right, bottom = left + side, top + side
    cropped = image.crop((math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)))
    actual_left, actual_top = math.floor(left), math.floor(top)
    scale_x = image_size / cropped.width
    scale_y = image_size / cropped.height
    keypoints[:, 0] = (keypoints[:, 0] - actual_left) * scale_x
    keypoints[:, 1] = (keypoints[:, 1] - actual_top) * scale_y
    transformed = cropped.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if training and random.random() < flip_probability:
        transformed = transformed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        keypoints[:, 0] = image_size - 1 - keypoints[:, 0]
        keypoints = keypoints[np.asarray(FLIP_PERMUTATION)]
        finite = finite[np.asarray(FLIP_PERMUTATION)]
    if training and brightness_jitter > 0:
        transformed = ImageEnhance.Brightness(transformed).enhance(
            random.uniform(1 - brightness_jitter, 1 + brightness_jitter)
        )
    if training and contrast_jitter > 0:
        transformed = ImageEnhance.Contrast(transformed).enhance(
            random.uniform(1 - contrast_jitter, 1 + contrast_jitter)
        )
    visible = finite & (keypoints[:, 0] >= 0) & (keypoints[:, 0] < image_size)
    visible &= (keypoints[:, 1] >= 0) & (keypoints[:, 1] < image_size)
    keypoints[~visible] = np.nan
    return transformed, keypoints.astype(np.float32), visible.astype(bool), [left, top, right, bottom]


def make_heatmaps(
    keypoints: np.ndarray,
    visible: np.ndarray,
    image_size: int,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    grid_y, grid_x = torch.meshgrid(
        torch.arange(heatmap_size, dtype=torch.float32),
        torch.arange(heatmap_size, dtype=torch.float32), indexing="ij",
    )
    result = torch.zeros((len(JOINT_NAMES), heatmap_size, heatmap_size), dtype=torch.float32)
    scale = heatmap_size / image_size
    for index, valid in enumerate(visible):
        if not valid:
            continue
        x, y = keypoints[index] * scale
        result[index] = torch.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))
    return result


def pose_scales(keypoints: np.ndarray, visible: np.ndarray) -> tuple[float, float]:
    def distance(a: int, b: int) -> float:
        if visible[a] and visible[b]:
            return float(np.linalg.norm(keypoints[a] - keypoints[b]))
        return float("nan")
    torso_values = [distance(8, 3), distance(9, 2)]
    torso_values = [v for v in torso_values if math.isfinite(v) and v > 1e-6]
    torso = float(np.mean(torso_values)) if torso_values else float("nan")
    head = distance(12, 13)
    head = 2.0 * head if math.isfinite(head) and head > 1e-6 else float("nan")
    return torso, head


def _discover_roots(data_root: Path) -> tuple[Path, Path] | None:
    directories = [data_root, *(p for p in data_root.rglob("*") if p.is_dir())]
    lsp_candidates, lspet_candidates = [], []
    for directory in directories:
        try:
            count = len(list(_find_image_dir(directory).glob("*")))
            _find_joints_mat(directory)
        except FileNotFoundError:
            continue
        name = directory.name.lower()
        if "lspet" in name or count > 5000:
            lspet_candidates.append(directory)
        elif "lsp" in name or count >= 1000:
            lsp_candidates.append(directory)
    if not lsp_candidates or not lspet_candidates:
        return None
    return lsp_candidates[0], lspet_candidates[0]


def _find_image_dir(root: Path) -> Path:
    for candidate in (root / "images", root / "Images", root):
        if candidate.is_dir() and any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(f"No image directory found below {root}")


def _find_joints_mat(root: Path) -> Path:
    direct = root / "joints.mat"
    if direct.is_file():
        return direct
    matches = list(root.glob("*joints*.mat"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No joints.mat found below {root}")


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def _stable_limit(records: list[PoseRecord], limit: int | None, seed: int) -> list[PoseRecord]:
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise ValueError("Dataset limits must be positive or null")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    return [records[i] for i in sorted(order[:limit])]


def _download_and_extract(
    urls: tuple[str, ...], root: Path, filename: str, extract_subdir: str,
) -> None:
    archive = root / filename
    extraction_root = root / extract_subdir
    completion_marker = extraction_root / ".download_complete"
    if completion_marker.is_file():
        return
    failures = []
    for url in urls:
        try:
            if url.startswith("hf://"):
                _download_huggingface_snapshot(url, extraction_root)
                completion_marker.write_text(
                    f"source={url}\ncompleted_at={time.time()}\n", encoding="utf-8"
                )
                return
            if not archive.is_file() or not zipfile.is_zipfile(archive):
                archive.unlink(missing_ok=True)
                _download_http_resumable(url, archive)
            if not zipfile.is_zipfile(archive):
                raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {archive}")
            extraction_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                root_resolved = extraction_root.resolve()
                members = bundle.infolist()
                if not members:
                    raise RuntimeError(f"ZIP archive contains no files: {archive}")
                for member in members:
                    target = (extraction_root / member.filename).resolve()
                    if root_resolved not in target.parents and target != root_resolved:
                        raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
                print(
                    f"Extracting {len(members):,} files from {archive.name} "
                    f"to {extraction_root}", flush=True,
                )
                bundle.extractall(extraction_root)
            completion_marker.write_text(
                f"source={url}\ncompleted_at={time.time()}\n", encoding="utf-8"
            )
            return
        except Exception as exc:  # every configured mirror is attempted
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            # A completed but invalid archive must not poison the next mirror.
            if archive.is_file() and not zipfile.is_zipfile(archive):
                archive.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download/extract {filename}. Attempts:\n  " + "\n  ".join(failures)
    )


def _download_huggingface_snapshot(url: str, extraction_root: Path) -> None:
    """Download the small original LSP mirror with resumable HF transfers."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face LSP mirror requires huggingface_hub. "
            "Install experiments/.../requirements.txt first."
        ) from exc
    repo_id = url.removeprefix("hf://").strip("/")
    if repo_id.count("/") != 1:
        raise ValueError(f"Invalid Hugging Face dataset source: {url}")
    endpoint = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
    extraction_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading Hugging Face dataset {repo_id} from {endpoint} "
        f"to {extraction_root}", flush=True,
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=extraction_root,
        allow_patterns=["images/*", "joints.mat", "README.md"],
        max_workers=16,
        endpoint=endpoint,
    )
    if not (extraction_root / "joints.mat").is_file():
        raise RuntimeError(f"Hugging Face snapshot lacks joints.mat: {extraction_root}")


def _download_http_resumable(url: str, archive: Path) -> None:
    """Stream an HTTP archive with Range-based resume and visible progress."""
    partial = archive.with_name(archive.name + ".part")
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            total_size = int(response.headers.get("Content-Length", "0"))
            accepts_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception as exc:
        print(f"  HEAD request failed ({exc}); using single-stream resume", flush=True)
        total_size, accepts_ranges = 0, False
    if total_size >= 512 * 1024 * 1024 and accepts_ranges:
        _download_http_parallel(url, archive, partial, total_size, workers=8)
        return

    last_error: Exception | None = None
    for attempt in range(1, 4):
        existing = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            print(
                f"Downloading {url} -> {archive} "
                f"(attempt {attempt}/3, resume={existing:,} bytes)", flush=True,
            )
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", response.getcode())
                append = existing > 0 and status == 206
                if existing > 0 and not append:
                    existing = 0
                response_length = response.headers.get("Content-Length")
                expected = (
                    existing + int(response_length) if response_length is not None else None
                )
                mode = "ab" if append else "wb"
                downloaded = existing
                next_report = downloaded + 256 * 1024 * 1024
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            suffix = f"/{expected:,}" if expected is not None else ""
                            print(
                                f"  downloaded {downloaded:,}{suffix} bytes", flush=True
                            )
                            next_report = downloaded + 256 * 1024 * 1024
                if expected is not None and downloaded != expected:
                    raise RuntimeError(
                        f"Incomplete transfer: received {downloaded:,}/{expected:,} bytes"
                    )
            partial.replace(archive)
            return
        except Exception as exc:
            last_error = exc
            print(f"  download attempt {attempt} failed: {exc}", flush=True)
    raise RuntimeError(f"HTTP download failed after 3 attempts: {last_error}")


def _download_http_parallel(
    url: str, archive: Path, partial: Path, total_size: int, *, workers: int,
) -> None:
    """Download a large byte-range-capable archive in resumable segments."""
    parts_root = archive.with_name(archive.name + ".parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    segment_size = math.ceil(total_size / workers)
    ranges = [
        (index, index * segment_size, min(total_size - 1, (index + 1) * segment_size - 1))
        for index in range(workers)
        if index * segment_size < total_size
    ]

    # Preserve a prefix produced by the former single-stream downloader.
    if partial.is_file() and not any(parts_root.glob("part_*.bin")):
        prefix_size = partial.stat().st_size
        print(
            f"Migrating existing {prefix_size:,}-byte prefix into range segments", flush=True
        )
        with partial.open("rb") as source:
            remaining = prefix_size
            for index, start, end in ranges:
                if remaining <= 0:
                    break
                count = min(remaining, end - start + 1)
                with (parts_root / f"part_{index:03d}.bin").open("wb") as output:
                    to_copy = count
                    while to_copy:
                        chunk = source.read(min(to_copy, 16 * 1024 * 1024))
                        if not chunk:
                            raise RuntimeError(
                                "Existing partial archive ended before its reported size"
                            )
                        output.write(chunk)
                        to_copy -= len(chunk)
                remaining -= count
        partial.unlink()

    print(
        f"Parallel download: {workers} ranges, total={total_size:,} bytes", flush=True
    )

    def fetch_segment(spec: tuple[int, int, int]) -> tuple[int, int]:
        index, start, end = spec
        path = parts_root / f"part_{index:03d}.bin"
        target_size = end - start + 1
        if path.is_file() and path.stat().st_size > target_size:
            path.unlink()
        last_error: Exception | None = None
        for attempt in range(1, 4):
            have = path.stat().st_size if path.is_file() else 0
            if have == target_size:
                return index, target_size
            offset = start + have
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Range": f"bytes={offset}-{end}",
                    },
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    status = getattr(response, "status", response.getcode())
                    if status != 206:
                        raise RuntimeError(
                            f"server ignored Range for segment {index}: HTTP {status}"
                        )
                    with path.open("ab") as output:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                size = path.stat().st_size
                if size != target_size:
                    raise RuntimeError(
                        f"segment {index} incomplete: {size:,}/{target_size:,} bytes"
                    )
                return index, size
            except Exception as exc:
                last_error = exc
                print(
                    f"  segment {index} attempt {attempt}/3 failed: {exc}", flush=True
                )
                time.sleep(attempt)
        raise RuntimeError(f"segment {index} failed: {last_error}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_segment, spec) for spec in ranges]
        for future in as_completed(futures):
            index, size = future.result()
            print(f"  segment {index + 1}/{len(ranges)} complete ({size:,} bytes)", flush=True)

    with partial.open("wb") as output:
        for index, _, _ in ranges:
            part = parts_root / f"part_{index:03d}.bin"
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    if partial.stat().st_size != total_size:
        raise RuntimeError(
            f"Assembled archive size mismatch: {partial.stat().st_size:,}/{total_size:,}"
        )
    partial.replace(archive)
    shutil.rmtree(parts_root)


def _persist(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset.json").write_text(
        json.dumps(bundle.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "data_split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "split", "source_index", "image_path"])
        writer.writeheader()
        for record in [*bundle.train, *bundle.test]:
            writer.writerow({
                "sample_id": record.sample_id, "source": record.source,
                "split": record.split, "source_index": record.source_index,
                "image_path": str(record.image_path),
            })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_message(root: Path) -> str:
    found = [str(p.relative_to(root)) for p in root.rglob("*")][:50] if root.exists() else []
    return (
        f"LSP/LSPET data were not found under {root}. Expected lsp_dataset/images + "
        f"joints.mat and lspet_dataset/images + joints.mat. Found: {found}"
    )
