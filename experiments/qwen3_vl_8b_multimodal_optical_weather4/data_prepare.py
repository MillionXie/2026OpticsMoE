from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


WEATHER4_CLASSES = ("clear", "rainy", "snowy", "foggy")
BDD100K_ASSETS = {
    "kaggle_bdd100k": (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "awsaf49/bdd100k-dataset?datasetVersionNumber=1"
    ),
}


def ensure_weather4_dataset(
    root: Path,
    train_name: str = "train",
    test_name: str = "test",
) -> dict[str, Any]:
    """Download BDD100K and create the four-class ImageFolder dataset.

    BDD100K's labelled validation split is used as this experiment's test split.
    Original images are kept under ``_raw`` and linked into class directories to
    avoid duplicating several gigabytes of image data.
    """

    root = root.resolve()
    if _prepared(root / train_name) and _prepared(root / test_name):
        return _read_existing_manifest(root, train_name, test_name)

    downloads = root / "_downloads"
    raw = root / "_raw"
    downloads.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    for asset, url in BDD100K_ASSETS.items():
        # Do not derive the filename from the URL: the Kaggle endpoint contains
        # query parameters and redirects to a signed Google Storage URL.
        archive = downloads / f"{asset}.zip"
        _download_with_resume(url, archive)
        _extract_once(archive, raw, raw / f".{asset}.extracted")

    train_images = _find_image_split(raw, "train")
    val_images = _find_image_split(raw, "val")
    train_labels = _find_label_file(raw, "train")
    val_labels = _find_label_file(raw, "val")
    train_stats = prepare_weather_split(
        train_images, train_labels, root / train_name
    )
    test_stats = prepare_weather_split(val_images, val_labels, root / test_name)
    manifest = {
        "dataset": "bdd100k_weather4",
        "source": BDD100K_ASSETS,
        "source_train_split": "train",
        "source_test_split": "val",
        "train": train_stats,
        "test": test_stats,
        "storage": "symlink_or_hardlink_with_copy_fallback",
    }
    _write_json(root / "weather4_manifest.json", manifest)
    return manifest


def prepare_weather_split(
    images_dir: Path,
    labels_file: Path,
    destination: Path,
) -> dict[str, Any]:
    with labels_file.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in BDD100K labels: {labels_file}")

    counts = {name: 0 for name in WEATHER4_CLASSES}
    ignored = 0
    missing_images = 0
    for class_name in WEATHER4_CLASSES:
        (destination / class_name).mkdir(parents=True, exist_ok=True)

    for record in records:
        attributes = record.get("attributes") or {}
        weather = str(attributes.get("weather", "")).strip().lower()
        if weather not in counts:
            ignored += 1
            continue
        image_name = Path(str(record.get("name", ""))).name
        source = images_dir / image_name
        if not image_name or not source.is_file():
            missing_images += 1
            continue
        target = destination / weather / image_name
        _link_without_duplicate(source, target)
        counts[weather] += 1

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(
            f"Prepared split {destination} has no samples for: {', '.join(empty)}. "
            f"Labels used: {labels_file}"
        )
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "ignored_non_weather4": ignored,
        "missing_images": missing_images,
        "images_dir": str(images_dir),
        "labels_file": str(labels_file),
    }


def _download_with_resume(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"[dataset] archive ready: {destination}", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "2026OpticsMoE/BDD100K-preparer"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(
        f"[dataset] downloading {url}"
        + (f" (resume at {offset / 1024**3:.2f} GiB)" if offset else ""),
        flush=True,
    )
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial.exists():
            partial.replace(destination)
            return
        raise RuntimeError(f"BDD100K download failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"BDD100K download connection failed for {url}: {exc.reason}. "
            "Check server DNS/network access; rerunning resumes an existing .part file."
        ) from exc

    append = offset > 0 and getattr(response, "status", None) == 206
    if offset and not append:
        offset = 0
    content_length = int(response.headers.get("Content-Length", "0") or 0)
    total = offset + content_length if content_length else 0
    downloaded = offset
    next_report = downloaded + 256 * 1024**2
    with response, partial.open("ab" if append else "wb") as handle:
        while True:
            chunk = response.read(8 * 1024**2)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                suffix = f"/{total / 1024**3:.2f} GiB" if total else " GiB"
                print(
                    f"[dataset] downloaded {downloaded / 1024**3:.2f}{suffix}",
                    flush=True,
                )
                next_report = downloaded + 256 * 1024**2
    if total and downloaded != total:
        raise RuntimeError(
            f"Incomplete BDD100K download for {url}: expected {total}, got {downloaded} bytes. "
            "Run the same command again to resume."
        )
    partial.replace(destination)


def _extract_once(archive: Path, destination: Path, marker: Path) -> None:
    if marker.is_file():
        return
    print(f"[dataset] extracting {archive.name}", flush=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe path in archive {archive}: {member.filename}")
        bundle.extractall(destination)
    marker.write_text(archive.name + "\n", encoding="utf-8")


def _find_image_split(raw: Path, split: str) -> Path:
    preferred_paths = (
        raw / "bdd100k" / "images" / "100k" / split,
        raw / "bdd100k" / "bdd100k" / "images" / "100k" / split,
    )
    for preferred in preferred_paths:
        if preferred.is_dir():
            return preferred
    candidates = [
        path
        for path in raw.rglob(split)
        if path.is_dir()
        and path.parent.name == "100k"
        and any(path.glob("*.jpg"))
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate extracted BDD100K {split} images under {raw}; "
            f"found {len(candidates)} candidates"
        )
    return candidates[0]


def _find_label_file(raw: Path, split: str) -> Path:
    exact_names = (
        f"det_{split}.json",
        f"bdd100k_labels_images_{split}.json",
        f"det_v2_{split}_release.json",
    )
    for name in exact_names:
        matches = list(raw.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find BDD100K {split} image labels under {raw}. "
        f"Expected one of: {', '.join(exact_names)}"
    )


def _link_without_duplicate(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        relative_source = os.path.relpath(source, target.parent)
        target.symlink_to(relative_source)
        return
    except OSError:
        pass
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _prepared(split_dir: Path) -> bool:
    return all(
        (split_dir / name).is_dir()
        and any(path.is_file() for path in (split_dir / name).iterdir())
        for name in WEATHER4_CLASSES
    )


def _read_existing_manifest(root: Path, train_name: str, test_name: str) -> dict[str, Any]:
    manifest_path = root / "weather4_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "dataset": "bdd100k_weather4",
        "source": "preexisting_imagefolder",
        "train": {"counts": _folder_counts(root / train_name)},
        "test": {"counts": _folder_counts(root / test_name)},
    }


def _folder_counts(split_dir: Path) -> dict[str, int]:
    return {
        name: sum(path.is_file() for path in (split_dir / name).iterdir())
        for name in WEATHER4_CLASSES
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
