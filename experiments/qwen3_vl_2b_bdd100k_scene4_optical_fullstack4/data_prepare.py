from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


SCENE4_CLASSES = ("highway", "city_street", "residential", "other")
SCENE_LABEL_MAP = {
    "highway": "highway",
    "city street": "city_street",
    "residential": "residential",
    "parking lot": "other",
    "tunnel": "other",
    "gas stations": "other",
}
IGNORED_SCENE_LABELS = {"", "undefined"}
BDD100K_URL = "https://www.kaggle.com/api/v1/datasets/download/awsaf49/bdd100k-dataset?datasetVersionNumber=1"


def normalize_scene_label(value: Any) -> str | None:
    return SCENE_LABEL_MAP.get(str(value or "").strip().lower())


def ensure_scene4_dataset(root: Path, train_name: str = "train", test_name: str = "test") -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "scene4_manifest.json"
    report_path = root / "scene4_dataset_report.md"
    if _prepared(root / train_name) and _prepared(root / test_name):
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            train_counts = _counts(root / train_name)
            test_counts = _counts(root / test_name)
            manifest = {
                "dataset": "bdd100k_scene4",
                "source": "preexisting_imagefolder",
                "source_train_split": "train",
                "source_test_split": "val",
                "class_mapping": dict(SCENE_LABEL_MAP),
                "train": {"counts": train_counts, "total": sum(train_counts.values())},
                "test": {"counts": test_counts, "total": sum(test_counts.values())},
            }
            _write_json(manifest_path, manifest)
        _write_report(report_path, manifest)
        return manifest

    raw = _find_existing_raw(root)
    if raw is None:
        downloads = root / "_downloads"
        raw = root / "_raw"
        downloads.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        archive = downloads / "kaggle_bdd100k.zip"
        _download(BDD100K_URL, archive)
        _extract(archive, raw, raw / ".extracted")

    train_stats = prepare_scene_split(_find_images(raw, "train"), _find_labels(raw, "train"), root / train_name)
    test_stats = prepare_scene_split(_find_images(raw, "val"), _find_labels(raw, "val"), root / test_name)
    manifest = {
        "dataset": "bdd100k_scene4",
        "source_train_split": "train",
        "source_test_split": "val",
        "source_raw": str(raw),
        "class_names": list(SCENE4_CLASSES),
        "class_mapping": dict(SCENE_LABEL_MAP),
        "ignored_scene_values": sorted(IGNORED_SCENE_LABELS),
        "train": train_stats,
        "test": test_stats,
        "storage": "symlink_or_hardlink_with_copy_fallback",
        "report_path": str(report_path),
    }
    _write_json(manifest_path, manifest)
    _write_report(report_path, manifest)
    return manifest


def prepare_scene_split(images_dir: Path, labels_file: Path, destination: Path) -> dict[str, Any]:
    records = json.loads(labels_file.read_text(encoding="utf-8"))
    counts = {name: 0 for name in SCENE4_CLASSES}
    source_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    other_source_counts: Counter[str] = Counter()
    missing = 0
    for name in SCENE4_CLASSES:
        (destination / name).mkdir(parents=True, exist_ok=True)
    for record in records:
        source_label = str((record.get("attributes") or {}).get("scene") or "").strip().lower()
        source_counts[source_label or "<empty>"] += 1
        normalized = normalize_scene_label(source_label)
        if normalized is None:
            ignored_counts[source_label or "<empty>"] += 1
            continue
        image_name = Path(str(record.get("name", ""))).name
        source = images_dir / image_name
        if not image_name or not source.is_file():
            missing += 1
            continue
        _link(source, destination / normalized / image_name)
        counts[normalized] += 1
        if normalized == "other":
            other_source_counts[source_label] += 1
    empty = [name for name, value in counts.items() if value == 0]
    if empty:
        raise RuntimeError(f"No samples prepared for {empty} from {labels_file}")
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "source_scene_counts": dict(sorted(source_counts.items())),
        "other_source_counts": dict(sorted(other_source_counts.items())),
        "ignored_scene_labels": dict(sorted(ignored_counts.items())),
        "ignored_total": sum(ignored_counts.values()),
        "missing_images": missing,
        "images_dir": str(images_dir),
        "labels_file": str(labels_file),
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    mapping_rows = "\n".join(f"| `{source}` | `{target}` |" for source, target in manifest.get("class_mapping", SCENE_LABEL_MAP).items())
    sections = [
        "# BDD100K Scene-4 Dataset Report",
        "",
        "## Definition",
        "",
        "BDD100K `scene` labels are converted to four ImageFolder classes. `undefined`, empty, and unexpected labels are excluded rather than injected into `other`.",
        "",
        "| BDD100K source label | Scene-4 class |",
        "|---|---|",
        mapping_rows,
    ]
    for split in ("train", "test"):
        stats = manifest.get(split, {})
        sections.extend(["", f"## {split.title()} split", "", "| Class | Samples |", "|---|---:|"])
        for name in SCENE4_CLASSES:
            sections.append(f"| `{name}` | {stats.get('counts', {}).get(name, 0)} |")
        sections.extend([
            f"| **Total retained** | **{stats.get('total', 0)}** |",
            "",
            f"Ignored labels: `{json.dumps(stats.get('ignored_scene_labels', {}), ensure_ascii=False)}`  ",
            f"Missing images: `{stats.get('missing_images', 'unknown')}`  ",
            f"Composition of `other`: `{json.dumps(stats.get('other_source_counts', {}), ensure_ascii=False)}`",
        ])
    sections.extend([
        "",
        "## Imbalance warning",
        "",
        "`other` is much smaller than highway/city_street/residential. Training and evaluation must report macro-F1, balanced accuracy, per-class recall, and the confusion matrix in addition to overall accuracy.",
        "",
        f"Storage mode: `{manifest.get('storage', manifest.get('source', 'unknown'))}`.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _find_existing_raw(root: Path) -> Path | None:
    candidates = [root / "_raw"]
    experiments = Path(__file__).resolve().parents[1]
    candidates.extend(sorted(experiments.glob("*/data/bdd100k_weather4/_raw")))
    candidates.extend(sorted(experiments.glob("*/data/bdd100k_timeofday3/_raw")))
    for candidate in candidates:
        if candidate.is_dir():
            try:
                _find_images(candidate, "train")
                _find_images(candidate, "val")
                _find_labels(candidate, "train")
                _find_labels(candidate, "val")
                return candidate
            except FileNotFoundError:
                pass
    return None


def _find_images(raw: Path, split: str) -> Path:
    preferred = (raw / "bdd100k" / "images" / "100k" / split, raw / "bdd100k" / "bdd100k" / "images" / "100k" / split)
    for path in preferred:
        if path.is_dir():
            return path
    candidates = [path for path in raw.rglob(split) if path.is_dir() and path.parent.name == "100k" and any(path.glob("*.jpg"))]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Could not uniquely find BDD100K {split} images under {raw}; found {len(candidates)}")
    return candidates[0]


def _find_labels(raw: Path, split: str) -> Path:
    for name in (f"det_{split}.json", f"bdd100k_labels_images_{split}.json", f"det_v2_{split}_release.json"):
        matches = list(raw.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find BDD100K {split} labels under {raw}")


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    partial = destination.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "2026OpticsMoE/BDD100K-preparer"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"BDD100K download failed: {exc.reason}") from exc
    append = offset > 0 and getattr(response, "status", None) == 206
    with response, partial.open("ab" if append else "wb") as handle:
        downloaded = offset if append else 0
        while True:
            chunk = response.read(8 * 1024**2)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded % (256 * 1024**2) < len(chunk):
                print(f"[dataset] downloaded {downloaded / 1024**3:.2f} GiB", flush=True)
    partial.replace(destination)


def _extract(archive: Path, destination: Path, marker: Path) -> None:
    if marker.is_file():
        return
    base = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if base != target and base not in target.parents:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)
    marker.write_text("ok\n")


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(os.path.relpath(source, target.parent))
        return
    except OSError:
        pass
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _prepared(path: Path) -> bool:
    return all((path / name).is_dir() and any(item.is_file() for item in (path / name).iterdir()) for name in SCENE4_CLASSES)


def _counts(path: Path) -> dict[str, int]:
    return {name: sum(item.is_file() for item in (path / name).iterdir()) for name in SCENE4_CLASSES}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
