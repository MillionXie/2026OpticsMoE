from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from FixedFeedbackSFT.paths import REPOSITORY_ROOT, RUNS_ROOT


REPO_ROOT = REPOSITORY_ROOT


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _files(directory: Path, *, ignored_prefix: str = "._") -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(ignored_prefix)
        and path.name != ".DS_Store"
    ]


def audit_imagenet(data_root: Path, clip_cache: Path) -> dict[str, object]:
    dataset_root = data_root / "huggingface_cache" / "ILSVRC___imagenet-1k"
    arrow_files = list(dataset_root.rglob("*.arrow"))
    info_paths = list(dataset_root.rglob("dataset_info.json"))
    if len(info_paths) != 1:
        raise RuntimeError(f"Expected one ImageNet dataset_info.json, found {len(info_paths)}")
    dataset_info = json.loads(info_paths[0].read_text(encoding="utf-8"))
    split_counts = {
        name: int(values["num_examples"])
        for name, values in dataset_info.get("splits", {}).items()
    }
    if split_counts.get("train") != 1_281_167 or split_counts.get("validation") != 50_000:
        raise RuntimeError(f"Unexpected ImageNet split counts: {split_counts}")
    reports: dict[str, object] = {
        "data_root": str(data_root),
        "arrow_files": len(arrow_files),
        "arrow_bytes": sum(path.stat().st_size for path in arrow_files),
        "split_counts": split_counts,
    }
    expected = {
        "train": (1_281_167, 4, 512),
        "validation": (50_000, 1, 512),
    }
    cache: dict[str, object] = {}
    for split, (samples, views, feature_dim) in expected.items():
        metadata_path = clip_cache / f"{split}_metadata.json"
        complete_path = clip_cache / f"{split}_complete.npy"
        feature_path = clip_cache / f"{split}_clip_embeddings.npy"
        if not all(path.exists() for path in (metadata_path, complete_path, feature_path)):
            raise FileNotFoundError(f"Incomplete CLIP cache for split={split}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        complete = np.load(complete_path, mmap_mode="r")
        features = np.load(feature_path, mmap_mode="r")
        if metadata.get("status") != "complete":
            raise RuntimeError(f"CLIP cache status is not complete for {split}")
        if tuple(features.shape) != (samples, views, feature_dim):
            raise RuntimeError(
                f"Unexpected {split} feature shape {features.shape}; "
                f"expected {(samples, views, feature_dim)}"
            )
        if tuple(complete.shape) != (samples, views) or not bool(complete.all()):
            raise RuntimeError(f"CLIP completion mask is incomplete for {split}")
        cache[split] = {
            "status": metadata["status"],
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "complete_entries": int(complete.sum()),
            "dataset_digest": metadata.get("dataset_digest"),
            "source_fingerprint": metadata.get("source_fingerprint"),
        }
    if cache["train"]["dataset_digest"] != cache["validation"]["dataset_digest"]:
        raise RuntimeError("Train/validation CLIP caches use different dataset digests")
    reports["clip_cache"] = cache
    reports["ready"] = True
    return reports


def audit_caltech101(root: Path) -> dict[str, object]:
    categories = root / "caltech-101" / "101_ObjectCategories"
    class_dirs = sorted(path for path in categories.iterdir() if path.is_dir())
    object_dirs = [path for path in class_dirs if path.name != "BACKGROUND_Google"]
    images = [path for directory in object_dirs for path in _files(directory)]
    if len(object_dirs) != 101 or not images:
        raise RuntimeError("Caltech-101 layout is incomplete")
    return {
        "root": str(root),
        "object_classes": len(object_dirs),
        "background_excluded": "BACKGROUND_Google" in {path.name for path in class_dirs},
        "images": len(images),
        "protocol": "class-stratified train/validation/test split",
        "ready": True,
    }


def audit_kadid10k(root: Path) -> dict[str, object]:
    metadata_path = root / "dmos.csv"
    images_root = root / "images"
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8-sig")))
    required = {"dist_img", "ref_img", "dmos"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("KADID-10k DMOS metadata is missing required columns")
    missing = [row["dist_img"] for row in rows if not (images_root / row["dist_img"]).is_file()]
    missing_references = [
        name for name in {row["ref_img"] for row in rows} if not (images_root / name).is_file()
    ]
    if missing or missing_references:
        raise RuntimeError(
            f"KADID-10k is missing {len(missing)} distorted and "
            f"{len(missing_references)} reference images"
        )
    references = sorted({row["ref_img"] for row in rows})
    return {
        "root": str(root),
        "distorted_images": len(rows),
        "reference_images": len(references),
        "protocol": "split by reference image; never split distortions of one reference across folds",
        "metrics": ["SRCC", "PLCC"],
        "ready": True,
    }


def audit_isic2016(root: Path) -> dict[str, object]:
    output: dict[str, object] = {"root": str(root)}
    for split, prefix in (("train", "Training"), ("test", "Test")):
        images_root = root / f"ISBI2016_ISIC_Part1_{prefix}_Data"
        masks_root = root / f"ISBI2016_ISIC_Part1_{prefix}_GroundTruth"
        images = _files(images_root)
        masks = _files(masks_root)
        image_ids = {path.stem for path in images}
        mask_ids = {path.stem.removesuffix("_Segmentation") for path in masks}
        if image_ids != mask_ids:
            raise RuntimeError(f"ISIC2016 {split} image/mask pairing is incomplete")
        output[split] = {"images": len(images), "masks": len(masks)}
    output["metrics"] = ["Dice", "IoU"]
    output["ready"] = True
    return output


def build_report(
    *,
    imagenet_root: Path,
    clip_cache: Path,
    caltech_root: Path,
    kadid_root: Path,
    isic_root: Path,
) -> dict[str, object]:
    return {
        "imagenet1k": audit_imagenet(imagenet_root, clip_cache),
        "caltech101": audit_caltech101(caltech_root),
        "kadid10k": audit_kadid10k(kadid_root),
        "isic2016": audit_isic2016(isic_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P06/P07 pretraining and transfer assets")
    parser.add_argument("--imagenet-root", type=Path, default=REPO_ROOT / "data/imagenet1k")
    parser.add_argument(
        "--clip-cache",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill"
            / "runs/optical_mlp_mixer_moe9_imagenet1k_clip_distill/clip_cache"
        ),
    )
    parser.add_argument("--caltech-root", type=Path, default=REPO_ROOT / "data/Caltech101")
    parser.add_argument(
        "--kadid-root", type=Path, default=REPO_ROOT / "data/kadid10k/kadid10k"
    )
    parser.add_argument("--isic-root", type=Path, default=REPO_ROOT / "data/ISIC2016")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            RUNS_ROOT
            / "d2nn_cifar10_high_performance_optical_backbone"
            / "p06_general_backbone/assets.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        imagenet_root=args.imagenet_root,
        clip_cache=args.clip_cache,
        caltech_root=args.caltech_root,
        kadid_root=args.kadid_root,
        isic_root=args.isic_root,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
