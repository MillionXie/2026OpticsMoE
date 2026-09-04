from __future__ import annotations

import argparse
import csv
import json
import math
import posixpath
import random
from pathlib import Path
from typing import Any


def _records(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            has_path = any(node.get(key) not in (None, "") for key in ("path", "video_path", "filename"))
            if has_path and "spatial_quality" in node and "temporal_quality" in node:
                found.append(node)
            else:
                for child in node.values():
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    unique: dict[str, dict[str, Any]] = {}
    for item in found:
        key = str(item.get("path") or item.get("video_path") or item.get("filename")).replace("\\", "/")
        if key in unique and unique[key] != item:
            raise RuntimeError(f"Conflicting duplicate metadata for {key!r}")
        unique[key] = item
    return list(unique.values())


def _key(value: Any, root: Path) -> str:
    text = str(value).strip().replace("\\", "/")
    absolute_root = str(root.resolve()).replace("\\", "/").rstrip("/")
    if text.lower().startswith(absolute_root.lower() + "/"):
        text = text[len(absolute_root) + 1 :]
    return posixpath.normpath(text).removeprefix("./")


def _mos(path: Path, root: Path) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 4:
            raise RuntimeError(f"MOS.txt line {line_number} must have four fields")
        key = _key(parts[0], root)
        if key in result:
            raise RuntimeError(f"Duplicate MOS path {key!r}")
        result[key] = tuple(float(value) for value in parts[1:])
    return result


def prepare_manifest(dataset_root: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    records = _records(json.loads((root / "prompt_cls.json").read_text(encoding="utf-8")))
    if len(records) != 2808:
        raise RuntimeError(f"Expected 2808 LGVQ rows, got {len(records)}")
    mos = _mos(root / "MOS.txt", root)
    groups: dict[str, list[dict[str, Any]]] = {}
    canonical: list[dict[str, Any]] = []
    for item in records:
        relative = str(item.get("path") or item.get("video_path") or item.get("filename"))
        sample_id = _key(relative, root)
        if sample_id not in mos:
            raise RuntimeError(f"Metadata path is absent from MOS.txt: {sample_id}")
        spatial, temporal, _ = mos[sample_id]
        if not math.isclose(spatial, float(item["spatial_quality"]), abs_tol=1.0e-6):
            raise RuntimeError(f"Spatial MOS mismatch for {sample_id}")
        if not math.isclose(temporal, float(item["temporal_quality"]), abs_tol=1.0e-6):
            raise RuntimeError(f"Temporal MOS mismatch for {sample_id}")
        relative_path = Path(relative)
        candidates = (relative_path,) if relative_path.is_absolute() else (root / relative_path, root / "videos" / relative_path)
        existing = [path.resolve() for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(f"Expected one video for {sample_id}; found {existing}")
        group = str(item.get("prompt") or item.get("code") or Path(relative).stem)
        row = {"sample_id": sample_id, "video_path": str(existing[0]), "spatial": spatial, "temporal": temporal, "group": group}
        canonical.append(row)
        groups.setdefault(group, []).append(row)
    if len(groups) != 468 or any(len(group) != 6 for group in groups.values()):
        raise RuntimeError("LGVQ must contain 468 prompt groups of six videos")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    assignments = {name: ("train" if index < 375 else "test") for index, name in enumerate(names)}
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "video_path", "split", "spatial", "temporal"))
        writer.writeheader()
        for row in sorted(canonical, key=lambda item: item["sample_id"]):
            writer.writerow({**{key: row[key] for key in ("sample_id", "video_path", "spatial", "temporal")}, "split": assignments[row["group"]]})
    return {"manifest": str(output), "rows": 2808, "counts": {"train": 2250, "test": 558}, "validation": False, "alignment_exported": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the fixed group-disjoint LGVQ split")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_manifest(args.dataset_root, args.output, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
