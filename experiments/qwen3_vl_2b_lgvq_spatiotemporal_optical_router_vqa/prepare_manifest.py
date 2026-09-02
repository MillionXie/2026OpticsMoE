from __future__ import annotations

import argparse
import csv
import json
import math
import posixpath
import random
from pathlib import Path
from typing import Any


def _flatten_quality_records(value: Any) -> list[dict[str, Any]]:
    """Flatten LGVQ's nested prompt_cls.json without relying on dict order.

    The uploaded release is a two-key mapping whose values are large record
    lists; some derived copies add another mapping level. A leaf is accepted
    only when it carries a video path and both requested MOS values.
    """

    records: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            has_path = any(node.get(key) not in (None, "") for key in ("path", "video_path", "filename"))
            if has_path and "spatial_quality" in node and "temporal_quality" in node:
                records.append(node)
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("path") or record.get("video_path") or record.get("filename")).replace("\\", "/")
        previous = unique.get(key)
        if previous is not None and previous != record:
            raise RuntimeError(f"Conflicting duplicate LGVQ metadata for {key!r}")
        unique[key] = record
    return list(unique.values())


def _normalized_video_key(value: Any, dataset_root: Path) -> str:
    text = str(value).strip().replace("\\", "/")
    root = str(dataset_root.resolve()).replace("\\", "/").rstrip("/")
    if text.lower().startswith(root.lower() + "/"):
        text = text[len(root) + 1 :]
    return posixpath.normpath(text).removeprefix("./")


def _read_mos_by_path(path: Path, dataset_root: Path) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 4:
            raise RuntimeError(f"MOS.txt line {line_number} must have four semicolon fields")
        key = _normalized_video_key(parts[0], dataset_root)
        if key in result:
            raise RuntimeError(f"Duplicate MOS.txt video path {key!r}")
        result[key] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return result


def prepare_manifest(dataset_root: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    metadata_path = dataset_root / "prompt_cls.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = _flatten_quality_records(raw)
    if len(records) != 2808:
        raise RuntimeError(f"Expected 2808 LGVQ rows, got {len(records)}")
    mos_path = dataset_root / "MOS.txt"
    if not mos_path.exists():
        raise FileNotFoundError(f"LGVQ MOS.txt is required for path-keyed cross-check: {mos_path}")
    mos_by_path = _read_mos_by_path(mos_path, dataset_root)
    groups: dict[str, list[dict[str, Any]]] = {}
    canonical: list[dict[str, Any]] = []
    for item in records:
        relative = str(item.get("path") or item.get("video_path") or item.get("filename"))
        sample_id = _normalized_video_key(relative, dataset_root)
        if sample_id not in mos_by_path:
            raise RuntimeError(f"prompt_cls.json path is absent from MOS.txt: {sample_id!r}")
        mos_spatial, mos_temporal, _mos_alignment = mos_by_path[sample_id]
        json_spatial = float(item["spatial_quality"])
        json_temporal = float(item["temporal_quality"])
        if not (
            math.isclose(json_spatial, mos_spatial, rel_tol=0.0, abs_tol=1.0e-6)
            and math.isclose(json_temporal, mos_temporal, rel_tol=0.0, abs_tol=1.0e-6)
        ):
            raise RuntimeError(
                f"MOS mismatch for {sample_id!r}: JSON={(json_spatial, json_temporal)} "
                f"MOS.txt={(mos_spatial, mos_temporal)}"
            )
        video = Path(relative)
        if not video.is_absolute():
            video = dataset_root / video
        group = str(item.get("prompt") or item.get("code") or Path(relative).stem)
        row = {
            "sample_id": sample_id,
            "video_path": str(video.resolve()),
            "group": group,
            "spatial": json_spatial,
            "temporal": json_temporal,
        }
        canonical.append(row)
        groups.setdefault(group, []).append(row)
    if len(groups) != 468 or any(len(rows) != 6 for rows in groups.values()):
        raise RuntimeError("LGVQ must contain 468 prompt groups with six videos each")
    if set(mos_by_path) != {row["sample_id"] for row in canonical}:
        raise RuntimeError("MOS.txt and prompt_cls.json video-path sets differ")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    # The user-requested protocol has no validation split: 375 prompt groups
    # train and the fixed remaining 93 groups test. All six generators for a
    # prompt stay together, so content does not cross the split boundary.
    assignments = {
        name: ("train" if index < 375 else "test")
        for index, name in enumerate(names)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "video_path", "split", "spatial", "temporal"),
        )
        writer.writeheader()
        for row in sorted(canonical, key=lambda value: value["sample_id"]):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "video_path": row["video_path"],
                    "split": assignments[row["group"]],
                    "spatial": row["spatial"],
                    "temporal": row["temporal"],
                }
            )
    return {
        "manifest": str(output.resolve()),
        "rows": len(canonical),
        "prompt_groups": len(groups),
        "counts": {"train": 2250, "validation": 0, "test": 558},
        "alignment_exported": False,
        "mos_crosscheck": "2808/2808 spatial+temporal values matched by normalized path",
        "join_key": "normalized relative video path/sample_id, never row number",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LGVQ 375-group train / 93-group test split")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_manifest(args.dataset_root, args.output, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
