"""Prepare fixed-view standard-chair pairs for text-directed turntable synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dataset import validate_split_contract, write_manifest
from .prepare_abo_cleanrender import (
    ARCHIVE_URL, LICENSE, SOURCE_URL, _archive, _download_remote_jobs,
    _listings, _render_index, _save_render,
)


VIEWS = tuple(range(0, 30, 3))


def _english(row: dict, key: str) -> str:
    return next((str(x.get("value", "")) for x in row.get(key, []) if x.get("language_tag") == "en_US"), "")


def prepare(abo_root: Path, allowlist: Path, output_dir: Path, *, image_size: int = 128, workers: int = 16) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    selected = json.loads(allowlist.read_text(encoding="utf-8"))
    wanted = {item for split in ("train", "val", "test") for item in selected[split]}
    listing = {str(row.get("item_id", "")): row for row in _listings(abo_root) if str(row.get("item_id", "")) in wanted}
    manifests = {split: [] for split in ("train", "val", "test")}; jobs = []
    with _archive(ARCHIVE_URL) as archive:
        render_index = _render_index(archive.namelist())
        info = {value.filename: value for value in archive.infolist()}
        for split in manifests:
            for item_id in selected[split]:
                missing = set(VIEWS) - set(render_index.get(item_id, {}))
                if missing:
                    raise ValueError(f"{item_id} lacks canonical views {sorted(missing)}")
                row = listing[item_id]
                title = _english(row, "item_name")
                for view in VIEWS:
                    member = render_index[item_id][view]
                    relative = Path("images") / split / "chair" / f"{item_id}_{view:02d}.jpg"
                    jobs.append((member, output_dir / relative))
                    manifests[split].append({
                        "sample_id": f"chair-{item_id}-{view:02d}", "sequence_id": item_id,
                        "category": "chair", "caption": "a chair on a clean white studio background",
                        "image_path": relative.as_posix(), "license": LICENSE, "source_url": SOURCE_URL,
                        "source_member": member, "source_title": title, "view_index": view,
                    })
    _download_remote_jobs(ARCHIVE_URL, info, jobs, image_size, workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items(): write_manifest(output_dir / f"{split}.jsonl", rows)
    license_path = abo_root / "LICENSE-CC-BY-4.0.txt"
    (output_dir / "LICENSE-CC-BY-4.0.txt").write_bytes(license_path.read_bytes())
    summary = validate_split_contract(output_dir)
    summary.update({
        "task": "fixed-elevation turntable view synthesis", "canonical_views": list(VIEWS),
        "views_per_identity": len(VIEWS), "image_size": image_size,
        "source_license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
    })
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abo-root", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(prepare(args.abo_root, args.allowlist, args.output_dir, image_size=args.image_size, workers=args.workers), indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
