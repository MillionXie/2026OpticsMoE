from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


OWNER_ROOTS = ("experiments", "FixedFeedbackSFT", "opticalmoe", "opticalmoe_experiments")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
PERIODIC_PATTERN = re.compile(r"^(?:epoch|step|iter|iteration)[_-]?\d+", re.IGNORECASE)
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "data", "datasets", "vendor_sdk"}


def find_runs_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    for owner_name in OWNER_ROOTS:
        owner = root / owner_name
        if not owner.is_dir():
            continue
        for directory, dirnames, _ in os.walk(owner):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            current = Path(directory)
            if current.name == "runs":
                roots.append(current)
                dirnames[:] = []
    return sorted(set(roots), key=lambda item: item.as_posix().lower())


def checkpoint_files(directory: Path) -> list[Path]:
    return sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() in CHECKPOINT_SUFFIXES
        ),
        key=lambda item: item.name.lower(),
    )


def candidates(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for runs_root in find_runs_roots(root):
        for directory, dirnames, _ in os.walk(runs_root):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            current = Path(directory)
            files = checkpoint_files(current)
            if not files:
                continue
            lower_stems = [path.stem.lower() for path in files]
            best_paths = [path for path in files if "best" in path.stem.lower()]
            last_paths = [path for path in files if path.stem.lower() in {"last", "latest"}]
            if not best_paths or not last_paths:
                continue
            for path in files:
                stem = path.stem.lower()
                if not PERIODIC_PATTERN.match(stem):
                    continue
                stat = path.stat()
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": stat.st_size,
                        "checkpoint_directory": current.relative_to(root).as_posix(),
                        "preserved_best": "|".join(
                            item.relative_to(root).as_posix() for item in best_paths
                        ),
                        "preserved_last": "|".join(
                            item.relative_to(root).as_posix() for item in last_paths
                        ),
                        "reason": "periodic checkpoint has sibling best and last/latest",
                    }
                )
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "bytes",
        "checkpoint_directory",
        "preserved_best",
        "preserved_last",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively identify periodic checkpoints only when the same "
            "directory contains both best and last/latest checkpoints."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint_prune_manifest.csv",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    rows = candidates(root)
    write_manifest(manifest, rows)
    total = sum(int(row["bytes"]) for row in rows)
    if args.apply:
        for row in rows:
            target = (root / str(row["path"])).resolve()
            target.relative_to(root)
            if target.is_file():
                target.unlink()
    print(
        {
            "mode": "apply" if args.apply else "dry-run",
            "candidate_count": len(rows),
            "candidate_bytes": total,
            "candidate_gib": round(total / 1024**3, 6),
            "manifest": str(manifest),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
