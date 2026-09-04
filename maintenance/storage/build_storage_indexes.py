from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


OWNER_ROOTS = ("experiments", "FixedFeedbackSFT", "opticalmoe", "opticalmoe_experiments")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    count = 0
    size = 0
    for directory, _, filenames in os.walk(path):
        for filename in filenames:
            candidate = Path(directory, filename)
            try:
                size += candidate.stat().st_size
                count += 1
            except OSError:
                continue
    return count, size


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def storage_rows(root: Path) -> list[dict[str, object]]:
    candidates = list(root.iterdir())
    for owner in OWNER_ROOTS:
        owner_path = root / owner
        if owner_path.is_dir():
            candidates.extend(owner_path.iterdir())
    seen: set[Path] = set()
    rows: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: item.as_posix().lower()):
        if candidate in seen or candidate.name == ".git":
            continue
        seen.add(candidate)
        files, size = tree_stats(candidate)
        rows.append(
            {
                "relative_path": candidate.relative_to(root).as_posix(),
                "kind": "directory" if candidate.is_dir() else "file",
                "file_count": files,
                "bytes": size,
                "gib": f"{size / 1024**3:.6f}",
                "modified_utc": candidate.stat().st_mtime,
            }
        )
    return rows


def file_rows(root: Path, predicate, *, skip_git: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for directory, dirnames, filenames in os.walk(root):
        if skip_git and Path(directory).name == ".git":
            dirnames[:] = []
            continue
        for filename in filenames:
            candidate = Path(directory, filename)
            if not predicate(candidate):
                continue
            stat = candidate.stat()
            rows.append(
                {
                    "relative_path": candidate.relative_to(root).as_posix(),
                    "bytes": stat.st_size,
                    "modified_utc": stat.st_mtime,
                    "sha256": file_sha256(candidate),
                }
            )
    return sorted(rows, key=lambda row: str(row["relative_path"]).lower())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output_dir or Path(__file__).resolve().parent).resolve()

    storage = storage_rows(root)
    transfer_root = root / ".codex_transfer"
    transfer = (
        file_rows(transfer_root, lambda _: True) if transfer_root.is_dir() else []
    )
    for row in transfer:
        row["relative_path"] = ".codex_transfer/" + str(row["relative_path"])
    bundles = file_rows(
        root,
        lambda item: item.suffix.lower() == ".bundle",
        skip_git=False,
    )

    write_csv(
        output / "storage_inventory.csv",
        ["relative_path", "kind", "file_count", "bytes", "gib", "modified_utc"],
        storage,
    )
    write_csv(
        output / "transfer_inventory.csv",
        ["relative_path", "bytes", "modified_utc", "sha256"],
        transfer,
    )
    write_csv(
        output / "bundle_index.csv",
        ["relative_path", "bytes", "modified_utc", "sha256"],
        bundles,
    )
    print(
        {
            "root": str(root),
            "storage_rows": len(storage),
            "transfer_files": len(transfer),
            "bundle_files": len(bundles),
            "output_dir": str(output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
