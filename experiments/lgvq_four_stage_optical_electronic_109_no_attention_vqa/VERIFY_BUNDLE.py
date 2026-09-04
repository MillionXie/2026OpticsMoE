"""Verify every payload byte in the extracted LGVQ laboratory bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parent
    # The builder also places a copy at ZIP root for a one-command first check.
    if not (root / "BUNDLE_MANIFEST.json").is_file():
        root = root.parents[1]
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            failures.append(f"size mismatch: {record['path']}")
            continue
        if sha256(path) != record["sha256"]:
            failures.append(f"SHA256 mismatch: {record['path']}")
    if failures:
        print("BUNDLE VERIFICATION FAILED")
        for failure in failures[:30]:
            print(" -", failure)
        return 1
    print(
        "BUNDLE VERIFIED:",
        len(manifest["files"]),
        "files,",
        manifest["total_bytes"],
        "payload bytes",
    )
    print("Checkpoint SHA256:", manifest["checkpoint_sha256"])
    print("Full frame cache included:", manifest["full_frame_cache_included"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

