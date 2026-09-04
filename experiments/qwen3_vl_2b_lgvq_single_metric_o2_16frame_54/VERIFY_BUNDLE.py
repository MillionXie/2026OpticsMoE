"""Verify every declared file after extracting a delivery ZIP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            failures.append(f"size: {row['path']}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"sha256: {row['path']}")
    if failures:
        print("BUNDLE VERIFICATION FAILED")
        print("\n".join(f" - {value}" for value in failures[:30]))
        return 1
    print(f"BUNDLE VERIFIED: {len(manifest['files'])} files")
    print("purpose:", manifest["purpose"])
    print("checkpoint_sha256:", manifest.get("checkpoint_sha256"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

