"""Verify the integrity and identity of an extracted teacher release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = "5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "SHA256.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing release manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in sorted(manifest.items()):
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    checkpoint = root / "weights" / "best_checkpoint.pt"
    if checkpoint.is_file() and sha256(checkpoint) != CHECKPOINT_SHA256:
        errors.append("checkpoint identity mismatch")
    if errors:
        raise RuntimeError("Release verification failed:\n" + "\n".join(errors))
    print(json.dumps({"status": "passed", "files": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
