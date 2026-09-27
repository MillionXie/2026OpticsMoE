"""Check immutable teacher runtime provenance (no tensor assets required)."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import sha256


def verify():
    manifest = json.loads((ROOT / "reference" / "source_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Teacher source changed or missing: {name}")
    return {"status": "passed", "files": len(manifest["files"]),
            "reference_srcc": manifest["reference_metrics"]["srcc"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
