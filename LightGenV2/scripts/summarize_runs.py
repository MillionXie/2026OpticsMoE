"""List LightGenV2 runs without loading checkpoints or caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="Task directory name; omit for all tasks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "tasks"
    tasks = [root / args.task] if args.task else sorted(root.glob("t[0-9][0-9]_*"))
    rows = []
    for task in tasks:
        for manifest in sorted((task / "runs").glob("*/*/run_manifest.json")):
            try:
                raw = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = {"status": "invalid_manifest"}
            rows.append(
                {
                    "task": task.name,
                    "run": manifest.parent.name,
                    "kind": manifest.parent.parent.name,
                    "status": raw.get("status", "unknown"),
                    "path": str(manifest.parent),
                }
            )
    if not rows:
        print("No indexed LightGenV2 runs found.")
        return 0
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
