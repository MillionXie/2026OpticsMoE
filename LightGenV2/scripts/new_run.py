"""Create a non-destructive standard LightGenV2 run directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="For example t06_video_quality_assessment")
    parser.add_argument("--kind", choices=("smoke", "simulation", "hardware"), required=True)
    parser.add_argument("--name", required=True, help="Short semantic name, not final2/new_new")
    args = parser.parse_args()

    root = _repo_root()
    task = root / "LightGenV2" / "tasks" / args.task
    if not (task / "README.md").is_file():
        raise FileNotFoundError(f"Unknown task: {task}")
    safe_name = "_".join(args.name.strip().replace("-", "_").split())
    if not safe_name or any(mark in safe_name.lower() for mark in ("final_final", "new_new")):
        raise ValueError("Use a short semantic run name")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + safe_name
    destination = task / "runs" / args.kind / run_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "task": args.task,
        "kind": args.kind,
        "status": "created",
        "git_commit": _git_commit(root),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
