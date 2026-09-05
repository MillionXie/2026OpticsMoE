"""Single LightGenV2 entry for T06 simulation, training and evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .project import (
    CURRENT_PROFILE,
    REPO_ROOT,
    TASK_DIR,
    load_profile,
    materialize_launch_config,
    repo_path,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _default_run_dir(profile: str, phase: str) -> Path:
    group = "smoke" if phase == "smoke" else "simulation"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TASK_DIR / "runs" / group / f"{stamp}_{profile}_{phase}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=CURRENT_PROFILE)
    parser.add_argument(
        "--phase", choices=("smoke", "preflight", "train", "evaluate"), required=True
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--initialization-checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else _default_run_dir(args.profile, args.phase)
    )
    initialization = (
        None
        if args.initialization_checkpoint is None
        else Path(args.initialization_checkpoint).expanduser().resolve()
    )
    _, launch_config, package = materialize_launch_config(
        args.profile, run_dir, initialization_checkpoint=initialization
    )
    command = [
        sys.executable,
        "-m",
        package,
        "--config",
        str(launch_config),
        "--phase",
        args.phase,
    ]
    if args.checkpoint is not None:
        command.extend(["--checkpoint", str(Path(args.checkpoint).expanduser().resolve())])
    elif args.phase == "evaluate":
        command.extend(
            [
                "--checkpoint",
                str(repo_path(profile["artifacts"]["canonical_checkpoint"])),
            ]
        )
    manifest = {
        "schema_version": 1,
        "task": "t06_video_quality_assessment",
        "profile": args.profile,
        "phase": args.phase,
        "status": "prepared" if args.dry_run else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "command": command,
        "run_dir": str(run_dir),
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT)
    manifest["status"] = "completed" if completed.returncode == 0 else "failed"
    manifest["return_code"] = completed.returncode
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
