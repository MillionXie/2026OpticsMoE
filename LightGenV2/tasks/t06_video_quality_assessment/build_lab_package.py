"""Build the current Temporal-36 hardware-control and fine-tuning ZIP."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .project import CURRENT_PROFILE, REPO_ROOT, TASK_DIR, load_profile, repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=CURRENT_PROFILE)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    backend = profile["backend"]
    checkpoint = (
        repo_path(profile["artifacts"]["canonical_checkpoint"])
        if args.checkpoint is None
        else Path(args.checkpoint).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Canonical checkpoint is not present on this machine: {checkpoint}\n"
            "Use --checkpoint or run this command on the source training server."
        )
    output = (
        TASK_DIR
        / "releases"
        / f"{datetime.now().strftime('%Y%m%d')}_temporal36_balanced_full_lab.zip"
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    command = [
        sys.executable,
        "-m",
        f"{backend['package']}.build_delivery_packages",
        "lab",
        "--repo-root",
        str(REPO_ROOT),
        "--config",
        str(repo_path(backend["config"])),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--guide",
        str(repo_path(backend["lab_guide"])),
    ]
    print(subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
