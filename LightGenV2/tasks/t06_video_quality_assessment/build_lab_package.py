"""Build a self-contained T06 hardware-control and fine-tuning ZIP."""

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
    parser.add_argument("--bench", choices=("legacy", "shs"), default="legacy")
    parser.add_argument("--target", choices=("spatial", "temporal"))
    parser.add_argument("--source-root", default=str(REPO_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-fields", type=int, default=0)
    args = parser.parse_args()
    if args.bench == "shs":
        if args.target is None or args.output is None:
            parser.error("SHS requires --target and --output (new directory; ZIP is adjacent)")
        from .lab_bundle import build
        build(args)
        return 0
    profile = load_profile(args.profile)
    backend = profile["backend"]
    artifact = profile["artifacts"]
    checkpoint = (
        repo_path(artifact.get("canonical_checkpoint", artifact["checkpoint"]))
        if args.checkpoint is None
        else Path(args.checkpoint).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Canonical checkpoint is not present on this machine: {checkpoint}\n"
            "Use --checkpoint or run this command on the source training server."
        )
    safe_profile = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.profile
    )
    output = (
        TASK_DIR
        / "releases"
        / f"{datetime.now().strftime('%Y%m%d')}_{safe_profile}_full_lab.zip"
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
