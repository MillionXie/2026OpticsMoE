"""T06 six-pass laboratory wrapper using the current verified backend."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .project import CURRENT_PROFILE, REPO_ROOT, TASK_DIR, load_profile, repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("export-pass", "validate-capture", "finetune", "evaluate")
    )
    parser.add_argument("--profile", default=CURRENT_PROFILE)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument(
        "--optical-pass",
        choices=(
            "vision_router",
            "vision_expert",
            "vision_global",
            "language_router",
            "language_expert",
            "language_global",
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("vision_expert", "vision_global", "language_expert", "language_global"),
    )
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lab-config", default="experiments/lab_lgvq/LAB_CONFIG.yaml")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    backend = profile["backend"]
    checkpoint = (
        repo_path(profile["artifacts"]["canonical_checkpoint"])
        if args.checkpoint is None
        else Path(args.checkpoint).expanduser().resolve()
    )
    session = (
        TASK_DIR
        / "runs"
        / "hardware"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.profile}"
        if args.session_dir is None
        else Path(args.session_dir).expanduser().resolve()
    )
    command = [
        sys.executable,
        "-m",
        f"{backend['package']}.hardware_bridge",
        args.action,
        "--config",
        str(repo_path(backend["config"])),
        "--checkpoint",
        str(checkpoint),
        "--session-dir",
        str(session),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--lab-config",
        str(Path(args.lab_config)),
    ]
    if args.optical_pass:
        command.extend(["--optical-pass", args.optical_pass])
    if args.stage:
        command.extend(["--stage", args.stage])
    if args.all_data:
        command.append("--all-data")
    else:
        command.extend(["--max-train", str(args.max_train), "--max-test", str(args.max_test)])
    if args.action == "finetune":
        command.extend(
            ["--epochs", str(args.epochs), "--test-interval", str(args.test_interval)]
        )
    print(subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
