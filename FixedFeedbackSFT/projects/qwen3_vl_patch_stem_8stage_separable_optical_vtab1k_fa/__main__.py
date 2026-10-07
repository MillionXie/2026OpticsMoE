from __future__ import annotations

import argparse
from pathlib import Path

from .settings import METHODS, TASKS, load_settings
from .training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="P14 P11-to-VTAB-1k fixed-feedback transfer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    settings = load_settings(
        args.config,
        task=args.task,
        method=args.method,
        seed=args.seed,
        output_root=args.output_root,
        smoke=args.smoke,
    )
    run_experiment(settings, resume=not args.no_resume)


if __name__ == "__main__":
    main()
