from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


def _repository_root_for_direct_or_module_execution() -> Path:
    """Load the shared root helper without assuming a parent depth."""

    if not __package__:
        script = Path(__file__).resolve()
        for candidate in script.parents:
            if (candidate / "FixedFeedbackSFT" / "paths.py").is_file():
                root = str(candidate)
                if root not in sys.path:
                    sys.path.insert(0, root)
                break
        else:
            raise RuntimeError(f"Could not locate repository root above {script}")
    from FixedFeedbackSFT.paths import REPOSITORY_ROOT

    return REPOSITORY_ROOT


REPOSITORY_ROOT = _repository_root_for_direct_or_module_execution()

try:  # Support both ``python -m ...run`` and direct script execution.
    from .settings import add_settings_arguments, load_settings_from_args
except ImportError:  # pragma: no cover - exercised only by direct server invocation
    from settings import add_settings_arguments, load_settings_from_args  # type: ignore[no-redef]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one formal 50-epoch P12 downstream transfer job."
    )
    add_settings_arguments(parser)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="resume an identity-matching last checkpoint (default)",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="do not load an existing last checkpoint",
    )
    parser.set_defaults(resume=True)
    return parser


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _training_entrypoint():
    if __package__:
        training = importlib.import_module(f"{__package__}.training")
    else:  # pragma: no cover - direct script invocation
        repo_root = str(REPOSITORY_ROOT)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        training = importlib.import_module(
            "experiments."
            "qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.training"
        )
    # ``run_training`` is the public name. Keep compatibility with early P12
    # snapshots whose implementation was still named ``run_experiment``.
    return getattr(training, "run_training", training.run_experiment)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = None
    started = time.time()
    try:
        settings = load_settings_from_args(args)
        result = _training_entrypoint()(settings, resume=bool(args.resume))
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise RuntimeError("training returned without a complete result")
        return 0
    except Exception as error:
        trace = traceback.format_exc()
        if settings is not None:
            _write_json_atomic(
                settings.output_dir / "failure.json",
                {
                    "status": "failed",
                    "task": settings.task,
                    "method": settings.method,
                    "seed": settings.seed,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "started_at_unix": started,
                    "failed_at_unix": time.time(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": trace,
                    "resume_requested": bool(args.resume),
                },
            )
        print(trace, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
