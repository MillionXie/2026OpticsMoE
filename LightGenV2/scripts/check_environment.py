"""Fast environment and LightGenV2 task preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("t06",), default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    report: dict[str, object] = {
        "python": sys.version,
        "repo_root": str(repo),
        "modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("numpy", "PIL", "yaml", "torch")
        },
    }
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            report["torch"] = {
                "import_ok": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        except Exception as error:  # DLL/CUDA mismatches must be reported, not hidden.
            report["torch"] = {
                "import_ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    if args.task == "t06":
        from LightGenV2.tasks.t06_video_quality_assessment.project import inspect_profile

        report["t06"] = inspect_profile("temporal36_balanced")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required = report["modules"]
    torch_ok = bool(report.get("torch", {}).get("import_ok", False))
    return 0 if all(required.values()) and torch_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
