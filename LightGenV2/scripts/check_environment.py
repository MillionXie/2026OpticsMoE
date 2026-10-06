"""Fast environment and LightGenV2 task preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def assess_t06_report(report: dict[str, object]) -> dict[str, object]:
    """Fail closed on the identities/paths this inspector actually checks."""
    failures = []
    if report.get("error"):
        failures.append("profile_inspection_failed")
    for key in ("backend_config_present", "canonical_checkpoint_present", "canonical_checkpoint_hash_matches"):
        if report.get(key) is not True:
            failures.append(key)
    integrity = report.get("backend_integrity", {})
    if not integrity:
        failures.append("backend_integrity_missing")
    else:
        failures.extend("backend:" + key for key, value in integrity.items() if value.get("matches") is not True)
    inputs = report.get("resolved_inputs", {})
    for key in ("dataset_root", "manifest", "vision_cache", "language_cache"):
        if inputs.get(key, {}).get("present") is not True:
            failures.append("input:" + key)
    failures.extend("input:" + key for key, value in inputs.items()
                    if key not in {"dataset_root", "manifest", "vision_cache", "language_cache"}
                    and value.get("path") is not None and value.get("present") is not True)
    return {"passed": not failures, "failures": failures,
            "scope": "declared backend/config/checkpoint SHA and listed input paths only; no dataset evaluation, cache semantics, SDK or hardware validation"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("t06",), default=None)
    parser.add_argument("--profile", default=None,
                        help="T06 legacy compatibility profile; default remains temporal36_balanced, not the final 16x4 physical model")
    args = parser.parse_args()
    if args.profile is not None and args.task != "t06":
        parser.error("--profile requires --task t06")
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
        name = args.profile or "temporal36_balanced"
        try:
            from LightGenV2.tasks.t06_video_quality_assessment.project import inspect_profile
            report["t06"] = inspect_profile(name)
        except Exception as error:
            report["t06"] = {"profile": name, "error_type": type(error).__name__, "error": str(error)}
        report["t06_preflight"] = assess_t06_report(report["t06"])
        report["t06_entry_boundary"] = "legacy compatibility inspector; final Spatial/Temporal physical identities are in CURRENT_VERSION_20261004.md"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required = report["modules"]
    torch_ok = bool(report.get("torch", {}).get("import_ok", False))
    task_ok = bool(report.get("t06_preflight", {"passed": True})["passed"])
    return 0 if all(required.values()) and torch_ok and task_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
