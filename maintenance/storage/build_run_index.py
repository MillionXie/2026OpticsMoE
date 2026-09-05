from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path


OWNER_ROOTS = ("experiments", "FixedFeedbackSFT", "opticalmoe", "opticalmoe_experiments")
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data",
    "datasets",
    "vendor_sdk",
}
SKIP_RUN_NAMES = {"_assets", "_transfer", "_worktrees", "logs", "results"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
METRIC_NAMES = {
    "metrics.json",
    "test_metrics.json",
    "training_summary.json",
    "training_report.json",
    "results.json",
    "summary.json",
    "train_log.csv",
    "test_log.csv",
}
CONFIG_NAMES = {
    "config.json",
    "config.yaml",
    "config.yml",
    "resolved_config.json",
    "resolved_config.yaml",
}


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def find_run_roots(root: Path) -> list[Path]:
    found: list[Path] = []
    central_fa_runs = root / "FixedFeedbackSFT" / "runs"
    for owner_name in OWNER_ROOTS:
        owner = root / owner_name
        if not owner.is_dir():
            continue
        for directory, dirnames, _ in os.walk(owner):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            current = Path(directory)
            if current.name == "runs":
                if current == central_fa_runs:
                    found.extend(
                        child
                        for child in current.iterdir()
                        if child.is_dir() and child.name not in SKIP_RUN_NAMES
                    )
                else:
                    found.append(current)
                dirnames[:] = []
    return sorted(set(found), key=lambda item: item.as_posix().lower())


def is_best_checkpoint(path: Path) -> bool:
    stem = path.stem.lower()
    return "best" in stem or "formal" in stem or "selected" in stem


def summarize_run(root: Path, run_root: Path, run: Path) -> dict[str, object]:
    file_count = 0
    total_bytes = 0
    checkpoint_count = 0
    checkpoint_bytes = 0
    best_checkpoints: list[str] = []
    has_metrics = False
    has_config = False
    latest_mtime = run.stat().st_mtime

    for directory, dirnames, filenames in os.walk(run):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(directory, filename)
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            total_bytes += stat.st_size
            latest_mtime = max(latest_mtime, stat.st_mtime)
            lower_name = filename.lower()
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                checkpoint_count += 1
                checkpoint_bytes += stat.st_size
                if is_best_checkpoint(path):
                    best_checkpoints.append(path.relative_to(run).as_posix())
            has_metrics = has_metrics or lower_name in METRIC_NAMES or "metric" in lower_name
            has_config = has_config or lower_name in CONFIG_NAMES

    if best_checkpoints and has_metrics:
        state = "formal_candidate"
        recommendation = "keep_best_checkpoints_and_metrics; review other checkpoints"
    elif checkpoint_count:
        state = "checkpoint_without_clear_best"
        recommendation = "manual review before pruning"
    elif has_metrics:
        state = "metadata_only"
        recommendation = "keep small metadata; no checkpoint to prune"
    else:
        state = "unclassified"
        recommendation = "manual review"

    relative_run_root = run_root.relative_to(root).as_posix()
    central_fa_runs = root / "FixedFeedbackSFT" / "runs"
    try:
        project_name = run_root.relative_to(central_fa_runs).parts[0]
    except ValueError:
        owner = run_root.parent.relative_to(root).as_posix()
    else:
        owner = f"FixedFeedbackSFT/projects/{project_name}"
    return {
        "owner": owner,
        "runs_root": relative_run_root,
        "run_name": run.name,
        "run_path": run.relative_to(root).as_posix(),
        "file_count": file_count,
        "bytes": total_bytes,
        "gib": f"{total_bytes / 1024**3:.6f}",
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "best_checkpoint_count": len(best_checkpoints),
        "best_checkpoint_paths": "|".join(sorted(best_checkpoints)),
        "has_metrics": str(has_metrics).lower(),
        "has_config": str(has_config).lower(),
        "latest_modified_utc": utc_iso(latest_mtime),
        "classification": state,
        "retention_recommendation": recommendation,
    }


def build_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_root in find_run_roots(root):
        children = sorted(run_root.iterdir(), key=lambda item: item.name.lower())
        for child in children:
            if child.is_dir() and child.name not in SKIP_RUN_NAMES:
                rows.append(summarize_run(root, run_root, child))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index all experiment-owned run directories without modifying them."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "all_run_index.csv",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = build_rows(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "owner",
        "runs_root",
        "run_name",
        "run_path",
        "file_count",
        "bytes",
        "gib",
        "checkpoint_count",
        "checkpoint_bytes",
        "best_checkpoint_count",
        "best_checkpoint_paths",
        "has_metrics",
        "has_config",
        "latest_modified_utc",
        "classification",
        "retention_recommendation",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"root": str(root), "run_count": len(rows), "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
