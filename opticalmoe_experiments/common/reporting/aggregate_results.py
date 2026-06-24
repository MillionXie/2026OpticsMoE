import json
from pathlib import Path
from typing import Dict, List, Union

PathLike = Union[str, Path]

from .metrics_writer import write_rows


MASTER_FILES = {
    "runs": "master_runs.csv",
    "epoch_metrics": "master_epoch_metrics.csv",
    "final_metrics": "master_final_metrics.csv",
    "per_class_metrics": "master_per_class_metrics.csv",
    "expert_usage": "master_expert_usage.csv",
    "expert_ablation": "master_expert_ablation.csv",
    "optical_energy": "master_optical_energy.csv",
    "model_params": "master_model_params.csv",
}


def _load_json(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_summary_rows(runs_dir: PathLike) -> Dict[str, List[Dict]]:
    runs_dir = Path(runs_dir)
    buckets = {key: [] for key in MASTER_FILES}
    for summary_dir in sorted(runs_dir.glob("*/summary_for_master")):
        for key in buckets:
            path = summary_dir / f"{key}_rows.json"
            payload = _load_json(path)
            if isinstance(payload, dict):
                buckets[key].append(payload)
            elif isinstance(payload, list):
                buckets[key].extend(payload)
    return buckets


def rebuild_master_tables(runs_dir: PathLike, out_dir: PathLike) -> Dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = collect_summary_rows(runs_dir)
    counts = {}
    for key, rows in buckets.items():
        write_rows(out_dir / MASTER_FILES[key], rows)
        counts[key] = len(rows)
    return counts
