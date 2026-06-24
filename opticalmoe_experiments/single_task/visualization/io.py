import csv
import json
from pathlib import Path


MASTER_TABLE_FILES = {
    "runs": "master_runs.csv",
    "epoch_metrics": "master_epoch_metrics.csv",
    "final_metrics": "master_final_metrics.csv",
    "per_class_metrics": "master_per_class_metrics.csv",
    "expert_usage": "master_expert_usage.csv",
    "expert_ablation": "master_expert_ablation.csv",
    "optical_energy": "master_optical_energy.csv",
    "model_params": "master_model_params.csv",
}


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_run_epoch_metrics(run_dir):
    run_dir = Path(run_dir)
    rows = _read_csv_rows(run_dir / "metrics" / "epoch_metrics.csv")
    summary = load_run_summary(run_dir)
    for row in rows:
        row.setdefault("run_id", run_dir.name)
        for key in ("dataset_name", "model_type", "num_experts", "prompt_type", "routing_type"):
            if key in summary and key not in row:
                row[key] = summary[key]
    return rows


def load_run_final_metrics(run_dir):
    run_dir = Path(run_dir)
    payload = _read_json(run_dir / "metrics" / "final_metrics.json", default={})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("run_id", run_dir.name)
    summary = load_run_summary(run_dir)
    for key in ("dataset_name", "model_type", "num_experts", "prompt_type", "routing_type"):
        if key in summary and key not in payload:
            payload[key] = summary[key]
    payload.setdefault("run_dir", str(run_dir))
    return payload


def load_run_summary(run_dir):
    return _read_json(Path(run_dir) / "summary.json", default={})


def load_master_table(master_dir, table_name):
    file_name = MASTER_TABLE_FILES.get(table_name, table_name)
    if not file_name.endswith(".csv"):
        file_name += ".csv"
    return _read_csv_rows(Path(master_dir) / file_name)


def resolve_runs_from_args(args):
    run_dirs = [Path(path) for path in getattr(args, "run_dirs", []) or []]
    if run_dirs:
        return run_dirs
    master_dir = getattr(args, "master_dir", None)
    if not master_dir:
        return []
    rows = load_master_table(master_dir, "runs")
    wanted = set(getattr(args, "run_ids", []) or [])
    resolved = []
    for row in rows:
        if wanted and row.get("run_id") not in wanted:
            continue
        run_dir = row.get("run_dir")
        if run_dir:
            resolved.append(Path(run_dir))
        elif row.get("run_id"):
            resolved.append(Path(master_dir).parent / "runs" / row["run_id"])
    return resolved


def save_plot_data(rows, path):
    _write_csv_rows(path, rows)


def load_epoch_rows_from_args(args):
    run_dirs = resolve_runs_from_args(args)
    if run_dirs:
        rows = []
        for run_dir in run_dirs:
            rows.extend(load_run_epoch_metrics(run_dir))
        return rows
    master_dir = getattr(args, "master_dir", None)
    if master_dir:
        return load_master_table(master_dir, "epoch_metrics")
    return []


def load_final_rows_from_args(args):
    run_dirs = resolve_runs_from_args(args)
    if run_dirs:
        return [load_run_final_metrics(run_dir) for run_dir in run_dirs]
    master_dir = getattr(args, "master_dir", None)
    if master_dir:
        return load_master_table(master_dir, "final_metrics")
    return []


def load_expert_usage_rows_from_args(args):
    run_dirs = resolve_runs_from_args(args)
    if run_dirs:
        rows = []
        for run_dir in run_dirs:
            payload = _read_json(Path(run_dir) / "summary_for_master" / "expert_usage_rows.json", default=[])
            if isinstance(payload, list):
                rows.extend(payload)
        return rows
    master_dir = getattr(args, "master_dir", None)
    if master_dir:
        return load_master_table(master_dir, "expert_usage")
    return []
