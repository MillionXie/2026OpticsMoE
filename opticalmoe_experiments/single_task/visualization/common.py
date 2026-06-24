import argparse
import math
from collections import defaultdict
from pathlib import Path


DATASET_ORDER = ["mnist", "fashionmnist", "kmnist", "emnist", "cifar10"]
MODEL_ORDER = ["general_d2nn", "fixed_route_moe", "learnable_route_moe", "lenet5"]


def add_common_args(parser):
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--master_dir", default=None)
    parser.add_argument("--run_ids", nargs="*", default=[])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--model_types", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--out_dir", default="single_task/figures/custom_plots")
    parser.add_argument("--name", default=None)
    parser.add_argument("--width", type=float, default=6.5)
    parser.add_argument("--height", type=float, default=4.2)
    return parser


def parse_common_args(description=None):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser


def as_float(value, default=float("nan")):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def filter_rows(rows, args):
    run_ids = set(getattr(args, "run_ids", []) or [])
    model_types = set(getattr(args, "model_types", []) or [])
    if getattr(args, "model_type", None):
        model_types.add(args.model_type)
    dataset = getattr(args, "dataset", None)
    filtered = []
    for row in rows:
        if run_ids and row.get("run_id") not in run_ids:
            continue
        if dataset and row.get("dataset_name") != dataset and row.get("dataset") != dataset:
            continue
        if model_types and row.get("model_type") not in model_types:
            continue
        filtered.append(row)
    return filtered


def group_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "")].append(row)
    return groups


def pretty_label(row, fallback=None):
    if fallback:
        return fallback
    parts = []
    dataset = row.get("dataset_name") or row.get("dataset")
    model = row.get("model_type")
    run_id = row.get("run_id")
    if dataset:
        parts.append(str(dataset))
    if model:
        parts.append(str(model))
    if run_id and not parts:
        parts.append(str(run_id))
    elif run_id:
        parts.append(str(run_id))
    return " / ".join(parts) if parts else "run"


def sort_rows(rows):
    def key(row):
        dataset = row.get("dataset_name") or row.get("dataset") or ""
        model = row.get("model_type") or ""
        dataset_rank = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
        model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
        return (dataset_rank, model_rank, row.get("run_id", ""))

    return sorted(rows, key=key)


def ensure_out_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_rows(rows, source):
    if not rows:
        raise SystemExit(f"No plottable rows found from {source}. Check --run_dirs, --master_dir, and filters.")

