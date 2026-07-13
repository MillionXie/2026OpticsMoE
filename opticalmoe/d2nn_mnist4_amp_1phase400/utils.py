import csv
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import torch
import yaml


BASE_DIR = Path(__file__).resolve().parent


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_rows(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


def make_run_dir(run_name):
    run_dir = BASE_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def git_info():
    try:
        root = BASE_DIR.parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return {"commit": commit, "dirty": bool(status.strip()), "status_short": status}
    except Exception as exc:
        return {"commit": "", "dirty": "", "error": str(exc)}


def environment_info():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def phase_dropout_settings(config):
    cfg = config.get("regularization", {}).get("phase_dropout", {})
    enabled = bool(cfg.get("enabled", False))
    mode = cfg.get("mode", "none") if enabled else "none"
    return {
        "enabled": enabled,
        "mode": mode,
        "p": float(cfg.get("p", 0.0)) if enabled else 0.0,
        "block_size": int(cfg.get("block_size", 8)),
        "batch_shared": bool(cfg.get("batch_shared", True)),
        "start_epoch": int(cfg.get("start_epoch", 0)),
    }


def phase_dropout_active_for_epoch(settings, epoch):
    return bool(
        settings["enabled"]
        and settings["mode"] != "none"
        and settings["p"] > 0.0
        and int(epoch) >= int(settings["start_epoch"])
    )

