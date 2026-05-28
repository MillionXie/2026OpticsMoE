import json
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def copy_config(config_path: str, run_dir: str) -> None:
    dst = Path(run_dir) / "config.yaml"
    shutil.copyfile(config_path, dst)
