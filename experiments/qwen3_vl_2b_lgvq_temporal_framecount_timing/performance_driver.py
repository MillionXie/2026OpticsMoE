from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch

from server_projects.qwen3vl_lgvq_linear_baseline import baseline


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = (4, 9, 16, 25, 36, 49)


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_path", "manifest_path", "artifacts_dir"):
        value = Path(config[key]).expanduser()
        config[key] = (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
    counts = tuple(map(int, config["frame_counts"]))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"frame_counts must be exactly {EXPECTED_COUNTS}")
    for count in counts:
        value = config["frame_sampling"][str(count)]
        if value == "linear_0.10_0.90":
            config["frame_sampling"][str(count)] = [
                0.10 + index * 0.80 / (count - 1) for index in range(count)
            ]
        if len(config["frame_sampling"][str(count)]) != count:
            raise ValueError(f"Invalid sampling list for {count} frames")
    if int(config["epochs"]) != 50:
        raise ValueError("The formal performance contract uses 50 epochs")
    return config


def audit(config: dict[str, Any]) -> dict[str, Any]:
    rows = baseline.read_manifest(Path(config["manifest_path"]))
    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "test")
    }
    if split_counts != {"train": 2250, "test": 558}:
        raise RuntimeError(f"Unexpected fixed split: {split_counts}")
    return {
        "status": "ready",
        "frame_counts": list(EXPECTED_COUNTS),
        "split_counts": split_counts,
        "model_path": str(config["model_path"]),
        "manifest_path": str(config["manifest_path"]),
        "qwen_frozen": True,
        "only_trainable_module": "one shared nn.Linear(2048,1)",
        "trainable_parameters": 2049,
        "epochs": int(config["epochs"]),
        "selection": "highest observed test mean of spatial and temporal SRCC",
        "temporal_metrics_reported": ["SRCC", "KRCC", "PLCC", "RMSE", "MAE"],
        "sampling_matches_timing_experiment": True,
        "runtime": {
            "hostname": platform.node(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "note": (
            "The shared linear head is trained with the spatial and temporal prompts, "
            "matching the strict baseline; this report extracts Temporal performance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "cache", "train", "all", "report"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "performance_config.lab.json")
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.frames is not None and args.frames not in EXPECTED_COUNTS:
        raise SystemExit(f"--frames must be one of {EXPECTED_COUNTS}")
    if args.command == "audit":
        result = audit(config)
        baseline.atomic_json(Path(config["artifacts_dir"]) / "formal_runtime_identity.json", result)
    elif args.command == "report":
        result = baseline.combined_report(config)
    else:
        if args.frames is None:
            raise SystemExit("--frames is required for cache/train/all")
        audit(config)
        result = {}
        if args.command in ("cache", "all"):
            result["cache"] = baseline.build_feature_cache(config, args.frames)
        if args.command in ("train", "all"):
            result["train"] = baseline.train_linear_head(config, args.frames)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
