from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .settings import Settings


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_prompt_cache(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict):
        raise RuntimeError(f"Invalid prompt cache: {path}")
    return prompts


class OpenMojiEditingDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: Path, settings: Settings, prompts: dict[str, torch.Tensor]):
        self.settings = settings
        self.records = read_manifest(manifest)
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample_dir = self.settings.data_dir / record["relative_dir"]
        source = np.asarray(Image.open(sample_dir / "source.png").convert("RGB"), dtype=np.float32)
        key = record["prompt_key"]
        if key not in self.prompts:
            raise KeyError(f"Prompt {key} is absent from cache")
        source_grid = torch.tensor(record["source_grid"], dtype=torch.long)
        target_grid = torch.tensor(record["target_grid"], dtype=torch.long)
        return {
            "sample_id": record["sample_id"],
            "task": record["task"],
            "instruction": record["instruction"],
            "program": record["program"],
            "source_image": torch.from_numpy(source).permute(2, 0, 1).div(255.0),
            "source_grid": source_grid,
            "target_grid": target_grid,
            "edit_grid": source_grid.ne(target_grid).float(),
            "preserve_grid": source_grid.eq(target_grid).float(),
            "task_index": torch.tensor(record["task_index"], dtype=torch.long),
            "prompt_hidden": self.prompts[key].float(),
        }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "source_image",
        "source_grid",
        "target_grid",
        "edit_grid",
        "preserve_grid",
        "task_index",
    )
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in ("sample_id", "task", "instruction", "program", "prompt_hidden"):
        batch[key] = [sample[key] for sample in samples]
    return batch


__all__ = ["OpenMojiEditingDataset", "collate_samples", "load_prompt_cache", "read_manifest"]

