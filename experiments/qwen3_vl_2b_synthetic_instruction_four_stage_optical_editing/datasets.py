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
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"Dataset manifest is empty: {path}")
    return records


def load_prompt_cache(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt cache does not exist: {path}. Run --phase cache_prompts first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, dict) or not prompts:
        raise RuntimeError("Prompt cache has no prompt tensors")
    return prompts


class SyntheticEditingDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: Path, settings: Settings, prompt_cache: dict[str, torch.Tensor]) -> None:
        self.records = read_manifest(manifest)
        self.root = settings.data_dir
        self.prompt_cache = prompt_cache

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _image(path: Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    @staticmethod
    def _classes(path: Path) -> torch.Tensor:
        return torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.int64).copy())

    @staticmethod
    def _mask(path: Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(array.copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        folder = self.root / record["relative_dir"]
        files = record["files"]
        key = record["prompt_key"]
        if key not in self.prompt_cache:
            raise KeyError(f"Prompt {key} is absent from the Qwen cache")
        return {
            "sample_id": record["sample_id"],
            "task": record["task"],
            "task_index": int(record["task_index"]),
            "instruction": record["instruction"],
            "program": record["program"],
            "source_image": self._image(folder / files["source"]),
            "source_classes": self._classes(folder / files["source_classes"]),
            "target_classes": self._classes(folder / files["target_classes"]),
            "edit_mask": self._mask(folder / files["edit_mask"]),
            "preserve_mask": self._mask(folder / files["preserve_mask"]),
            "prompt_hidden": self.prompt_cache[key].float(),
        }


def collate_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "source_image",
        "source_classes",
        "target_classes",
        "edit_mask",
        "preserve_mask",
    )
    batch: dict[str, Any] = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    batch["task_index"] = torch.tensor([row["task_index"] for row in rows], dtype=torch.long)
    for key in ("sample_id", "task", "instruction", "program", "prompt_hidden"):
        batch[key] = [row[key] for row in rows]
    return batch


__all__ = [
    "SyntheticEditingDataset",
    "collate_samples",
    "load_prompt_cache",
    "read_manifest",
]
