from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.features import move_inputs, run_visual
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.io_utils import write_json
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.processor_cache import (ProcessorCacheStore,
                                                                                     validate_processor_cache)

from .capture import VisionPatchBypass


FEATURE_SCHEMA_VERSION = 1


class _CachedInputs(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], store: ProcessorCacheStore) -> None:
        self.dataset = dataset
        self.store = store

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.store.get(index)
        return row["pixel_values"], row["image_grid_thw"], float(self.dataset.targets[index]), index


def _collate(batch: list[Any]):
    pixels, grids, targets, indices = zip(*batch)
    return ({"pixel_values": torch.cat(pixels).float(), "image_grid_thw": torch.stack(grids)},
            torch.tensor(targets, dtype=torch.float32), torch.tensor(indices, dtype=torch.long))


def load_processor_stores(settings: Any, data: Any, source_settings: Any) -> dict[str, ProcessorCacheStore]:
    stores = {
        split: ProcessorCacheStore(settings.source_output_dir / "processor_cache" / f"{split}.pt", 8)
        for split in ("train", "test")
    }
    for split, dataset in (("train", data.train), ("test", data.test)):
        validate_processor_cache(stores[split], split, len(dataset), source_settings)
    return stores


@torch.inference_mode()
def extract_patch_features(split: str, model: torch.nn.Module, capture: VisionPatchBypass,
                           dataset: Dataset[Any], store: ProcessorCacheStore,
                           settings: Any, device: torch.device) -> Path:
    output = settings.output_dir / "features" / f"{split}_patch_hidden.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {
        "cache_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset": "spaq_mos",
        "task": "MOS",
        "split": split,
        "sample_count": len(dataset),
        "model_id": settings.model_id,
        "source_processor_cache": str(settings.source_output_dir / "processor_cache" / f"{split}.pt"),
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "feature_source": "frozen_qwen_visual_patch_embedding_before_transformer_blocks",
        "pooling": "valid_token_mean",
        "vision_transformer_used": False,
        "optical_moe_used": False,
        "language_model_used": False,
    }
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        changed = [key for key, value in expected.items() if payload["metadata"].get(key) != value]
        if changed:
            raise RuntimeError(f"Patch feature cache mismatch for {split}: {changed}. Delete {output} and rebuild it.")
        print(f"[extract_features] validated existing {output}", flush=True)
        return output

    loader = DataLoader(_CachedInputs(dataset, store), batch_size=settings.feature_batch_size,
                        shuffle=False, num_workers=0, collate_fn=_collate, pin_memory=torch.cuda.is_available())
    capture.activate(); model.requires_grad_(False).eval()
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    indices_all: list[torch.Tensor] = []
    token_counts: list[int] = []
    started = time.perf_counter()
    for batch_index, (inputs, batch_targets, indices) in enumerate(loader, start=1):
        run_visual(model, move_inputs(inputs, device))
        hidden = capture.capture.last_hidden
        counts = capture.capture.last_token_counts
        if hidden is None or len(counts) != len(batch_targets):
            raise RuntimeError("Patch-hidden capture failed to preserve per-image boundaries")
        groups = hidden.split(counts, dim=0)
        features.extend(group.float().mean(0).half().cpu() for group in groups)
        targets.append(batch_targets)
        indices_all.append(indices)
        token_counts.extend(counts)
        if batch_index % 100 == 0 or batch_index == len(loader):
            print(f"[extract_features] {split} batch={batch_index}/{len(loader)} cached={len(features)}/{len(dataset)}", flush=True)
    metadata = {
        **expected,
        "feature_dim": int(features[0].numel()),
        "storage_dtype": "float16",
        "elapsed_sec": time.perf_counter() - started,
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_count_mean": sum(token_counts) / len(token_counts),
    }
    torch.save({"metadata": metadata, "features": torch.stack(features), "targets": torch.cat(targets),
                "sample_indices": torch.cat(indices_all), "visual_token_counts": torch.tensor(token_counts)}, output)
    write_json(settings.output_dir / "metrics" / f"feature_extraction_{split}.json", metadata)
    return output


def load_feature_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Patch feature cache missing: {path}. Run --phase extract_features first.")
    return torch.load(path, map_location="cpu", weights_only=True)

