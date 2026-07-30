from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.datasets import (
    DatasetBundle,
    FSSSaliencyDataset,
    collate_saliency,
    prepare_fss1000,
)


def build_loaders(
    bundle: DatasetBundle,
    settings: Any,
) -> tuple[DataLoader, DataLoader]:
    common: dict[str, Any] = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
        "collate_fn": collate_saliency,
    }
    if settings.num_workers > 0:
        common["prefetch_factor"] = 2
    train = DataLoader(
        FSSSaliencyDataset(bundle.train_records, settings, training=True),
        batch_size=settings.student_batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    test = DataLoader(
        FSSSaliencyDataset(bundle.test_records, settings, training=False),
        batch_size=settings.inference_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train, test


__all__ = ["DatasetBundle", "build_loaders", "prepare_fss1000"]

