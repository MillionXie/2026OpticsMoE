from typing import Any, Dict, Optional

import torch


def _pin_memory_value(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "auto":
            return bool(torch.cuda.is_available())
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if value is None:
        return bool(torch.cuda.is_available())
    return bool(value)


def dataloader_kwargs(dataset_cfg: Dict, shuffle: bool = False, seed: Optional[int] = None) -> Dict:
    """Build safe DataLoader kwargs from config.

    `persistent_workers` and `prefetch_factor` are only valid when
    `num_workers > 0`, so this helper omits them for single-process loading.
    """

    num_workers = int(dataset_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(dataset_cfg.get("batch_size", 64)),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": _pin_memory_value(dataset_cfg.get("pin_memory", "auto")),
    }
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(int(seed))
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dataset_cfg.get("persistent_workers", True))
        prefetch = dataset_cfg.get("prefetch_factor", 4)
        if prefetch is not None:
            kwargs["prefetch_factor"] = int(prefetch)
    return kwargs


def apply_smoke_loader_overrides(dataset_cfg: Dict) -> Dict:
    dataset_cfg["num_workers"] = 0
    dataset_cfg["persistent_workers"] = False
    dataset_cfg["prefetch_factor"] = None
    return dataset_cfg
