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


def _loader_prefetch_factor(loader) -> Optional[int]:
    if int(getattr(loader, "num_workers", 0)) <= 0:
        return None
    return getattr(loader, "prefetch_factor", None)


def loader_summary_from_loaders(train_loader, val_loader, test_loader, dataset_cfg: Dict) -> Dict:
    """Return a serializable summary of effective split sizes and loader settings."""

    protocol = dataset_cfg.get("sampling_protocol", {}) or {}
    name = str(dataset_cfg.get("name", "")).lower()
    summary = {
        "train_samples": int(len(train_loader.dataset)),
        "val_samples": int(len(val_loader.dataset)),
        "test_samples": int(len(test_loader.dataset)),
        "total_effective_samples": int(len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)),
        "batch_size": int(getattr(train_loader, "batch_size", dataset_cfg.get("batch_size", 64)) or dataset_cfg.get("batch_size", 64)),
        "num_workers": int(getattr(train_loader, "num_workers", dataset_cfg.get("num_workers", 0))),
        "pin_memory": bool(getattr(train_loader, "pin_memory", _pin_memory_value(dataset_cfg.get("pin_memory", "auto")))),
        "persistent_workers": bool(getattr(train_loader, "persistent_workers", False)),
        "prefetch_factor": _loader_prefetch_factor(train_loader),
        "val_split": dataset_cfg.get("val_split"),
        "test_split": dataset_cfg.get("test_split"),
        "max_train_samples": dataset_cfg.get("max_train_samples"),
        "max_val_samples": dataset_cfg.get("max_val_samples"),
        "max_test_samples": dataset_cfg.get("max_test_samples"),
        "sampling_protocol": {
            "enabled": bool(protocol.get("enabled", False)),
            "total_size": protocol.get("total_size"),
            "train_test_ratio": protocol.get("train_test_ratio", [4, 1]),
            "class_balanced": bool(protocol.get("class_balanced", True)),
            "seed_offset": int(protocol.get("seed_offset", 0)),
        },
    }
    if name == "dsprites":
        summary["sampling_protocol"]["class_balanced_effective"] = False
        summary["sampling_protocol"]["balancing_note"] = "multi_label_balancing_not_supported"
    return summary


def print_loader_summary(summary: Dict, prefix: str = "") -> None:
    label = f"{prefix}: " if prefix else ""
    protocol = summary.get("sampling_protocol", {})
    print(
        f"{label}train={summary.get('train_samples')} val={summary.get('val_samples')} "
        f"test={summary.get('test_samples')} total={summary.get('total_effective_samples')} | "
        f"batch_size={summary.get('batch_size')} num_workers={summary.get('num_workers')} "
        f"pin_memory={summary.get('pin_memory')} persistent_workers={summary.get('persistent_workers')} "
        f"prefetch_factor={summary.get('prefetch_factor')} | "
        f"sampling_enabled={protocol.get('enabled')} total_size={protocol.get('total_size')} "
        f"train_test_ratio={protocol.get('train_test_ratio')} class_balanced={protocol.get('class_balanced')}"
    )
