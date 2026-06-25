from __future__ import annotations

from typing import Dict, Iterable, Mapping


HEAD_FIELDS = [
    "detector_size",
    "detector_layout",
    "readout_type",
    "normalize_detector_energy",
    "logit_scale",
    "input_norm",
    "norm_affine",
    "hidden_dim",
    "hidden_layers",
    "activation",
    "dropout",
]


def global_head_defaults(config: Mapping) -> Dict:
    detector = dict(config.get("detector", {}) or {})
    readout = dict(config.get("readout", {}) or {})
    return normalize_head_config(
        {
            "detector_size": detector.get("detector_size", 32),
            "detector_layout": detector.get("layout", detector.get("detector_layout", "grid")),
            "readout_type": readout.get("type", readout.get("readout_type", "mlp")),
            "normalize_detector_energy": readout.get("normalize_detector_energy", True),
            "logit_scale": readout.get("logit_scale", 10.0),
            "input_norm": readout.get("input_norm", "layernorm"),
            "norm_affine": readout.get("norm_affine", True),
            "hidden_dim": readout.get("hidden_dim", 64),
            "hidden_layers": readout.get("hidden_layers", 1),
            "activation": readout.get("activation", "gelu"),
            "dropout": readout.get("dropout", 0.1),
        }
    )


def normalize_head_config(head: Mapping, defaults: Mapping | None = None) -> Dict:
    merged = dict(defaults or {})
    merged.update(dict(head or {}))
    if "layout" in merged and "detector_layout" not in merged:
        merged["detector_layout"] = merged["layout"]
    if "type" in merged and "readout_type" not in merged:
        merged["readout_type"] = merged["type"]
    normalized = {
        "detector_size": int(merged.get("detector_size", 32)),
        "detector_layout": str(merged.get("detector_layout", "grid")),
        "readout_type": str(merged.get("readout_type", "mlp")),
        "normalize_detector_energy": bool(merged.get("normalize_detector_energy", True)),
        "logit_scale": float(merged.get("logit_scale", 10.0)),
        "input_norm": str(merged.get("input_norm", "layernorm")),
        "norm_affine": bool(merged.get("norm_affine", True)),
        "hidden_dim": int(merged.get("hidden_dim", 64)),
        "hidden_layers": int(merged.get("hidden_layers", 1)),
        "activation": str(merged.get("activation", "gelu")),
        "dropout": float(merged.get("dropout", 0.1)),
    }
    return normalized


def _validate_task_names(task_names: Iterable[str]) -> list[str]:
    names = [str(name).lower() for name in task_names]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate task names in task head config: {names}")
    return names


def resolve_dataset_switching_task_heads(config: Mapping, task_names: Iterable[str]) -> Dict[str, Dict]:
    names = _validate_task_names(task_names)
    defaults = global_head_defaults(config)
    task_cfgs = {
        str(task.get("name", "")).lower(): dict(task.get("head", {}) or {})
        for task in config.get("training", {}).get("multitask", {}).get("tasks", [])
        if isinstance(task, Mapping)
    }
    unknown = sorted(set(task_cfgs) - set(names))
    if unknown:
        raise ValueError(f"Head configs contain unknown tasks: {unknown}; valid tasks: {names}")
    return {name: normalize_head_config(task_cfgs.get(name, {}), defaults) for name in names}


def resolve_same_input_task_heads(config: Mapping, task_names: Iterable[str]) -> Dict[str, Dict]:
    names = _validate_task_names(task_names)
    defaults = global_head_defaults(config)
    task_heads = {
        str(name).lower(): dict(settings or {})
        for name, settings in (config.get("training", {}).get("task_heads", {}) or {}).items()
    }
    unknown = sorted(set(task_heads) - set(names))
    if unknown:
        raise ValueError(f"training.task_heads contains unknown tasks: {unknown}; valid tasks: {names}")
    return {name: normalize_head_config(task_heads.get(name, {}), defaults) for name in names}
