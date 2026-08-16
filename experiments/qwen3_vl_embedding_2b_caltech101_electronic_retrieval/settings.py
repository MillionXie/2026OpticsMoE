from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    Caltech101Settings,
    load_settings as load_caltech_settings,
    save_resolved_config as save_caltech_resolved_config,
)


def _dropout(name: str, value: Any) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be in [0,1)")
    return result


def load_settings(path: str | Path) -> Caltech101Settings:
    config_path = Path(path).expanduser().resolve()
    settings = load_caltech_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.electronic_width = int(d("electronic.width", 256))
    settings.electronic_layers = int(d("electronic.layers", 2))
    settings.electronic_heads = int(d("electronic.attention_heads", 8))
    settings.electronic_ff_multiplier = float(d("electronic.ff_multiplier", 2.5))
    settings.electronic_dropout = _dropout(
        "electronic.dropout", d("electronic.dropout", 0.10)
    )
    settings.electronic_attention_dropout = _dropout(
        "electronic.attention_dropout",
        d("electronic.attention_dropout", 0.05),
    )
    settings.electronic_initial_residual_weight = float(
        d("electronic.initial_residual_weight", 0.10)
    )
    settings.electronic_readout_hidden = int(
        d("electronic.embedding_head.hidden_dim", 384)
    )
    settings.electronic_readout_dropout = _dropout(
        "electronic.embedding_head.dropout",
        d("electronic.embedding_head.dropout", 0.15),
    )

    if settings.use_all_categories or len(settings.selected_skus) != 10:
        raise ValueError("The direct electronic experiment requires exactly 10 categories")
    if settings.electronic_width <= 0 or settings.electronic_layers <= 0:
        raise ValueError("Electronic width/layers must be positive")
    if settings.electronic_heads <= 0 or settings.electronic_width % settings.electronic_heads:
        raise ValueError("Electronic width must be divisible by attention_heads")
    if settings.electronic_ff_multiplier <= 0.0:
        raise ValueError("Electronic ff_multiplier must be positive")
    if not 0.0 < settings.electronic_initial_residual_weight < 1.0:
        raise ValueError("Electronic initial_residual_weight must be in (0,1)")
    if settings.electronic_readout_hidden <= 0:
        raise ValueError("Electronic embedding-head hidden_dim must be positive")

    # Keep the shared training/evaluation engine on its single deterministic
    # route. There is no learned router, expert selection, or optical phase.
    settings.num_experts = 1
    settings.top_k = 1
    settings.detector_output_size = settings.electronic_width
    settings.lambda_router_balance = 0.0
    settings.lambda_router_importance = 0.0
    settings.router_learning_rate = 0.0
    settings.phase_learning_rate = 0.0
    settings.phase_focus_enabled = False
    return settings


def save_resolved_config(settings: Caltech101Settings) -> None:
    save_caltech_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Optical/robust-hybrid fields exist only because the shared loader owns
    # common Qwen/data settings. They are not part of this resolved model.
    values.pop("optical", None)
    values.pop("robust_hybrid", None)
    values["student_initialization"] = {
        "student_checkpoint": None,
        "all101_pretraining": False,
        "teacher_qwen_frozen": True,
    }
    values["electronic"] = {
        "type": "dense_transformer_replacement",
        "optical_enabled": False,
        "moe_enabled": False,
        "width": settings.electronic_width,
        "layers": settings.electronic_layers,
        "attention_heads": settings.electronic_heads,
        "ff_multiplier": settings.electronic_ff_multiplier,
        "dropout": settings.electronic_dropout,
        "attention_dropout": settings.electronic_attention_dropout,
        "initial_residual_weight": settings.electronic_initial_residual_weight,
        "embedding_head": {
            "hidden_dim": settings.electronic_readout_hidden,
            "dropout": settings.electronic_readout_dropout,
        },
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
