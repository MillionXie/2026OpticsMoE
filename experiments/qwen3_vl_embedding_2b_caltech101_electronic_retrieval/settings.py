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

    settings.electronic_width = int(d("electronic.width", 128))
    settings.electronic_layers = int(d("electronic.layers", 2))
    settings.electronic_expansion = float(d("electronic.expansion", 2.0))
    settings.electronic_dropout = _dropout(
        "electronic.dropout", d("electronic.dropout", 0.10)
    )
    settings.electronic_initial_residual_weight = float(
        d("electronic.initial_residual_weight", 0.10)
    )
    settings.electronic_token_mixer_enabled = bool(
        d("electronic.token_mixer.enabled", False)
    )
    settings.electronic_token_mixer_kernel_size = int(
        d("electronic.token_mixer.kernel_size", 5)
    )
    settings.electronic_pooling = str(d("electronic.pooling", "mean"))
    settings.reserve_test_before_train = bool(
        d("dataset.reserve_test_before_train", True)
    )
    settings.episodic_prototype_loss_enabled = bool(
        d("training.episodic_prototype_loss", True)
    )
    settings.teacher_enabled = bool(d("training.teacher_enabled", False))
    settings.learning_rate_schedule = str(
        d("training.learning_rate_schedule", "cosine")
    )
    settings.learning_rate_warmup_ratio = float(
        d("training.learning_rate_warmup_ratio", 0.05)
    )

    if settings.use_all_categories or len(settings.selected_skus) != 10:
        raise ValueError("The direct electronic experiment requires exactly 10 categories")
    if settings.electronic_width <= 0 or settings.electronic_layers <= 0:
        raise ValueError("Electronic width/layers must be positive")
    if settings.electronic_expansion <= 0.0:
        raise ValueError("Electronic expansion must be positive")
    if not 0.0 < settings.electronic_initial_residual_weight < 1.0:
        raise ValueError("Electronic initial_residual_weight must be in (0,1)")
    if (
        settings.electronic_token_mixer_kernel_size <= 0
        or settings.electronic_token_mixer_kernel_size % 2 == 0
    ):
        raise ValueError("Electronic token mixer kernel_size must be a positive odd integer")
    if settings.electronic_pooling not in {"mean", "mean_max"}:
        raise ValueError("Electronic pooling must be mean or mean_max")
    if not settings.reserve_test_before_train:
        raise ValueError("Electronic all-data training must reserve test before train")
    if not settings.episodic_prototype_loss_enabled:
        raise ValueError("Electronic training requires episodic prototype loss")
    if settings.learning_rate_schedule not in {"constant", "cosine"}:
        raise ValueError("Unsupported learning_rate_schedule")
    if not 0.0 <= settings.learning_rate_warmup_ratio < 1.0:
        raise ValueError("learning_rate_warmup_ratio must be in [0,1)")
    teacher_weights = (
        settings.lambda_kd,
        settings.lambda_relational_kd,
        settings.lambda_teacher_gallery,
    )
    if not settings.teacher_enabled and any(
        value != 0.0
        for value in teacher_weights
    ):
        raise ValueError("Pure electronic training disables every teacher loss")
    if settings.teacher_enabled and not any(value > 0.0 for value in teacher_weights):
        raise ValueError("Teacher-enabled training requires a positive teacher loss")

    # Keep the shared training/evaluation engine on its single deterministic
    # route. There is no learned router, expert selection, or optical phase.
    settings.num_experts = 1
    settings.top_k = 1
    settings.detector_output_size = settings.electronic_width * (
        2 if settings.electronic_pooling == "mean_max" else 1
    )
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
    if not settings.teacher_enabled:
        values.pop("teacher_cache", None)
    values["student_initialization"] = {
        "student_checkpoint": None,
        "all101_pretraining": False,
        "base_qwen_pretrained": True,
        "base_qwen_frozen": True,
        "teacher_embedding_supervision": settings.teacher_enabled,
    }
    values["electronic"] = {
        "type": (
            "depthwise_token_mixer_residual_mlp"
            if settings.electronic_token_mixer_enabled
            else "shared_tokenwise_residual_mlp"
        ),
        "optical_enabled": False,
        "moe_enabled": False,
        "width": settings.electronic_width,
        "layers": settings.electronic_layers,
        "attention_enabled": False,
        "token_mixing_enabled": settings.electronic_token_mixer_enabled,
        "token_mixer": {
            "type": "depthwise_conv1d_pointwise_linear",
            "kernel_size": settings.electronic_token_mixer_kernel_size,
            "language_padding": "causal_left",
            "vision_padding": "symmetric",
        },
        "expansion": settings.electronic_expansion,
        "dropout": settings.electronic_dropout,
        "initial_residual_weight": settings.electronic_initial_residual_weight,
        "pooling": settings.electronic_pooling,
        "embedding_head": (
            f"LayerNorm({settings.detector_output_size}) -> "
            f"Linear({settings.detector_output_size},64) -> L2Normalize"
        ),
    }
    values["training"]["teacher_used"] = settings.teacher_enabled
    values["training"]["episodic_prototype_loss"] = True
    values["training"]["learning_rate_schedule"] = settings.learning_rate_schedule
    values["training"]["learning_rate_warmup_ratio"] = (
        settings.learning_rate_warmup_ratio
    )
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
