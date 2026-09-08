"""Audited OpenMoji settings with a shared 17 um / 10 cm optical contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    _nested,
    _read_config,
)
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.settings import (
    find_local_qwen_checkpoint,
)


VARIANTS = {
    "optical_router_scale_matched_moe",
    "d2nn_active_expert_matched",
    "qwen_frozen_pending_5090d",
}


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


class Settings:
    def __init__(self, config_path: Path, raw: dict[str, Any]) -> None:
        self.config_path = config_path
        d = lambda key, default=None: _nested(raw, key, default)
        base = config_path.parent
        self.seed = int(d("seed", 73))
        self.data_dir = _resolve(d("dataset.data_dir"), base)
        self.asset_dir = _resolve(d("dataset.asset_dir"), base)
        self.output_dir = _resolve(d("output_dir"), base)
        self.image_size = int(d("dataset.image_size", 224))
        self.grid_size = int(d("dataset.grid_size", 6))
        self.icon_size = int(d("dataset.icon_size", 30))
        self.icon_classes = int(d("dataset.icon_classes", 16))
        self.train_samples = int(d("dataset.train_samples", 5000))
        self.test_samples = int(d("dataset.test_samples", 1000))
        self.prompt_templates_per_operation = int(
            d("dataset.prompt_templates_per_operation", 3)
        )
        self.qwen_model_id = str(d("qwen.model_id", "Qwen/Qwen3-VL-2B-Instruct"))
        checkpoint = d("qwen.checkpoint", "auto")
        if checkpoint in {None, "auto"}:
            try:
                self.qwen_checkpoint = find_local_qwen_checkpoint(self.qwen_model_id)
            except FileNotFoundError:
                self.qwen_checkpoint = Path("__QWEN3_VL_2B_INSTRUCT_NOT_FOUND__").resolve()
        else:
            self.qwen_checkpoint = _resolve(checkpoint, base)
        self.prompt_cache_path = self.data_dir / str(d("qwen.prompt_cache", "prompt_hidden.pt"))
        self.prompt_cache_batch_size = int(d("qwen.prompt_cache_batch_size", 16))
        self.prompt_max_tokens = int(d("qwen.prompt_max_tokens", 64))
        self.optical_base_config = (
            Path(__file__).resolve().parents[3]
            / "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml"
        )
        self.optical_enabled = True
        self.electronic_width = int(d("model.electronic_width", 192))
        self.max_language_tokens = int(d("model.max_language_tokens", 64))
        self.optical_fusion_initial = float(d("model.optical_fusion_initial", 0.055))
        self.embedding_only = bool(d("model.embedding_only", False))
        self.fusion_alpha_minimum = float(d("model.fusion_alpha_minimum", 0.01))
        self.fusion_alpha_maximum = float(d("model.fusion_alpha_maximum", 0.95))
        self.editor_depth = int(d("model.editor_depth", 3))
        self.position_scale = float(d("model.position_scale", 0.1))
        if self.embedding_only:
            if not 0.4 < self.fusion_alpha_minimum < self.optical_fusion_initial < self.fusion_alpha_maximum < 1:
                raise ValueError('Embedding-only contract requires 0.4 < alpha_min < initial < alpha_max < 1')
            if self.editor_depth not in (1, 2, 3):
                raise ValueError('editor_depth must be 1, 2, or 3')
            if self.prompt_cache_path.name == 'prompt_hidden.pt':
                raise ValueError('Embedding-only profile must not reuse contextual prompt_hidden.pt')
        self.optical_shift_pixels = 16
        self.phase_dropout_p = 0.08
        self.epochs = int(d("training.epochs", 40))
        self.batch_size = int(d("training.batch_size", 16))
        self.num_workers = int(d("training.num_workers", 4))
        self.learning_rate = float(d("training.learning_rate", 1.0e-4))
        self.adapter_learning_rate = float(d("training.adapter_learning_rate", 1.0e-4))
        self.phase_learning_rate = float(d("training.phase_learning_rate", 0.006))
        self.router_learning_rate = float(d("training.router_learning_rate", 5.0e-4))
        self.readout_learning_rate = float(d("training.readout_learning_rate", 5.0e-5))
        self.decoder_learning_rate = float(d("training.decoder_learning_rate", 3.0e-4))
        self.weight_decay = float(d("training.weight_decay", 0.01))
        self.warmup_electronic_epochs = int(d("training.warmup_electronic_epochs", 0))
        self.gradient_clip_norm = float(d("training.gradient_clip_norm", 1.0))
        self.ema_decay = float(d("training.ema_decay", 0.995))
        self.amp_enabled = bool(d("training.amp_enabled", True))
        self.resume = bool(d("training.resume", False))
        self.changed_cell_weight = float(d("loss.changed_cell_weight", 8.0))
        self.foreground_cell_weight = float(d("loss.foreground_cell_weight", 2.0))
        self.category_loss_weight = float(d("loss.category", 1.0))
        self.edit_loss_weight = float(d("loss.edit", 1.0))
        self.preservation_loss_weight = float(d("loss.preservation", 0.2))
        self.task_loss_weight = float(d("loss.task", 0.1))
        self.ccd_loss_weight = float(d("loss.ccd", 0.02))
        self.router_balance_weight = float(d("loss.router_balance", 0.08))
        self.router_importance_weight = float(d("loss.router_importance", 0.02))
        self.router_hard_load_balance_weight = float(
            d("loss.router_hard_load_balance", 0.50)
        )
        self.router_semantic_code_weight = float(
            d("loss.router_semantic_code", 0.35)
        )
        self.phase_dc_weight = float(d("loss.phase_dc", 0.005))
        self.test_interval_epochs = int(d("protocol.test_interval_epochs", 5))
        self.legacy_warmstart_checkpoint = _resolve(
            d("protocol.legacy_warmstart_checkpoint"), base
        )
        self.log_interval = int(d("logging.interval_batches", 20))
        self.visualization_samples_per_task = int(
            d("logging.visualization_samples_per_task", 8)
        )
        self.lightgen_model_variant = str(
            d("lightgen.model_variant", "optical_router_scale_matched_moe")
        )
        if self.lightgen_model_variant not in VARIANTS:
            raise ValueError(f"Unknown T04 variant: {self.lightgen_model_variant}")
        if self.train_samples != 5000 or self.test_samples != 1000:
            raise ValueError("T04 formal protocol is fixed to 5000 train / 1000 test")
        if self.lightgen_model_variant == "optical_router_scale_matched_moe":
            if self.router_hard_load_balance_weight <= 0:
                raise ValueError("T04 optical Router requires hard-load regularization")
            if self.router_semantic_code_weight <= 0:
                raise ValueError("T04 optical Router requires semantic-code supervision")

    @property
    def train_manifest(self) -> Path:
        return self.data_dir / "train.jsonl"

    @property
    def test_manifest(self) -> Path:
        return self.data_dir / "test.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
            if not key.startswith("_")
        }


def load_settings(path: str | Path) -> Settings:
    config = Path(path).expanduser().resolve()
    return Settings(config, _read_config(config))


__all__ = ["Settings", "load_settings"]
