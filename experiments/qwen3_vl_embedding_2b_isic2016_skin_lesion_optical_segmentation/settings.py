from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.settings import (
    Settings as ArchitectureSettings,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.settings import (
    load_settings as load_architecture_settings,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
INITIALIZATION_MODES = {
    "scratch_end_to_end",
    "coco_duts_pretrained",
    "isic_checkpoint_finetune",
}


def _nested(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _resolve(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


class Settings:
    """ISIC training settings over the validated three-stage optical backbone."""

    def __init__(
        self,
        architecture: ArchitectureSettings,
        raw: dict[str, Any],
        config_path: Path,
    ) -> None:
        self.__dict__.update(copy.deepcopy(vars(architecture)))
        self.raw = copy.deepcopy(raw)
        self.config_path = config_path
        base = config_path.parent
        d = lambda key, default=None: _nested(raw, key, default)

        self.source_architecture_config = architecture.config_path
        self.initialization_mode = str(
            d("initialization.mode", "scratch_end_to_end")
        )
        self.source_checkpoint = _resolve(
            d("initialization.pretrained_checkpoint"),
            base,
        )
        self.load_pretrained_segmentation_head = bool(
            d("initialization.load_segmentation_head", True)
        )

        self.output_dir = _resolve(
            d("output_dir", "../runs/isic2016_skin_lesion_scratch"),
            base,
        )
        self.data_root = _resolve(
            d("dataset.data_root", "../../../data/ISIC2016"),
            base,
        )
        self.auto_download = bool(d("dataset.auto_download", True))
        self.remove_archives_after_extract = bool(
            d("dataset.remove_archives_after_extract", True)
        )
        self.expected_train_samples = int(
            d("dataset.expected_train_samples", 900)
        )
        self.expected_test_samples = int(
            d("dataset.expected_test_samples", 379)
        )
        self.train_limit = _optional_int(d("dataset.train_limit"))
        self.test_limit = _optional_int(d("dataset.test_limit"))
        self.resize_mode = str(d("dataset.resize_mode", "stretch"))
        self.train_image_url = str(d("dataset.urls.train_images"))
        self.train_mask_url = str(d("dataset.urls.train_masks"))
        self.test_image_url = str(d("dataset.urls.test_images"))
        self.test_mask_url = str(d("dataset.urls.test_masks"))

        self.student_batch_size = int(d("batching.train_batch_size", 8))
        self.inference_batch_size = int(d("batching.inference_batch_size", 8))
        self.num_workers = int(d("batching.num_workers", 8))
        self.duts_batch_size = self.student_batch_size

        self.head_warmup_epochs = int(d("training.head_warmup_epochs", 0))
        self.joint_finetune_epochs = int(
            d("training.joint_finetune_epochs", 100)
        )
        self.optical_learning_rate = float(
            d("training.optical_learning_rate", 1.0e-4)
        )
        self.router_learning_rate = float(
            d("training.router_learning_rate", 2.0e-4)
        )
        self.recombiner_learning_rate = float(
            d("training.recombiner_learning_rate", 2.0e-4)
        )
        self.head_learning_rate = float(
            d("training.head_learning_rate", 1.0e-3)
        )
        self.weight_decay = float(d("training.weight_decay", 0.0))
        self.evaluate_test_each_epoch = bool(
            d("training.evaluate_test_each_epoch", True)
        )
        self.checkpoint_interval_epochs = int(
            d("training.checkpoint_interval_epochs", 10)
        )
        self.log_interval_batches = int(
            d("training.log_interval_batches", 25)
        )
        self.amp_enabled = bool(d("training.amp_enabled", True))
        self.random_seed = int(d("training.random_seed", 42))

        self.bce_weight = float(d("loss.bce_weight", 1.0))
        self.dice_weight = float(d("loss.dice_weight", 1.0))
        self.soft_iou_weight = float(d("loss.soft_iou_weight", 0.75))
        self.boundary_weight = float(d("loss.boundary_weight", 0.25))
        self.router_balance_weight = float(
            d("loss.router_balance_weight", 0.03)
        )
        self.router_importance_weight = float(
            d("loss.router_importance_weight", 0.0)
        )

        self.augmentation_enabled = bool(d("augmentation.enabled", True))
        self.crop_scale_min = float(d("augmentation.crop_scale_min", 0.92))
        self.horizontal_flip_probability = float(
            d("augmentation.horizontal_flip_probability", 0.5)
        )
        self.vertical_flip_probability = float(
            d("augmentation.vertical_flip_probability", 0.5)
        )
        self.brightness_jitter = float(
            d("augmentation.brightness_jitter", 0.08)
        )
        self.contrast_jitter = float(
            d("augmentation.contrast_jitter", 0.08)
        )
        self.rotation_degrees = float(
            d("augmentation.rotation_degrees", 10.0)
        )

        self.visualization_interval_epochs = int(
            d("visualization.interval_epochs", 5)
        )
        self.visualization_sample_count = int(
            d("visualization.sample_count", 12)
        )

        if self.output_dir is None or self.data_root is None:
            raise ValueError("output_dir and dataset.data_root are required")
        self.validate()

    @property
    def total_epochs(self) -> int:
        return self.head_warmup_epochs + self.joint_finetune_epochs

    @property
    def duts_total_epochs(self) -> int:
        return self.total_epochs

    def resolve_architecture(self, model: Any) -> None:
        ArchitectureSettings.resolve_architecture(self, model)

    def validate(self) -> None:
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(
                f"initialization.mode must be one of {sorted(INITIALIZATION_MODES)}"
            )
        if (
            self.initialization_mode != "scratch_end_to_end"
            and self.source_checkpoint is None
        ):
            raise ValueError(
                f"A checkpoint is required for {self.initialization_mode}"
            )
        if self.image_size != 224 or self.detector_output_size != 224:
            raise ValueError("ISIC keeps the validated 224x224 optical interface")
        if self.expert_layers != 3:
            raise ValueError("ISIC comparison requires exactly three optical stages")
        if self.num_experts != 16 or self.top_k != 4:
            raise ValueError("ISIC comparison requires MoE16 with Top-4 routing")
        if self.total_epochs <= 0:
            raise ValueError("At least one training epoch is required")
        if self.head_warmup_epochs < 0 or self.joint_finetune_epochs < 0:
            raise ValueError("Training epoch counts cannot be negative")
        if self.initialization_mode == "scratch_end_to_end" and self.head_warmup_epochs:
            raise ValueError(
                "scratch_end_to_end must jointly train all task modules from epoch 1"
            )
        if self.resize_mode not in {"stretch"}:
            raise ValueError("dataset.resize_mode currently supports only stretch")
        if self.train_limit is not None and self.train_limit <= 0:
            raise ValueError("dataset.train_limit must be positive or null")
        if self.test_limit is not None and self.test_limit <= 0:
            raise ValueError("dataset.test_limit must be positive or null")
        if self.num_workers < 0:
            raise ValueError("batching.num_workers cannot be negative")
        if self.student_batch_size <= 0 or self.inference_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        for name in (
            "optical_learning_rate",
            "router_learning_rate",
            "recombiner_learning_rate",
            "head_learning_rate",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "weight_decay",
            "bce_weight",
            "dice_weight",
            "soft_iou_weight",
            "boundary_weight",
            "router_balance_weight",
            "router_importance_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.bce_weight + self.dice_weight <= 0:
            raise ValueError("At least BCE or Dice loss must be enabled")
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError("augmentation.crop_scale_min must be in (0,1]")
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in (
            "train_image_url",
            "train_mask_url",
            "test_image_url",
            "test_mask_url",
        ):
            if not str(getattr(self, name)).startswith(("http://", "https://")):
                raise ValueError(f"{name} must be an HTTP(S) URL")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return {
            "config": convert(self.raw),
            "resolved": {
                "experiment_dir": str(EXPERIMENT_DIR),
                "project_dir": str(PROJECT_DIR),
                "output_dir": str(self.output_dir),
                "data_root": str(self.data_root),
                "source_architecture_config": str(
                    self.source_architecture_config
                ),
                "initialization_mode": self.initialization_mode,
                "source_checkpoint": (
                    None
                    if self.source_checkpoint is None
                    else str(self.source_checkpoint)
                ),
                "expert_stages": self.expert_layers,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "total_epochs": self.total_epochs,
                "test_used_for_checkpoint_selection": False,
            },
        }


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    architecture_path = _resolve(
        _nested(raw, "source.architecture_config"),
        config_path.parent,
    )
    if architecture_path is None:
        raise ValueError("source.architecture_config is required")
    architecture = load_architecture_settings(architecture_path)
    return Settings(architecture, raw, config_path)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
