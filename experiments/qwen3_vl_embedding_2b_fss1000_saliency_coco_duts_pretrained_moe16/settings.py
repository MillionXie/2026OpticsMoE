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
    """Fine-tuning settings layered over the exact pretrained architecture.

    Optical geometry and Qwen runtime fields are copied from the COCO/DUTS
    source configuration.  This prevents a fine-tuning config from silently
    constructing a shape-compatible but physically different backbone.
    """

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
        self.source_checkpoint = _resolve(
            d("source.duts_checkpoint"),
            base,
        )
        self.output_dir = _resolve(
            d(
                "output_dir",
                "../runs/fss1000_saliency_coco_duts_pretrained_moe16",
            ),
            base,
        )
        self.data_root = _resolve(
            d("dataset.data_root", "../../../data/FSS-1000"),
            base,
        )
        self.download = bool(d("dataset.download", True))
        self.download_source = str(d("dataset.download_source", "auto"))
        self.download_file_id = str(
            d("dataset.download_file_id", "16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI")
        )
        self.huggingface_dataset_id = str(
            d("dataset.huggingface_dataset_id", "nobg/FSS-1000")
        )
        self.huggingface_endpoint = str(
            d("dataset.huggingface_endpoint", "https://hf-mirror.com")
        )
        self.official_test_list_url = str(
            d(
                "dataset.official_test_list_url",
                "https://raw.githubusercontent.com/HKUSTCV/FSS-1000/"
                "master/fss_test_set.txt",
            )
        )
        self.merge_official_validation_into_train = True
        self.train_class_limit = d("dataset.train_class_limit")
        self.test_class_limit = d("dataset.test_class_limit")
        self.images_per_class_limit = d("dataset.images_per_class_limit")

        self.student_batch_size = int(d("batching.train_batch_size", 4))
        self.inference_batch_size = int(d("batching.inference_batch_size", 4))
        self.num_workers = int(d("batching.num_workers", 8))
        self.duts_batch_size = self.student_batch_size

        self.duts_head_warmup_epochs = int(d("training.head_warmup_epochs", 3))
        self.duts_finetune_epochs = int(d("training.joint_finetune_epochs", 47))
        self.duts_optical_learning_rate = float(
            d("training.optical_learning_rate", 5.0e-5)
        )
        self.duts_phase_learning_rate = float(
            d("training.phase_learning_rate", 1.0e-3)
        )
        self.duts_recombiner_learning_rate = float(
            d("training.recombiner_learning_rate", 1.0e-4)
        )
        self.duts_head_learning_rate = float(
            d("training.head_learning_rate", 5.0e-4)
        )
        self.duts_weight_decay = float(d("training.weight_decay", 0.0))
        self.evaluate_duts_test_each_epoch = bool(
            d("training.evaluate_test_each_epoch", True)
        )
        self.checkpoint_interval_epochs = int(
            d("training.checkpoint_interval_epochs", 10)
        )
        self.log_interval_batches = int(
            d("training.log_interval_batches", 125)
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
        self.phase_dc_weight = float(d("loss.phase_dc_weight", 0.0))

        self.augmentation_enabled = bool(d("augmentation.enabled", True))
        self.crop_scale_min = float(d("augmentation.crop_scale_min", 0.90))
        self.horizontal_flip_probability = float(
            d("augmentation.horizontal_flip_probability", 0.5)
        )
        self.brightness_jitter = float(
            d("augmentation.brightness_jitter", 0.10)
        )
        self.contrast_jitter = float(
            d("augmentation.contrast_jitter", 0.10)
        )
        self.rotation_degrees = float(
            d("augmentation.rotation_degrees", 0.0)
        )

        self.visualization_interval_epochs = int(
            d("visualization.interval_epochs", 5)
        )
        self.visualization_sample_count = int(
            d("visualization.sample_count", 12)
        )

        if self.source_checkpoint is None:
            raise ValueError("source.duts_checkpoint is required")
        if self.output_dir is None or self.data_root is None:
            raise ValueError("output_dir and dataset.data_root are required")
        self.validate()

    @property
    def total_epochs(self) -> int:
        return self.duts_head_warmup_epochs + self.duts_finetune_epochs

    @property
    def duts_total_epochs(self) -> int:
        return self.total_epochs

    def resolve_architecture(self, model: Any) -> None:
        ArchitectureSettings.resolve_architecture(self, model)

    def validate(self) -> None:
        if self.image_size != 224 or self.detector_output_size != 224:
            raise ValueError("FSS transfer keeps the validated 224x224 interface")
        if self.expert_layers != 3:
            raise ValueError(
                "The COCO/DUTS transfer experiment requires exactly 3 expert stages"
            )
        if self.num_experts != 16 or self.top_k != 4:
            raise ValueError("The transferred backbone must remain MoE16 top-4")
        if self.total_epochs <= 0:
            raise ValueError("At least one fine-tuning epoch is required")
        for name in (
            "duts_optical_learning_rate",
            "duts_phase_learning_rate",
            "duts_recombiner_learning_rate",
            "duts_head_learning_rate",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.phase_dc_weight < 0:
            raise ValueError("loss.phase_dc_weight cannot be negative")
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError("augmentation.crop_scale_min must be in (0,1]")

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
                "source_checkpoint": str(self.source_checkpoint),
                "expert_stages": self.expert_layers,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "total_epochs": self.total_epochs,
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
