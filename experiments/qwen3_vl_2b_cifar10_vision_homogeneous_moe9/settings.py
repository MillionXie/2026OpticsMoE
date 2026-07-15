from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from . import MODEL_ID


PROJECT_DIR = Path(__file__).resolve().parent
PATH_FIELDS = {"data_root", "output_dir", "cache_dir"}
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class Settings:
    dataset: str = "cifar10"
    data_root: Path = PROJECT_DIR.parent.parent / "opticalmoe_experiments" / "data"
    download: bool = True
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_cifar10_vision_homogeneous_moe9"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int = 25600
    processor_max_pixels: int = 25600
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    feature_batch_size: int = 1
    student_batch_size: int = 1
    inference_batch_size: int = 1
    head_batch_size: int = 512
    teacher_cache_shard_size: int = 128
    teacher_cache_lru_shards: int = 8
    num_workers: int = 8
    cache_dtype: str = "float16"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    epochs: int = 30
    validation_fraction: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    head_type: str = "normalized_linear"
    dropout: float = 0.0
    input_adapter_dim: int = 120
    max_visual_tokens: int = 120
    canvas_size: int = 480
    active_size: int = 450
    expert_size: int = 120
    expert_pitch: int = 150
    num_experts: int = 9
    top_k: int = 3
    router_pool_size: int = 10
    router_temperature: float = 1.0
    expert_layers: int = 5
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 16.0
    prompt_focal_length_m: float = 0.0722
    prompt_to_expert_distance_m: float = 0.1444
    expert_interlayer_distance_m: float = 0.20
    last_expert_to_global_distance_m: float = 0.20
    global_to_detector_distance_m: float = 0.20
    phase_parameterization: str = "sigmoid"
    phase_init: str = "zeros"
    phase_init_std: float = 0.02
    k_space_constraint_enabled: bool = False
    theta_max_deg: float = 1.0
    interlayer_layernorm_eps: float = 1e-5
    interlayer_nonlinearity: str = "relu"
    detector_pool_kernel: int = 4
    detector_layernorm_eps: float = 1e-5
    detector_layernorm_affine: bool = False
    detector_nonlinearity: str = "relu"
    loss_hidden_weight: float = 1.0
    loss_kd_weight: float = 0.5
    loss_ce_weight: float = 0.5
    router_balance_weight: float = 0.1
    router_importance_weight: float = 0.0
    distill_temperature: float = 2.0
    log_interval_batches: int = 100
    seed: int = 42
    progress: bool = True
    vision_depth: int | None = None
    vision_hidden_size: int | None = None

    def validate(self) -> None:
        if self.dataset != "cifar10":
            raise ValueError("This experiment requires dataset='cifar10'")
        if self.model_id != MODEL_ID and not Path(self.model_id).is_dir():
            raise ValueError(f"model_id must be {MODEL_ID} or an existing local directory")
        if self.head_type != "normalized_linear":
            raise ValueError("This experiment intentionally uses the small normalized_linear teacher/student head")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels must be <= processor_max_pixels")
        if (self.canvas_size, self.active_size, self.expert_size, self.expert_pitch, self.num_experts) != (480, 450, 120, 150, 9):
            raise ValueError("The verified homogeneous MoE geometry is fixed at canvas480/active450/expert120/pitch150/9 experts")
        if self.input_adapter_dim != 120 or self.max_visual_tokens != 120:
            raise ValueError("Direct token-row mapping requires input_adapter_dim=max_visual_tokens=120")
        if self.expert_layers != 5 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("The experiment requires five expert layers and a valid top_k")
        if abs(self.prompt_to_expert_distance_m - 2.0 * self.prompt_focal_length_m) > 1e-6:
            raise ValueError("The verified global fan-out prompt requires prompt_to_expert_distance_m = 2 * prompt_focal_length_m")
        if self.detector_pool_kernel != 4 or self.canvas_size // self.detector_pool_kernel != 120:
            raise ValueError("The full detector must pool exactly from 480x480 to 120x120")
        if self.detector_layernorm_affine:
            raise ValueError("Post-detector LayerNorm must use elementwise_affine=False")
        if self.interlayer_nonlinearity not in {"relu", "softplus"} or self.detector_nonlinearity not in {"relu", "softplus"}:
            raise ValueError("nonlinearities must be relu or softplus")
        if self.cache_dtype not in {"float16", "float32"} or self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported dtype/cache_dtype")
        if not 0.0 < self.validation_fraction < 1.0 or self.distill_temperature <= 0:
            raise ValueError("Invalid validation fraction or distillation temperature")
        for name in ("wavelength_nm", "pixel_pitch_um", "prompt_focal_length_m", "prompt_to_expert_distance_m",
                     "expert_interlayer_distance_m", "last_expert_to_global_distance_m", "global_to_detector_distance_m"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        positive = ("feature_batch_size", "student_batch_size", "inference_batch_size", "head_batch_size", "teacher_cache_shard_size", "teacher_cache_lru_shards", "num_workers", "epochs", "log_interval_batches")
        for name in positive:
            if int(getattr(self, name)) < (0 if name == "num_workers" else 1):
                raise ValueError(f"{name} has an invalid value")
        for name in ("train_limit", "test_limit", "train_limit_per_class", "test_limit_per_class"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        for name in ("loss_hidden_weight", "loss_kd_weight", "loss_ce_weight", "router_balance_weight", "router_importance_weight"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(Settings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = dict(raw)
    if values.get("model_id") and values["model_id"] != MODEL_ID:
        values["model_id"] = str(resolve_path(values["model_id"], config_path.parent, "model_id"))
    for name in PATH_FIELDS:
        if values.get(name) is not None:
            values[name] = resolve_path(values[name], config_path.parent, name)
    settings = Settings(**values)
    settings.validate()
    return settings


def resolve_path(value: str | Path, base: Path, field_name: str) -> Path:
    raw = os.path.expanduser(str(value))
    missing = sorted({a or b for a, b in ENV_REFERENCE.findall(raw) if not os.environ.get(a or b)})
    if missing:
        raise ValueError(f"{field_name} references unset environment variables: {', '.join(missing)}")
    expanded = os.path.expandvars(raw)
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()
