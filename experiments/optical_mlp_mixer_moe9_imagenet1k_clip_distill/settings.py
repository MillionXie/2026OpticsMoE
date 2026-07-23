from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]


def _path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (PROJECT_DIR / result).resolve()


@dataclass
class DatasetSettings:
    name: str = "imagenet1k"
    source: str = "huggingface"
    root: Path = PROJECT_DIR / "data" / "imagenet1k"
    train_split: str = "train"
    validation_split: str = "validation"
    download: bool = True
    hf_dataset_id: str = "ILSVRC/imagenet-1k"
    hf_revision: str = "main"
    hf_cache_dir: Path = PROJECT_DIR / "data" / "imagenet1k" / "huggingface_cache"
    strict_standard_counts: bool = True
    expected_train_samples: int = 1_281_167
    expected_validation_samples: int = 50_000
    train_limit: int | None = None
    validation_limit: int | None = None
    seed: int = 42


@dataclass
class GeometrySettings:
    canvas_size: int = 792
    active_size: int = 762
    num_experts: int = 9
    expert_size: int = 224
    expert_pitch: int = 254
    outer_padding_per_side: int = 15


@dataclass
class RouterSettings:
    top_k: int = 3
    pool_size: int = 14
    temperature: float = 1.0
    input_layernorm_enabled: bool = True
    input_layernorm_eps: float = 1e-5
    amplitude_weight_domain: str = "amplitude"
    amplitude_input_normalization: str = "none"
    balance_weight: float = 0.03
    importance_weight: float = 0.0


@dataclass
class OpticsSettings:
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 16.0
    inter_layer_distance_m: float = 0.10
    readout_to_global_distance_m: float = 0.10
    global_to_detector_distance_m: float = 0.10
    phase_parameterization: str = "sigmoid"
    phase_init: str = "zeros"
    phase_init_std: float = 0.02
    amplitude_mask_enabled: bool = False
    k_space_constraint_enabled: bool = False
    theta_max_deg: float = 1.0


@dataclass
class OEOSettings:
    enabled: bool = True
    per_expert_enabled: bool = True
    elementwise_affine: bool = False
    hard_route_mask: bool = True
    reapply_routing_weights: bool = True
    layernorm_eps: float = 1e-5
    nonlinearity: str = "relu"


@dataclass
class PhaseDropoutSettings:
    enabled: bool = False
    mode: str = "none"
    p: float = 0.0
    block_size: int = 8
    batch_shared: bool = True
    start_epoch: int = 0


@dataclass
class ModelSettings:
    name: str = "OpticalMixerMoE9"
    image_size: int = 224
    patch_size: int = 16
    token_count: int = 196
    hidden_size: int = 224
    num_blocks: int = 7
    token_stages_per_block: int = 2
    channel_stages_per_block: int = 3
    phase_layers_per_block: int = 5
    clip_projection_dim: int = 512
    num_classes: int = 1000
    residual_scale: float = 1.0
    residual_scale_trainable: bool = False
    final_layernorm_eps: float = 1e-5
    detector_layernorm_eps: float = 1e-5


@dataclass
class ClipSettings:
    model_name: str = "ViT-B/16"
    cache_dir: Path | None = None
    local_clip_repository: Path = PROJECT_DIR / "CLIP"
    views_per_train_image: int = 4
    cache_dtype: str = "float16"
    cache_batch_size: int = 256
    train_transform_version: str = "rrc_flip_randaugment_v1"
    random_resized_crop_scale: tuple[float, float] = (0.08, 1.0)
    random_resized_crop_ratio: tuple[float, float] = (0.75, 1.3333333333333333)
    horizontal_flip_probability: float = 0.5
    randaugment_enabled: bool = True
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    text_prompt_templates: list[str] = field(default_factory=lambda: [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a bright photo of a {}.",
        "a dark photo of a {}.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}.",
        "a photo of the {}.",
    ])
    logit_temperature: float = 1.0


@dataclass
class LossSettings:
    feature_cosine_weight: float = 1.0
    clip_logit_kd_weight: float = 1.0
    supervised_ce_weight: float = 0.5
    distill_temperature: float = 2.0


@dataclass
class OptimizerSettings:
    name: str = "adamw"
    learning_rate: float = 5e-4
    router_learning_rate: float = 5e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    minimum_learning_rate: float = 1e-6


@dataclass
class TrainingSettings:
    output_dir: Path = PROJECT_DIR / "runs" / "optical_mlp_mixer_moe9_imagenet1k_clip_distill"
    device: str = "cuda"
    epochs: int = 90
    batch_size: int = 16
    validation_batch_size: int = 16
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    log_interval_batches: int = 100
    validation_interval_epochs: int = 1
    checkpoint_interval_epochs: int = 5
    resume_checkpoint: Path | None = None
    seed: int = 42
    progress: bool = True
    amp_enabled: bool = False


@dataclass
class VisualizationSettings:
    enabled: bool = True
    interval_epochs: int = 10
    sample_count: int = 4
    capture_block_indices: list[int] = field(default_factory=lambda: [0, 3, 6])
    save_raw_tensors: bool = True
    percentile_clip: float = 99.0
    save_phase_overview: bool = True
    save_router_charts: bool = True
    save_training_curves: bool = True


@dataclass
class ExperimentSettings:
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    router: RouterSettings = field(default_factory=RouterSettings)
    optics: OpticsSettings = field(default_factory=OpticsSettings)
    oeo: OEOSettings = field(default_factory=OEOSettings)
    phase_dropout: PhaseDropoutSettings = field(default_factory=PhaseDropoutSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    clip: ClipSettings = field(default_factory=ClipSettings)
    loss: LossSettings = field(default_factory=LossSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    config_path: Path | None = None

    def validate(self) -> None:
        if self.dataset.name != "imagenet1k":
            raise ValueError("This experiment only supports dataset='imagenet1k'")
        if self.dataset.source not in {"huggingface", "imagefolder"}:
            raise ValueError("dataset.source must be 'huggingface' or 'imagefolder'")
        if self.dataset.source == "huggingface":
            if not self.dataset.hf_dataset_id.strip():
                raise ValueError("dataset.hf_dataset_id cannot be empty")
            if not self.dataset.hf_revision.strip():
                raise ValueError("dataset.hf_revision cannot be empty")
        if self.model.name != "OpticalMixerMoE9":
            raise ValueError("Only OpticalMixerMoE9 is implemented in this experiment")
        if self.model.image_size % self.model.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        patches_per_axis = self.model.image_size // self.model.patch_size
        expected_tokens = patches_per_axis * patches_per_axis
        if self.model.token_count != expected_tokens:
            raise ValueError(
                f"token_count must equal (image_size/patch_size)^2={expected_tokens}, "
                f"got {self.model.token_count}"
            )
        if self.model.hidden_size != self.geometry.expert_size:
            raise ValueError("hidden_size must equal expert_size for interpolation-free [T,C] optical mapping")
        if self.model.token_count > self.geometry.expert_size:
            raise ValueError("token_count cannot exceed expert_size because truncation is forbidden")
        if self.geometry.num_experts != 9:
            raise ValueError("OpticalMixerMoE9 requires exactly nine experts")
        if self.geometry.expert_pitch - self.geometry.expert_size != 30:
            raise ValueError("Expert gap must remain exactly 30 pixels")
        if self.geometry.active_size != 3 * self.geometry.expert_pitch:
            raise ValueError("active_size must equal 3*expert_pitch")
        if self.geometry.canvas_size - self.geometry.active_size != 2 * self.geometry.outer_padding_per_side:
            raise ValueError("canvas/active sizes do not preserve configured outer padding")
        if self.model.token_stages_per_block + self.model.channel_stages_per_block != self.model.phase_layers_per_block:
            raise ValueError("token and channel stage counts must sum to phase_layers_per_block")
        if self.model.num_blocks != 7:
            raise ValueError("The formal experiment is fixed to seven OpticalMixerMoE9 blocks")
        if not 1 <= self.router.top_k <= self.geometry.num_experts:
            raise ValueError("router.top_k must be in [1,num_experts]")
        if self.router.amplitude_weight_domain not in {"amplitude", "power"}:
            raise ValueError("router.amplitude_weight_domain must be amplitude or power")
        if self.router.amplitude_input_normalization not in {"none", "per_sample_max"}:
            raise ValueError("Unsupported amplitude input normalization")
        if self.optics.amplitude_mask_enabled:
            raise ValueError("This phase-only experiment does not implement trainable amplitude masks")
        if self.oeo.nonlinearity not in {"relu", "softplus"}:
            raise ValueError("OEO nonlinearity must be relu or softplus")
        if self.phase_dropout.enabled:
            raise ValueError("The formal experiment keeps phase dropout disabled")
        if self.optimizer.name.lower() != "adamw":
            raise ValueError("The requested formal experiment uses AdamW")
        if self.optimizer.weight_decay != 0.0:
            raise ValueError("weight_decay must remain 0.0 for this optical phase experiment")
        if self.training.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.clip.views_per_train_image <= 0:
            raise ValueError("views_per_train_image must be positive")
        if self.clip.cache_dtype not in {"float16", "float32"}:
            raise ValueError("clip.cache_dtype must be float16 or float32")
        if self.loss.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")

    @property
    def optical_parameter_formula(self) -> dict[str, int]:
        expert_plane = self.geometry.num_experts * self.geometry.expert_size ** 2
        expert_per_block = self.model.phase_layers_per_block * expert_plane
        global_per_block = self.geometry.active_size ** 2
        per_block = expert_per_block + global_per_block
        return {
            "phase_parameters_per_expert_plane": expert_plane,
            "expert_phase_parameters_per_block": expert_per_block,
            "global_phase_parameters_per_block": global_per_block,
            "optical_phase_parameters_per_block": per_block,
            "expert_phase_parameters_total": expert_per_block * self.model.num_blocks,
            "global_phase_parameters_total": global_per_block * self.model.num_blocks,
            "optical_phase_parameters_total": per_block * self.model.num_blocks,
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _construct(cls: type, values: dict[str, Any]) -> Any:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} config keys: {unknown}")
    return cls(**values)


def load_settings(path: str | Path) -> ExperimentSettings:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed_sections = {
        "dataset", "geometry", "router", "optics", "oeo", "phase_dropout",
        "model", "clip", "loss", "optimizer", "training", "visualization",
    }
    unknown = sorted(set(raw) - allowed_sections)
    if unknown:
        raise ValueError(f"Unknown top-level config sections: {unknown}")

    dataset = dict(raw.get("dataset", {}))
    training = dict(raw.get("training", {}))
    clip = dict(raw.get("clip", {}))
    if "root" in dataset:
        dataset["root"] = _path(dataset["root"])
    if "hf_cache_dir" in dataset:
        dataset["hf_cache_dir"] = _path(dataset["hf_cache_dir"])
    if "output_dir" in training:
        training["output_dir"] = _path(training["output_dir"])
    if training.get("resume_checkpoint") is not None:
        training["resume_checkpoint"] = _path(training["resume_checkpoint"])
    if clip.get("cache_dir") is not None:
        clip["cache_dir"] = _path(clip["cache_dir"])
    if "local_clip_repository" in clip:
        clip["local_clip_repository"] = _path(clip["local_clip_repository"])
    for key in ("random_resized_crop_scale", "random_resized_crop_ratio"):
        if key in clip:
            clip[key] = tuple(float(value) for value in clip[key])

    settings = ExperimentSettings(
        dataset=_construct(DatasetSettings, dataset),
        geometry=_construct(GeometrySettings, raw.get("geometry", {})),
        router=_construct(RouterSettings, raw.get("router", {})),
        optics=_construct(OpticsSettings, raw.get("optics", {})),
        oeo=_construct(OEOSettings, raw.get("oeo", {})),
        phase_dropout=_construct(PhaseDropoutSettings, raw.get("phase_dropout", {})),
        model=_construct(ModelSettings, raw.get("model", {})),
        clip=_construct(ClipSettings, clip),
        loss=_construct(LossSettings, raw.get("loss", {})),
        optimizer=_construct(OptimizerSettings, raw.get("optimizer", {})),
        training=_construct(TrainingSettings, training),
        visualization=_construct(VisualizationSettings, raw.get("visualization", {})),
        config_path=config_path,
    )
    settings.validate()
    return settings
