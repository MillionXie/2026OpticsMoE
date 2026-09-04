from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


TARGET_PROMPTS = {
    "spatial": (
        "Please evaluate the spatial quality of this video and rate it using "
        "one of the following five levels: Excellent, Good, Fair, Poor, or Bad."
    ),
    "temporal": (
        "Please evaluate the temporal quality of this video and rate it using "
        "one of the following five levels: Excellent, Good, Fair, Poor, or Bad."
    ),
}
FEATURE_CONTRACT = "qwen3vl_front_patch_position_dynamic_frames_784_mean2x2_196_pool7_49x1024_v2"
FEATURE_CONTRACT_14 = "qwen3vl_front_patch_position_dynamic_frames_784_mean2x2_196_14x14x1024_v1"
LANGUAGE_CONTRACT = "qwen3vl_front_chat_template_embed_tokens_2048_v1"
QUALITY_CONTRACT = "fixed_quality14_dynamic_frames_adaptive_pool7x7_v2"
QUALITY_CONTRACT_14 = "fixed_quality14_dynamic_frames_adaptive_pool14x14_v1"


def feature_contract_for_grid(token_grid: int) -> str:
    if token_grid == 7:
        return FEATURE_CONTRACT
    if token_grid == 14:
        return FEATURE_CONTRACT_14
    raise ValueError("Qwen front token_grid must be 7 or 14")


def quality_contract_for_grid(token_grid: int) -> str:
    if token_grid == 7:
        return QUALITY_CONTRACT
    if token_grid == 14:
        return QUALITY_CONTRACT_14
    raise ValueError("Quality token_grid must be 7 or 14")


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_layered(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config reference: {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    base = raw.pop("base_config", None)
    if base is None:
        return raw
    base_path = Path(str(base)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge(_read_layered(base_path, seen), raw)


def _path(value: Any, config_path: Path) -> Path | None:
    if value in (None, ""):
        return None
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


@dataclass(frozen=True)
class Geometry:
    """518 simulation canvas with a centered 478-pixel hardware active field.

    Temporal supports either a 4x4 layout of 114-pixel lanes with 54-pixel
    experts or a compact 3x3 layout of 156-pixel lanes with 77-pixel experts.
    Spatial retains the earlier 2x2 layout of 232-pixel lanes with 109-pixel
    experts.  The later serial MoE4 always retains measured 109/123 geometry.
    """

    canvas_size: int = 518
    active_size: int = 478
    lane_grid: int = 4
    lane_size: int = 114
    lane_pitch: int = 120
    lane_offset: int = 2
    parallel_expert_size: int = 54
    parallel_expert_pitch: int = 60
    serial_expert_size: int = 109
    serial_expert_pitch: int = 123

    @property
    def active_margin(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def lane_origins(self) -> tuple[tuple[int, int], ...]:
        axis = tuple(
            self.lane_offset + index * self.lane_pitch
            for index in range(self.lane_grid)
        )
        return tuple((top, left) for top in axis for left in axis)

    @property
    def parallel_expert_origins(self) -> tuple[tuple[int, int], ...]:
        axis = (0, self.parallel_expert_pitch)
        return tuple((top, left) for top in axis for left in axis)

    @property
    def vision_expert_size(self) -> int:
        return self.parallel_expert_size

    @property
    def vision_expert_pitch(self) -> int:
        return self.parallel_expert_pitch

    @property
    def vision_expert_origins(self) -> tuple[tuple[int, int], ...]:
        return self.parallel_expert_origins

    @property
    def serial_expert_origins(self) -> tuple[tuple[int, int], ...]:
        axis = (self.serial_expert_pitch, 2 * self.serial_expert_pitch)
        return tuple((top, left) for top in axis for left in axis)

    def validate(self, *, formal: bool = True) -> None:
        dimensions = {
            name: value for name, value in asdict(self).items() if name != "lane_offset"
        }
        if min(dimensions.values()) <= 0 or self.lane_offset < 0:
            raise ValueError("All geometry dimensions must be positive")
        if self.canvas_size < self.active_size or (self.canvas_size - self.active_size) % 2:
            raise ValueError("The active plane must be centered on the canvas")
        axis = sorted({top for top, _ in self.lane_origins})
        if len(axis) != self.lane_grid or axis[0] != self.lane_offset:
            raise ValueError("Parallel lane offset contract is inconsistent")
        if axis[-1] + self.lane_size != self.active_size - self.lane_offset:
            raise ValueError("Parallel lanes are not symmetric inside the active field")
        if self.parallel_expert_pitch + self.parallel_expert_size != self.lane_size:
            raise ValueError("Two experts plus their gap must fill one frame lane")
        if 2 * self.serial_expert_pitch + self.serial_expert_size > self.active_size:
            raise ValueError("Serial MoE4 experts exceed the active plane")
        common = {
            "canvas_size": 518,
            "active_size": 478,
            "serial_expert_size": 109,
            "serial_expert_pitch": 123,
        }
        temporal16 = {
            **common,
            "lane_grid": 4,
            "lane_size": 114,
            "lane_pitch": 120,
            "lane_offset": 2,
            "parallel_expert_size": 54,
            "parallel_expert_pitch": 60,
        }
        temporal9 = {
            **common,
            "lane_grid": 3,
            "lane_size": 156,
            "lane_pitch": 160,
            "lane_offset": 1,
            "parallel_expert_size": 77,
            "parallel_expert_pitch": 79,
        }
        spatial4 = {
            **common,
            "lane_grid": 2,
            "lane_size": 232,
            "lane_pitch": 246,
            "lane_offset": 0,
            "parallel_expert_size": 109,
            "parallel_expert_pitch": 123,
        }
        if formal and asdict(self) not in (temporal16, temporal9, spatial4):
            raise ValueError(
                "Formal geometry must be either Spatial-4 "
                f"{spatial4}, Temporal-9 {temporal9}, or Temporal-16 {temporal16}"
            )


# Backward-friendly alias for code that uses the older geometry class name.
OpticalGeometry = Geometry


@dataclass
class ExperimentSettings:
    config_path: Path
    output_dir: Path
    dataset_root: Path | None
    manifest_path: Path | None
    vision_cache_path: Path | None
    language_cache_path: Path | None
    quality_feature_cache_path: Path | None = None
    raw_frame_cache_path: Path | None = None
    vgg_feature_cache_path: Path | None = None
    training_soft_targets_path: Path | None = None
    initialization_checkpoint: Path | None = None
    frame_stem_checkpoint: Path | None = None
    qwen_model_path: Path | None = None
    reset_serial_router_phase_on_initialization: bool = False
    target_name: str = "spatial"
    prompt: str = TARGET_PROMPTS["spatial"]
    device: str = "cuda"
    random_seed: int = 42
    geometry: Geometry = field(default_factory=Geometry)
    frame_count: int = 16
    token_grid: int = 7
    vision_input_width: int = 1024
    quality_input_width: int = 14
    language_input_width: int = 2048
    maximum_language_tokens: int = 96
    model_width: int = 192
    detector_projection_size: int = 96
    spatial_readout_mode: str = "statistics"
    spatial_residual_max: float = 0.10
    quality_adapter_mode: str = "linear"
    quality_gate_initial: float = 0.25
    qwen_gate_enabled: bool = False
    qwen_gate_initial: float = 0.10
    electronic_skip_enabled: bool = False
    electronic_skip_initial: float = 0.0
    electronic_skip_max: float = 1.0
    quality_refiner_enabled: bool = False
    quality_refiner_max: float = 0.50
    late_input_correction_enabled: bool = False
    late_input_correction_max: float = 0.50
    trainable_frame_stem_enabled: bool = False
    vgg_correction_max: float = 0.50
    vgg_correction_mode: str = "local"
    top_k: int = 2
    router_temperature: float = 1.0
    router_noise_std: float = 0.03
    serial_router_input_size: int = 109
    serial_router_flatfield_calibration: bool = False
    serial_router_channel_standardization: bool = False
    parallel_router_intervals: tuple[tuple[int, int], tuple[int, int]] = ((37, 55), (59, 77))
    serial_router_intervals: tuple[tuple[int, int], tuple[int, int]] = ((164, 223), (255, 314))
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    distance_m: float = 0.10
    k_space_enabled: bool = True
    theta_max_deg: float = 1.0
    input_shift_pixels: int = 8
    phase_shift_pixels: int = 8
    ccd_shift_pixels: int = 8
    phase_dropout_p: float = 0.05
    phase_dropout_cell_size: int = 4
    phase_init_std: float = 0.25
    ccd_relative_clip: float = 8.0
    ccd_log_compression: float = 1.0
    unmodulated_power_fraction_min: float = 0.20
    unmodulated_power_fraction_max: float = 0.35
    unmodulated_power_fraction_eval: float = 0.20
    alpha_min: float = 0.50
    alpha_initial: float = 0.57
    alpha_max: float = 0.90
    fusion_epsilon: float = 1.0e-6
    head_width: int = 256
    dropout: float = 0.15
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    trainable_scope: str = "all"
    learning_rate: float = 2.0e-4
    phase_learning_rate: float = 8.0e-3
    router_phase_learning_rate: float = 1.2e-2
    weight_decay: float = 1.0e-4
    ranking_weight: float = 0.20
    correlation_weight: float = 0.30
    soft_spearman_weight: float = 0.0
    soft_rank_temperature: float = 0.10
    optical_alignment_weight: float = 0.05
    router_balance_weight: float = 0.02
    router_importance_weight: float = 0.002
    serial_router_balance_weight: float = 0.0
    serial_router_importance_weight: float = 0.0
    router_capture_weight: float = 0.02
    soft_target_weight: float = 0.0
    test_interval_epochs: int = 5
    phase_snapshot_interval_epochs: int = 5
    synthetic: bool = False

    @property
    def target(self) -> str:
        return self.target_name

    @property
    def token_count(self) -> int:
        return self.token_grid * self.token_grid

    @property
    def serial_vision_token_count(self) -> int:
        # One learned/electronic summary per sampled frame.  Prompt length is
        # dynamic and therefore deliberately not folded into this property.
        return self.frame_count

    @property
    def architecture_label(self) -> str:
        base = (
            f"qwenfront_{self.target_name}_o2_{self.frame_count}f{self.token_count}_"
            f"vexpert{self.geometry.parallel_expert_size}_"
            f"alpha{round(self.alpha_min * 100):02d}_no_attention_v1"
        )
        suffixes: list[str] = []
        if self.spatial_readout_mode == "spatial_grid":
            suffixes.append("spatialgrid_v1")
        elif self.spatial_readout_mode == "spatial_multiscale":
            suffixes.append("spatialmultiscale_v1")
        elif self.spatial_readout_mode == "spatial_grid_residual":
            residual_tag = int(round(self.spatial_residual_max * 1000.0))
            suffixes.append(f"spatialgridresidual_rmax{residual_tag:03d}_v1")
        elif self.spatial_readout_mode == "spatial_pyramid_residual":
            residual_tag = int(round(self.spatial_residual_max * 1000.0))
            suffixes.append(f"spatialpyramidresidual_rmax{residual_tag:03d}_v1")
        elif self.spatial_readout_mode == "spatial_deep_residual":
            residual_tag = int(round(self.spatial_residual_max * 1000.0))
            suffixes.append(f"spatialdeepresidual_rmax{residual_tag:03d}_v1")
        if self.quality_adapter_mode == "spatial_conv":
            suffixes.append("qualityconv_v1")
        elif self.quality_adapter_mode == "identity":
            suffixes.append("qualityidentity_v1")
        if self.qwen_gate_enabled:
            qwen_tag = int(round(self.qwen_gate_initial * 100.0))
            suffixes.append(f"qwentrim{qwen_tag:02d}_v1")
        if self.electronic_skip_enabled:
            skip_tag = int(round(self.electronic_skip_initial * 100.0))
            suffixes.append(f"eskip{skip_tag:02d}_v1")
        if self.quality_refiner_enabled:
            suffixes.append("qualityrefine_v1")
        if self.late_input_correction_enabled:
            correction_tag = int(round(self.late_input_correction_max * 100.0))
            suffixes.append(f"lateinputcorr{correction_tag:03d}_v1")
        if self.trainable_frame_stem_enabled:
            suffixes.append("trainableconv5_v1")
        if self.vgg_feature_cache_path is not None:
            correction_tag = int(round(self.vgg_correction_max * 100.0))
            suffixes.append(
                f"plainvgg16corr{correction_tag:02d}_{self.vgg_correction_mode}_v1"
            )
        if self.serial_router_input_size != self.geometry.serial_expert_size:
            suffixes.append(f"sroutercrop{self.serial_router_input_size}_v1")
        if self.serial_router_flatfield_calibration:
            suffixes.append("srouterflatfield_v1")
        if self.serial_router_channel_standardization:
            suffixes.append("srouterstandardize_v1")
        return base if not suffixes else f"{base}_{'_'.join(suffixes)}"

    def validate(self) -> None:
        self.geometry.validate(formal=not self.synthetic)
        if self.target_name not in TARGET_PROMPTS:
            raise ValueError(f"task.target_name must be one of {sorted(TARGET_PROMPTS)}")
        if self.prompt.strip() != TARGET_PROMPTS[self.target_name]:
            raise ValueError(
                f"The {self.target_name} checkpoint must use its exact target-specific "
                "five-level prompt; cross-target prompt reuse is forbidden"
            )
        if self.frame_count not in (4, 9, 16):
            raise ValueError("Formal single-target models support 4, 9, or 16 frames")
        if self.frame_count != self.geometry.lane_grid**2:
            raise ValueError("frame_count must equal lane_grid squared")
        if not self.synthetic:
            allowed_frames = (4,) if self.target_name == "spatial" else (9, 16)
            if self.frame_count not in allowed_frames:
                raise ValueError(
                    f"{self.target_name} formally requires one of {allowed_frames} frames"
                )
        if self.token_grid not in (7, 14):
            raise ValueError("Qwen front token_grid must be 7 or 14")
        if not self.synthetic and self.target_name == "temporal" and self.token_grid != 7:
            raise ValueError("Formal Temporal retains the 7x7 front for bounded storage")
        if (self.vision_input_width, self.language_input_width, self.model_width) != (
            1024,
            2048,
            192,
        ):
            raise ValueError("Formal widths are locked to Vision 1024, Language 2048, model 192")
        expected_quality_width = 192 if self.quality_feature_cache_path is not None else 14
        if self.quality_input_width != expected_quality_width:
            raise ValueError(
                "quality_input_width must be 14 for the fixed bank or 192 when "
                "data.quality_feature_cache is configured"
            )
        if not self.frame_count < self.maximum_language_tokens <= self.geometry.serial_expert_size:
            raise ValueError(
                "maximum_language_tokens must leave room for all frame tokens and fit the 109-row serial field"
            )
        if self.detector_projection_size <= 0:
            raise ValueError("detector_projection_size must be positive")
        if self.spatial_readout_mode not in {
            "statistics",
            "spatial_grid",
            "spatial_multiscale",
            "spatial_grid_residual",
            "spatial_pyramid_residual",
            "spatial_deep_residual",
        }:
            raise ValueError(
                "model.spatial_readout_mode must be statistics, spatial_grid, "
                "spatial_multiscale, spatial_grid_residual, or "
                "spatial_pyramid_residual, or spatial_deep_residual"
            )
        if self.spatial_residual_max <= 0.0:
            raise ValueError("model.spatial_residual_max must be positive")
        if self.quality_adapter_mode not in {"linear", "spatial_conv", "identity"}:
            raise ValueError(
                "model.quality_adapter_mode must be linear, spatial_conv, or identity"
            )
        if (
            self.quality_adapter_mode == "identity"
            and self.quality_input_width != self.model_width
        ):
            raise ValueError(
                "model.quality_adapter_mode=identity requires quality_input_width=model_width"
            )
        if not 0.0 < self.quality_gate_initial < 1.0:
            raise ValueError("model.quality_gate_initial must be within (0,1)")
        if not 0.0 < self.qwen_gate_initial < 1.0:
            raise ValueError("model.qwen_gate_initial must be within (0,1)")
        if self.electronic_skip_max <= 0.0:
            raise ValueError("model.electronic_skip_max must be positive")
        if not 0.0 <= self.electronic_skip_initial < self.electronic_skip_max:
            raise ValueError(
                "model.electronic_skip_initial must be within [0,electronic_skip_max)"
            )
        if self.quality_refiner_max <= 0.0:
            raise ValueError("model.quality_refiner_max must be positive")
        if self.late_input_correction_max <= 0.0:
            raise ValueError("model.late_input_correction_max must be positive")
        if self.late_input_correction_enabled and self.target_name != "spatial":
            raise ValueError("The late input correction is only valid for Spatial")
        if self.trainable_frame_stem_enabled:
            if self.target_name != "spatial" or self.frame_count != 4 or self.token_grid != 14:
                raise ValueError(
                    "The trainable frame stem requires Spatial, four frames, and a 14x14 grid"
                )
            if self.raw_frame_cache_path is None or self.frame_stem_checkpoint is None:
                raise ValueError(
                    "The trainable frame stem requires data.raw_frame_cache and "
                    "initialization.frame_stem_checkpoint"
                )
        if self.vgg_feature_cache_path is not None:
            if self.target_name != "spatial" or self.frame_count != 4 or self.token_grid != 14:
                raise ValueError(
                    "The plain-VGG correction requires Spatial, four frames, and a 14x14 grid"
                )
            if self.vgg_correction_max <= 0.0:
                raise ValueError("model.vgg_correction_max must be positive")
            if self.vgg_correction_mode not in {"local", "context"}:
                raise ValueError("model.vgg_correction_mode must be local or context")
        if not 0 < self.serial_router_input_size <= self.geometry.serial_expert_size:
            raise ValueError(
                "router.serial_input_size must be within the serial expert field"
            )
        if self.target_name != "spatial" and self.spatial_readout_mode != "statistics":
            raise ValueError("The spatial-grid readout is only valid for the Spatial target")
        if self.top_k != 2:
            raise ValueError("The formal router is optical Top-2")
        for intervals, limit in (
            (self.parallel_router_intervals, self.geometry.lane_size),
            (self.serial_router_intervals, self.geometry.active_size),
        ):
            if len(intervals) != 2 or any(len(pair) != 2 for pair in intervals):
                raise ValueError("Each router contract requires two [start,end] intervals")
            if any(not 0 <= int(start) < int(end) <= limit for start, end in intervals):
                raise ValueError("Router detector intervals exceed their optical field")
            if intervals[0][1] > intervals[1][0]:
                raise ValueError("Router detector intervals must not overlap")
        if not 0.0 <= self.alpha_min < self.alpha_initial < self.alpha_max < 1.0:
            raise ValueError("Fusion alpha must satisfy min < initial < max within [0,1)")
        if not self.synthetic and self.alpha_min < 0.30:
            raise ValueError(
                "Relaxed Spatial/Temporal runs still require optical alpha >= 0.30"
            )
        if self.fusion_epsilon <= 0.0:
            raise ValueError("fusion_epsilon must be positive")
        if self.distance_m != 0.10 and not self.synthetic:
            raise ValueError("Formal propagation distance is 10 cm")
        if self.pixel_pitch_um != 17.0 and not self.synthetic:
            raise ValueError("Formal logical pixel pitch is 17 um")
        if self.wavelength_nm != 532.0 and not self.synthetic:
            raise ValueError("Formal wavelength is 532 nm")
        if not (
            0.0
            <= self.unmodulated_power_fraction_min
            <= self.unmodulated_power_fraction_eval
            <= self.unmodulated_power_fraction_max
            < 1.0
        ):
            raise ValueError(
                "Unmodulated power fractions must satisfy "
                "0 <= min <= eval <= max < 1"
            )
        if not self.synthetic and self.unmodulated_power_fraction_min < 0.20:
            raise ValueError("Formal runs require at least 20% nominal unmodulated power")
        if min(
            self.epochs,
            self.batch_size,
            self.num_workers + 1,
            self.test_interval_epochs,
            self.phase_snapshot_interval_epochs,
        ) <= 0:
            raise ValueError("Training counts must be positive (num_workers may be zero)")
        if self.trainable_scope not in {
            "all",
            "readout_only",
            "residual_only",
            "quality_refiner_only",
            "quality_refiner_readout",
            "late_input_correction_only",
            "frame_stem_only",
            "frame_stem_and_readout",
            "vgg_correction_only",
            "vgg_correction_and_readout",
            "vgg_correction_and_vision_path",
            "serial_router_and_readout",
        }:
            raise ValueError(
                "training.trainable_scope must be all, readout_only, residual_only, "
                "quality_refiner_only, quality_refiner_readout, or "
                "late_input_correction_only, frame_stem_only, or "
                "frame_stem_and_readout, vgg_correction_only, or "
                "vgg_correction_and_readout, vgg_correction_and_vision_path, or "
                "serial_router_and_readout"
            )
        if self.trainable_scope == "late_input_correction_only" and not (
            self.late_input_correction_enabled
        ):
            raise ValueError(
                "training.trainable_scope=late_input_correction_only requires "
                "model.late_input_correction_enabled=true"
            )
        if self.trainable_scope.startswith("frame_stem") and not (
            self.trainable_frame_stem_enabled
        ):
            raise ValueError(
                "A frame_stem training scope requires model.trainable_frame_stem_enabled=true"
            )
        if self.trainable_scope.startswith("vgg_correction") and (
            self.vgg_feature_cache_path is None
        ):
            raise ValueError(
                "A vgg_correction training scope requires data.vgg_feature_cache"
            )
        if self.trainable_scope == "residual_only" and self.spatial_readout_mode not in {
            "spatial_grid_residual",
            "spatial_pyramid_residual",
            "spatial_deep_residual",
        }:
            raise ValueError(
                "training.trainable_scope=residual_only requires a residual spatial readout"
            )
        if not self.synthetic and self.phase_snapshot_interval_epochs != 5:
            raise ValueError("Formal runs must save phase-only snapshots every 5 epochs")
        if self.soft_target_weight < 0.0:
            raise ValueError("soft_target_weight must be nonnegative")
        if self.serial_router_balance_weight < 0.0:
            raise ValueError("serial_router_balance_weight must be nonnegative")
        if self.serial_router_importance_weight < 0.0:
            raise ValueError("serial_router_importance_weight must be nonnegative")
        if self.soft_spearman_weight < 0.0:
            raise ValueError("soft_spearman_weight must be nonnegative")
        if self.soft_rank_temperature <= 0.0:
            raise ValueError("soft_rank_temperature must be positive")
        if self.soft_target_weight > 0.0 and self.training_soft_targets_path is None:
            raise ValueError("A positive soft_target_weight requires data.training_soft_targets")


def load_settings(path: str | Path, *, synthetic: bool = False) -> ExperimentSettings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_layered(config_path)
    get = lambda section, key, default=None: raw.get(section, {}).get(key, default)
    target_name = str(get("task", "target_name", get("task", "target", "spatial"))).lower()
    geometry = Geometry(**raw.get("geometry", {}))
    settings = ExperimentSettings(
        config_path=config_path,
        output_dir=_path(raw.get("output_dir", "../../runs/default"), config_path),
        dataset_root=_path(get("data", "dataset_root"), config_path),
        manifest_path=_path(get("data", "manifest"), config_path),
        vision_cache_path=_path(get("data", "vision_cache"), config_path),
        language_cache_path=_path(get("data", "language_cache"), config_path),
        quality_feature_cache_path=_path(
            get("data", "quality_feature_cache"), config_path
        ),
        raw_frame_cache_path=_path(get("data", "raw_frame_cache"), config_path),
        vgg_feature_cache_path=_path(get("data", "vgg_feature_cache"), config_path),
        training_soft_targets_path=_path(get("data", "training_soft_targets"), config_path),
        initialization_checkpoint=_path(get("training", "initialization_checkpoint"), config_path),
        frame_stem_checkpoint=_path(
            get("initialization", "frame_stem_checkpoint"), config_path
        ),
        qwen_model_path=_path(get("initialization", "qwen_model_path"), config_path),
        reset_serial_router_phase_on_initialization=bool(
            get("initialization", "reset_serial_router_phase", False)
        ),
        target_name=target_name,
        prompt=str(get("task", "prompt", TARGET_PROMPTS.get(target_name, ""))),
        device=str(raw.get("device", "cuda")),
        random_seed=int(raw.get("random_seed", 42)),
        geometry=geometry,
        frame_count=int(get("model", "frame_count", 16)),
        token_grid=int(get("model", "token_grid", 7)),
        vision_input_width=int(get("model", "vision_input_width", 1024)),
        quality_input_width=int(get("model", "quality_input_width", 14)),
        language_input_width=int(get("model", "language_input_width", 2048)),
        maximum_language_tokens=int(get("model", "maximum_language_tokens", 96)),
        model_width=int(get("model", "model_width", 192)),
        detector_projection_size=int(get("model", "detector_projection_size", 96)),
        spatial_readout_mode=str(get("model", "spatial_readout_mode", "statistics")),
        spatial_residual_max=float(get("model", "spatial_residual_max", 0.10)),
        quality_adapter_mode=str(get("model", "quality_adapter_mode", "linear")),
        quality_gate_initial=float(get("model", "quality_gate_initial", 0.25)),
        qwen_gate_enabled=bool(get("model", "qwen_gate_enabled", False)),
        qwen_gate_initial=float(get("model", "qwen_gate_initial", 0.10)),
        electronic_skip_enabled=bool(get("model", "electronic_skip_enabled", False)),
        electronic_skip_initial=float(get("model", "electronic_skip_initial", 0.0)),
        electronic_skip_max=float(get("model", "electronic_skip_max", 1.0)),
        quality_refiner_enabled=bool(get("model", "quality_refiner_enabled", False)),
        quality_refiner_max=float(get("model", "quality_refiner_max", 0.50)),
        late_input_correction_enabled=bool(
            get("model", "late_input_correction_enabled", False)
        ),
        late_input_correction_max=float(
            get("model", "late_input_correction_max", 0.50)
        ),
        trainable_frame_stem_enabled=bool(
            get("model", "trainable_frame_stem_enabled", False)
        ),
        vgg_correction_max=float(get("model", "vgg_correction_max", 0.50)),
        vgg_correction_mode=str(get("model", "vgg_correction_mode", "local")),
        head_width=int(get("model", "head_width", 256)),
        dropout=float(get("model", "dropout", 0.15)),
        top_k=int(get("router", "top_k", 2)),
        router_temperature=float(get("router", "temperature", 1.0)),
        router_noise_std=float(get("router", "noise_std", 0.03)),
        serial_router_input_size=int(
            get("router", "serial_input_size", geometry.serial_expert_size)
        ),
        serial_router_flatfield_calibration=bool(
            get("router", "serial_flatfield_calibration", False)
        ),
        serial_router_channel_standardization=bool(
            get("router", "serial_channel_standardization", False)
        ),
        parallel_router_intervals=tuple(tuple(map(int, pair)) for pair in get("router", "parallel_intervals", [[37, 55], [59, 77]])),
        serial_router_intervals=tuple(tuple(map(int, pair)) for pair in get("router", "serial_intervals", [[164, 223], [255, 314]])),
        wavelength_nm=float(get("optics", "wavelength_nm", 532.0)),
        pixel_pitch_um=float(get("optics", "pixel_pitch_um", 17.0)),
        distance_m=float(get("optics", "distance_m", 0.10)),
        k_space_enabled=bool(get("optics", "k_space_enabled", True)),
        theta_max_deg=float(get("optics", "theta_max_deg", 1.0)),
        phase_init_std=float(get("optics", "phase_init_std", 0.25)),
        ccd_relative_clip=float(get("optics", "ccd_relative_clip", 8.0)),
        ccd_log_compression=float(get("optics", "ccd_log_compression", 1.0)),
        unmodulated_power_fraction_min=float(
            get("optics", "unmodulated_power_fraction_min", 0.20)
        ),
        unmodulated_power_fraction_max=float(
            get("optics", "unmodulated_power_fraction_max", 0.35)
        ),
        unmodulated_power_fraction_eval=float(
            get("optics", "unmodulated_power_fraction_eval", 0.20)
        ),
        input_shift_pixels=int(get("robustness", "input_shift_pixels", 8)),
        phase_shift_pixels=int(get("robustness", "phase_shift_pixels", 8)),
        ccd_shift_pixels=int(get("robustness", "ccd_shift_pixels", 8)),
        phase_dropout_p=float(get("robustness", "phase_dropout_p", 0.05)),
        phase_dropout_cell_size=int(get("robustness", "phase_dropout_cell_size", 4)),
        alpha_min=float(get("fusion", "alpha_min", 0.50)),
        alpha_initial=float(get("fusion", "alpha_initial", 0.57)),
        alpha_max=float(get("fusion", "alpha_max", 0.90)),
        fusion_epsilon=float(get("fusion", "epsilon", 1.0e-6)),
        epochs=int(get("training", "epochs", 100)),
        batch_size=int(get("training", "batch_size", 8)),
        num_workers=int(get("training", "num_workers", 4)),
        trainable_scope=str(get("training", "trainable_scope", "all")),
        learning_rate=float(get("training", "learning_rate", 2.0e-4)),
        phase_learning_rate=float(get("training", "phase_learning_rate", 8.0e-3)),
        router_phase_learning_rate=float(get("training", "router_phase_learning_rate", 1.2e-2)),
        weight_decay=float(get("training", "weight_decay", 1.0e-4)),
        ranking_weight=float(get("loss", "ranking_weight", 0.20)),
        correlation_weight=float(get("loss", "correlation_weight", 0.30)),
        soft_spearman_weight=float(get("loss", "soft_spearman_weight", 0.0)),
        soft_rank_temperature=float(get("loss", "soft_rank_temperature", 0.10)),
        optical_alignment_weight=float(get("loss", "optical_alignment_weight", 0.05)),
        router_balance_weight=float(get("loss", "router_balance_weight", 0.02)),
        router_importance_weight=float(get("loss", "router_importance_weight", 0.002)),
        serial_router_balance_weight=float(
            get("loss", "serial_router_balance_weight", 0.0)
        ),
        serial_router_importance_weight=float(
            get("loss", "serial_router_importance_weight", 0.0)
        ),
        router_capture_weight=float(get("loss", "router_capture_weight", 0.02)),
        soft_target_weight=float(get("loss", "soft_target_weight", 0.0)),
        test_interval_epochs=int(get("training", "test_interval_epochs", 5)),
        phase_snapshot_interval_epochs=int(
            get("training", "phase_snapshot_interval_epochs", 5)
        ),
        synthetic=bool(synthetic),
    )
    if settings.output_dir is None:
        raise ValueError("output_dir is required")
    settings.validate()
    return settings


def resolved_dict(settings: ExperimentSettings) -> dict[str, Any]:
    result = asdict(settings)
    for key in (
        "config_path",
        "output_dir",
        "dataset_root",
        "manifest_path",
        "vision_cache_path",
        "language_cache_path",
        "quality_feature_cache_path",
        "raw_frame_cache_path",
        "vgg_feature_cache_path",
        "training_soft_targets_path",
        "initialization_checkpoint",
        "frame_stem_checkpoint",
        "qwen_model_path",
    ):
        result[key] = None if result[key] is None else str(result[key])
    result.update(
        {
            "token_count": settings.token_count,
            "serial_vision_token_count": settings.serial_vision_token_count,
            "architecture_label": settings.architecture_label,
            "feature_contract": feature_contract_for_grid(settings.token_grid),
            "language_contract": LANGUAGE_CONTRACT,
            "quality_contract": quality_contract_for_grid(settings.token_grid),
        }
    )
    return result


__all__ = [
    "ExperimentSettings",
    "FEATURE_CONTRACT",
    "FEATURE_CONTRACT_14",
    "Geometry",
    "LANGUAGE_CONTRACT",
    "QUALITY_CONTRACT",
    "QUALITY_CONTRACT_14",
    "OpticalGeometry",
    "TARGET_PROMPTS",
    "feature_contract_for_grid",
    "load_settings",
    "quality_contract_for_grid",
    "resolved_dict",
]
