"""Configuration contract for the 9-video x 4-frame optical VQA graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPORAL_PROMPT = (
    "Please evaluate the temporal quality of this video and rate it using one "
    "of the following five levels: Excellent, Good, Fair, Poor, or Bad."
)


def _repo_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_layered(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config reference: {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base_config", None)
    if base is None:
        return raw
    base_path = Path(str(base))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge(_read_layered(base_path, seen), raw)


@dataclass(frozen=True)
class MultiVideoGeometry:
    canvas_size: int = 518
    active_size: int = 478
    video_grid: int = 3
    video_count: int = 9
    video_tile_size: int = 154
    video_tile_pitch: int = 159
    video_tile_offset: int = 3
    frame_grid: int = 2
    frames_per_video: int = 4
    frame_lane_size: int = 75
    frame_lane_pitch: int = 79
    frame_expert_size: int = 36
    frame_expert_pitch: int = 39
    video_field_size: int = 72
    video_expert_pitch: int = 82
    video_phase_tile_size: int = 150

    @property
    def active_margin(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def video_origins(self) -> tuple[tuple[int, int], ...]:
        axis = tuple(
            self.video_tile_offset + index * self.video_tile_pitch
            for index in range(self.video_grid)
        )
        return tuple((top, left) for top in axis for left in axis)

    @property
    def frame_origins_local(self) -> tuple[tuple[int, int], ...]:
        axis = tuple(index * self.frame_lane_pitch for index in range(self.frame_grid))
        return tuple((top, left) for top in axis for left in axis)

    @property
    def frame_expert_origins_local(self) -> tuple[tuple[int, int], ...]:
        axis = (0, self.frame_expert_pitch)
        return tuple((top, left) for top in axis for left in axis)

    @property
    def video_expert_origins_local(self) -> tuple[tuple[int, int], ...]:
        axis = (0, self.video_expert_pitch)
        return tuple((top, left) for top in axis for left in axis)

    @property
    def video_phase_offset(self) -> int:
        return (self.video_tile_size - self.video_phase_tile_size) // 2

    @property
    def video_field_offset(self) -> int:
        return (self.video_tile_size - self.video_field_size) // 2

    def validate(self, *, synthetic: bool = False) -> None:
        if self.video_count != self.video_grid**2:
            raise ValueError("video_count must equal video_grid squared")
        if self.frames_per_video != self.frame_grid**2:
            raise ValueError("frames_per_video must equal frame_grid squared")
        if self.canvas_size < self.active_size or (self.canvas_size - self.active_size) % 2:
            raise ValueError("The active field must be centered on the propagation canvas")
        used = 2 * self.video_tile_offset + self.video_tile_size + 2 * self.video_tile_pitch
        if used != self.active_size:
            raise ValueError("The 3x3 video layout must exactly span the 478-pixel field")
        if self.frame_lane_size + self.frame_lane_pitch != self.video_tile_size:
            raise ValueError("The 2x2 frame lanes must exactly fill each video tile")
        if self.frame_expert_size + self.frame_expert_pitch != self.frame_lane_size:
            raise ValueError("The four frame experts must exactly fill each frame lane")
        if self.video_field_size + self.video_expert_pitch != self.video_tile_size:
            raise ValueError("The four video experts must exactly fill each video tile")
        if self.video_phase_tile_size > self.video_tile_size:
            raise ValueError("The video phase tile exceeds its optical slot")
        if not synthetic and asdict(self) != asdict(MultiVideoGeometry()):
            raise ValueError("Formal multivideo geometry is fixed at the audited 478-pixel layout")


@dataclass
class MultiVideoSettings:
    config_path: Path
    output_dir: Path
    manifest_path: Path
    vision_cache_path: Path
    language_cache_path: Path
    training_soft_targets_path: Path | None = None
    initialization_checkpoint: Path | None = None
    dataset_root: Path | None = None
    quality_feature_cache_path: Path | None = None
    raw_frame_cache_path: Path | None = None
    vgg_feature_cache_path: Path | None = None
    qwen_model_path: Path | None = None
    geometry: MultiVideoGeometry = field(default_factory=MultiVideoGeometry)
    target_name: str = "temporal"
    prompt: str = TEMPORAL_PROMPT
    device: str = "cuda"
    random_seed: int = 91
    videos_per_field: int = 9
    frame_count: int = 4
    token_grid: int = 7
    vision_input_width: int = 1024
    quality_input_width: int = 14
    language_input_width: int = 2048
    maximum_language_tokens: int = 64
    model_width: int = 192
    detector_projection_size: int = 96
    head_width: int = 512
    dropout: float = 0.10
    quality_gate_initial: float = 0.25
    electronic_skip_enabled: bool = False
    electronic_skip_initial: float = 0.0
    electronic_skip_max: float = 1.0
    top_k: int = 2
    router_temperature: float = 1.10
    router_noise_std: float = 0.06
    frame_router_intervals: tuple[tuple[int, int], tuple[int, int]] = ((17, 34), (41, 58))
    video_router_intervals: tuple[tuple[int, int], tuple[int, int]] = ((31, 67), (87, 123))
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    distance_m: float = 0.10
    k_space_enabled: bool = True
    theta_max_deg: float = 0.5
    input_shift_pixels: int = 4
    phase_shift_pixels: int = 4
    ccd_shift_pixels: int = 4
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
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 3.0e-4
    phase_learning_rate: float = 2.0e-2
    router_phase_learning_rate: float = 3.2e-2
    weight_decay: float = 1.0e-4
    ranking_weight: float = 0.30
    correlation_weight: float = 1.00
    optical_alignment_weight: float = 0.05
    router_balance_weight: float = 0.12
    router_importance_weight: float = 0.04
    router_diversity_weight: float = 0.0
    router_capture_weight: float = 0.02
    guard_energy_weight: float = 0.04
    slot_consistency_weight: float = 0.03
    slot_consistency_interval: int = 4
    soft_target_weight: float = 3.0
    test_interval_epochs: int = 5
    phase_snapshot_interval_epochs: int = 5
    synthetic: bool = False

    @property
    def token_count(self) -> int:
        return self.token_grid**2

    @property
    def architecture_label(self) -> str:
        return "lightgenv2_t06_temporal_multivideo9x4_visualrouter_o6_top2_no_attention_v2"

    # Compatibility attributes consumed by the audited frozen-cache loader.
    @property
    def serial_vision_token_count(self) -> int:
        return self.frame_count

    def validate(self) -> None:
        self.geometry.validate(synthetic=self.synthetic)
        if self.target_name != "temporal":
            raise ValueError("The 9x4 graph is a Temporal single-metric model")
        if self.videos_per_field != 9 or self.frame_count != 4:
            raise ValueError("Formal semantics are exactly 9 videos x 4 frames")
        if self.top_k != 2:
            raise ValueError("The formal optical router is Top-2")
        if self.maximum_language_tokens > self.geometry.video_field_size:
            raise ValueError("Image+prompt token sequence must fit the 72-row video field")
        if not 0 <= self.alpha_min < self.alpha_initial < self.alpha_max < 1:
            raise ValueError("Fusion alpha must satisfy min < initial < max < 1")
        if not (
            0 <= self.unmodulated_power_fraction_min
            <= self.unmodulated_power_fraction_eval
            <= self.unmodulated_power_fraction_max < 1
        ):
            raise ValueError("Invalid unmodulated-power interval")
        if not self.synthetic and self.unmodulated_power_fraction_min < 0.20:
            raise ValueError("Formal runs retain at least 20% nominal DC power")
        if min(self.epochs, self.batch_size, self.test_interval_epochs, self.slot_consistency_interval) <= 0:
            raise ValueError("Training counts must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.router_diversity_weight < 0:
            raise ValueError("router_diversity_weight cannot be negative")
        for intervals, limit in (
            (self.frame_router_intervals, self.geometry.frame_lane_size),
            (self.video_router_intervals, self.geometry.video_tile_size),
        ):
            if len(intervals) != 2 or any(not 0 <= a < b <= limit for a, b in intervals):
                raise ValueError("Router detector intervals exceed their local tile")


def load_settings(path: str | Path, *, synthetic: bool = False) -> MultiVideoSettings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_layered(config_path)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    data = raw.get("data", {})
    model = raw.get("model", {})
    router = raw.get("router", {})
    optics = raw.get("optics", {})
    robustness = raw.get("robustness", {})
    fusion = raw.get("fusion", {})
    training = raw.get("training", {})
    loss = raw.get("loss", {})
    settings = MultiVideoSettings(
        config_path=config_path,
        output_dir=_repo_path(raw.get("output_dir", "LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo9x4")),
        dataset_root=_repo_path(data.get("dataset_root")),
        manifest_path=_repo_path(data["manifest"]),
        vision_cache_path=_repo_path(data["vision_cache"]),
        language_cache_path=_repo_path(data["language_cache"]),
        training_soft_targets_path=_repo_path(data.get("training_soft_targets")),
        initialization_checkpoint=_repo_path(training.get("initialization_checkpoint")),
        geometry=MultiVideoGeometry(**raw.get("geometry", {})),
        target_name=str(raw.get("task", {}).get("target_name", "temporal")),
        prompt=str(raw.get("task", {}).get("prompt", TEMPORAL_PROMPT)),
        device=str(raw.get("device", "cuda")),
        random_seed=int(raw.get("random_seed", 91)),
        **{key: value for key, value in model.items() if key in MultiVideoSettings.__dataclass_fields__},
        **{key: value for key, value in optics.items() if key in MultiVideoSettings.__dataclass_fields__},
        **{key: value for key, value in robustness.items() if key in MultiVideoSettings.__dataclass_fields__},
        **{key: value for key, value in training.items() if key in MultiVideoSettings.__dataclass_fields__ and key != "initialization_checkpoint"},
        top_k=int(router.get("top_k", 2)),
        router_temperature=float(router.get("temperature", 1.10)),
        router_noise_std=float(router.get("noise_std", 0.06)),
        frame_router_intervals=tuple(tuple(v) for v in router.get("frame_intervals", ((17, 34), (41, 58)))),
        video_router_intervals=tuple(tuple(v) for v in router.get("video_intervals", ((31, 67), (87, 123)))),
        alpha_min=float(fusion.get("alpha_min", 0.50)),
        alpha_initial=float(fusion.get("alpha_initial", 0.57)),
        alpha_max=float(fusion.get("alpha_max", 0.90)),
        fusion_epsilon=float(fusion.get("epsilon", 1.0e-6)),
        **{key: value for key, value in loss.items() if key in MultiVideoSettings.__dataclass_fields__},
        synthetic=synthetic,
    )
    settings.validate()
    return settings


def resolved_dict(settings: MultiVideoSettings) -> dict[str, Any]:
    value = asdict(settings)
    for key, item in list(value.items()):
        if isinstance(item, Path):
            value[key] = str(item)
    return value


__all__ = ["MultiVideoGeometry", "MultiVideoSettings", "load_settings", "resolved_dict"]
