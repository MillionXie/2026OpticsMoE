from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


ALLOWED_TARGETS = {"spatiotemporal"}
ALLOWED_ROUTERS = {"electronic", "optical"}


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
    base_name = raw.pop("base_config", None)
    if base_name is None:
        return raw
    base_path = Path(str(base_name)).expanduser()
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
class OpticalGeometry:
    """One 1024 canvas: four 478 lanes, each containing a 2x2 MoE4."""

    canvas_size: int = 1024
    active_size: int = 986
    quadrant_size: int = 478
    expert_size: int = 224
    expert_pitch: int = 254

    @property
    def active_margin(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def lane_gap(self) -> int:
        return self.active_size - 2 * self.quadrant_size

    @property
    def lane_origins(self) -> tuple[tuple[int, int], ...]:
        second = self.quadrant_size + self.lane_gap
        return ((0, 0), (0, second), (second, 0), (second, second))

    def validate(self, *, formal: bool = True) -> None:
        if min(
            self.canvas_size,
            self.active_size,
            self.quadrant_size,
            self.expert_size,
            self.expert_pitch,
        ) <= 0:
            raise ValueError("All optical geometry dimensions must be positive")
        if self.canvas_size < self.active_size or (self.canvas_size - self.active_size) % 2:
            raise ValueError("The active plane must be centered on the canvas")
        if self.expert_pitch + self.expert_size != self.quadrant_size:
            raise ValueError("expert_pitch + expert_size must equal quadrant_size")
        if self.lane_gap < 0:
            raise ValueError("Two frame quadrants do not fit the active plane")
        for top, left in self.lane_origins:
            if top + self.quadrant_size > self.active_size or left + self.quadrant_size > self.active_size:
                raise ValueError("Frame quadrant exceeds active plane")
        if formal and asdict(self) != {
            "canvas_size": 1024,
            "active_size": 986,
            "quadrant_size": 478,
            "expert_size": 224,
            "expert_pitch": 254,
        }:
            raise ValueError(
                "Formal LGVQ runs lock canvas/active/quadrant/expert/pitch to "
                "1024/986/478/224/254"
            )


@dataclass
class ExperimentSettings:
    config_path: Path
    output_dir: Path
    manifest_path: Path | None
    cache_path: Path | None
    source_feature_cache: Path | None
    source_language_cache: Path | None
    optional_sister_checkpoint: Path | None
    qwen_model_path: Path | None
    target: str
    router_backend: str
    top_k: int
    random_seed: int
    device: str
    geometry: OpticalGeometry = field(default_factory=OpticalGeometry)
    frame_count: int = 4
    token_count: int = 196
    token_grid_height: int = 14
    token_grid_width: int = 14
    input_width: int = 1024
    language_input_width: int = 2048
    language_token_count: int = 64
    model_width: int = 192
    detector_projection_size: int = 196
    mixer_expansion: float = 2.0
    mixer_dropout: float = 0.10
    router_pool_size: int = 14
    router_temperature: float = 1.0
    router_noise_std: float = 0.05
    router_weight_normalization: str = "power_l2"
    router_straight_through: bool = True
    router_detector_intervals: tuple[tuple[int, int], tuple[int, int]] = (
        (164, 223),
        (255, 314),
    )
    router_input_shift_pixels: int = 16
    router_phase_shift_pixels: int = 16
    router_ccd_shift_pixels: int = 16
    router_phase_dropout_block_size: int = 8
    router_energy_eps: float = 1.0e-8
    router_score_normalization: str = "standardized_region_energy"
    router_capture_loss_scale: float = 0.10
    optical_distance_m: float = 0.10
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    k_space_enabled: bool = True
    theta_max_deg: float = 1.0
    phase_parameterization: str = "sigmoid"
    phase_init_std: float = 0.15
    phase_dropout_p: float = 0.05
    ccd_relative_clip: float = 8.0
    ccd_log_compression: float = 1.0
    fusion_alpha_min: float = 0.01
    fusion_alpha_max: float = 0.49
    fusion_alpha_initial: float = 0.055
    fusion_rms_epsilon: float = 1.0e-6
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 5.0e-5
    phase_learning_rate: float = 6.0e-3
    router_learning_rate: float = 1.0e-3
    optical_router_phase_learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-4
    ranking_loss_weight: float = 0.20
    router_balance_weight: float = 0.02
    router_importance_weight: float = 0.002
    test_interval_epochs: int = 5
    prompt: str = ""
    synthetic: bool = False

    @property
    def target_names(self) -> tuple[str, ...]:
        return ("spatial", "temporal")

    @property
    def architecture_label(self) -> str:
        return (
            "qwen3vl2b_lgvq_parallel16_balanced_"
            f"{self.router_backend}_k{self.top_k}_powerl2_correctedste_"
            "alpha0p01_0p49_v1"
        )

    def validate(self) -> None:
        self.geometry.validate(formal=not self.synthetic)
        if self.target not in ALLOWED_TARGETS:
            raise ValueError(
                "task.target must be spatiotemporal: the model predicts exactly "
                f"spatial and temporal; alignment is forbidden, got {self.target!r}"
            )
        if self.router_backend not in ALLOWED_ROUTERS:
            raise ValueError(f"router.backend must be one of {sorted(ALLOWED_ROUTERS)}")
        if self.top_k not in (1, 2, 4):
            raise ValueError("router.top_k must be one of 1, 2, 4")
        if self.router_backend == "optical" and self.top_k != 2:
            raise ValueError("The formal optical-router comparison is O2 (top_k=2)")
        if self.router_weight_normalization != "power_l2":
            raise ValueError("Fair E1/E2/E4/O2 comparison requires power_l2")
        if not self.router_straight_through:
            raise ValueError("Corrected straight-through routing must remain enabled")
        intervals = self.router_detector_intervals
        if len(intervals) != 2 or any(len(interval) != 2 for interval in intervals):
            raise ValueError("router.detector_intervals must contain two [start,end] intervals")
        if any(
            not 0 <= int(start) < int(end) <= self.geometry.quadrant_size
            for start, end in intervals
        ):
            raise ValueError("Router detector intervals must lie inside each 478 lane")
        if int(intervals[0][1]) > int(intervals[1][0]):
            raise ValueError("Router detector intervals must not overlap")
        if self.frame_count != 4:
            raise ValueError("LGVQ contract samples exactly four video frames")
        if self.token_count != self.token_grid_height * self.token_grid_width:
            raise ValueError("token_count must equal token_grid_height*token_grid_width")
        if self.token_count > self.geometry.expert_size:
            raise ValueError("Token rows cannot exceed the 224-row amplitude field")
        if not 0 < self.language_token_count + self.frame_count <= 224 or self.language_input_width <= 0:
            raise ValueError("Language cache must fit the 224-row MoE4 input field")
        if any(
            value < 0 or value >= self.geometry.canvas_size
            for value in (
                self.router_input_shift_pixels,
                self.router_phase_shift_pixels,
                self.router_ccd_shift_pixels,
            )
        ):
            raise ValueError("Router shifts must be nonnegative and smaller than the canvas")
        if self.router_phase_dropout_block_size <= 0:
            raise ValueError("Router phase-dropout block size must be positive")
        if self.router_energy_eps <= 0.0:
            raise ValueError("router.energy_eps must be positive")
        if self.router_score_normalization != "standardized_region_energy":
            raise ValueError("Formal optical router uses standardized_region_energy")
        if self.router_capture_loss_scale < 0.0:
            raise ValueError("router.capture_loss_scale must be nonnegative")
        if self.model_width <= 0 or self.detector_projection_size <= 0:
            raise ValueError("Model widths must be positive")
        if not 0.0 <= self.fusion_alpha_min < self.fusion_alpha_initial < self.fusion_alpha_max < 0.5:
            raise ValueError("Balanced alpha must satisfy 0<=min<initial<max<0.5")
        if self.fusion_rms_epsilon <= 0:
            raise ValueError("fusion.rms_epsilon must be positive")
        if self.epochs <= 0 or self.batch_size <= 0 or self.test_interval_epochs <= 0:
            raise ValueError("Training counts must be positive")
        expected_prompt = "Excellent, Good, Fair, Poor, or Bad"
        if expected_prompt not in self.prompt:
            raise ValueError(f"Qwen prompt must retain the five fixed levels: {expected_prompt}")
        required_prompt = (
            "Please evaluate the quality of this video and rate it using one of the "
            "following five levels: Excellent, Good, Fair, Poor, or Bad."
        )
        if self.prompt.strip() != required_prompt:
            raise ValueError("Formal Qwen input uses the exact fixed five-level prompt")
        if "alignment" in self.prompt.lower() or "consistent" in self.prompt.lower():
            raise ValueError("Prompt must not request image-text alignment")


def load_settings(path: str | Path, *, synthetic: bool = False) -> ExperimentSettings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_layered(config_path)
    get = lambda section, key, default=None: raw.get(section, {}).get(key, default)
    target = str(get("task", "target", "spatiotemporal")).lower()
    default_prompt = (
        "Please evaluate the quality of this video and rate it using one of the "
        "following five levels: Excellent, Good, Fair, Poor, or Bad."
    )
    geometry = OpticalGeometry(**raw.get("geometry", {}))
    value = ExperimentSettings(
        config_path=config_path,
        output_dir=_path(raw.get("output_dir", "../../runs/default"), config_path),
        manifest_path=_path(get("data", "manifest"), config_path),
        cache_path=_path(get("data", "cache"), config_path),
        source_feature_cache=_path(get("data", "source_feature_cache"), config_path),
        source_language_cache=_path(get("data", "source_language_cache"), config_path),
        optional_sister_checkpoint=_path(get("initialization", "sister_checkpoint"), config_path),
        qwen_model_path=_path(get("initialization", "qwen_model_path"), config_path),
        target=target,
        router_backend=str(get("router", "backend", "electronic")),
        top_k=int(get("router", "top_k", 2)),
        random_seed=int(raw.get("random_seed", 42)),
        device=str(raw.get("device", "cuda")),
        geometry=geometry,
        frame_count=int(get("model", "frame_count", 4)),
        token_count=int(get("model", "token_count", 196)),
        token_grid_height=int(get("model", "token_grid_height", 14)),
        token_grid_width=int(get("model", "token_grid_width", 14)),
        input_width=int(get("model", "input_width", 1024)),
        language_input_width=int(get("model", "language_input_width", 2048)),
        language_token_count=int(get("model", "language_token_count", 64)),
        model_width=int(get("model", "width", 192)),
        detector_projection_size=int(get("model", "detector_projection_size", 196)),
        mixer_expansion=float(get("model", "mixer_expansion", 2.0)),
        mixer_dropout=float(get("model", "dropout", 0.10)),
        router_pool_size=int(get("router", "pool_size", 14)),
        router_temperature=float(get("router", "temperature", 1.0)),
        router_noise_std=float(get("router", "noise_std", 0.05)),
        router_weight_normalization=str(get("router", "weight_normalization", "power_l2")),
        router_straight_through=bool(get("router", "straight_through", True)),
        router_detector_intervals=tuple(
            tuple(map(int, interval))
            for interval in get("router", "detector_intervals", [[164, 223], [255, 314]])
        ),
        router_input_shift_pixels=int(get("router", "input_shift_pixels", 16)),
        router_phase_shift_pixels=int(get("router", "phase_shift_pixels", 16)),
        router_ccd_shift_pixels=int(get("router", "ccd_shift_pixels", 16)),
        router_phase_dropout_block_size=int(get("router", "phase_dropout_block_size", 8)),
        router_energy_eps=float(get("router", "energy_eps", 1.0e-8)),
        router_score_normalization=str(
            get("router", "score_normalization", "standardized_region_energy")
        ),
        router_capture_loss_scale=float(get("router", "capture_loss_scale", 0.10)),
        optical_distance_m=float(get("optics", "distance_m", 0.10)),
        wavelength_nm=float(get("optics", "wavelength_nm", 532.0)),
        pixel_pitch_um=float(get("optics", "pixel_pitch_um", 17.0)),
        k_space_enabled=bool(get("optics", "k_space_enabled", True)),
        theta_max_deg=float(get("optics", "theta_max_deg", 1.0)),
        phase_parameterization=str(get("optics", "phase_parameterization", "sigmoid")),
        phase_init_std=float(get("optics", "phase_init_std", 0.15)),
        phase_dropout_p=float(get("optics", "phase_dropout_p", 0.05)),
        ccd_relative_clip=float(get("optics", "ccd_relative_clip", 8.0)),
        ccd_log_compression=float(get("optics", "ccd_log_compression", 1.0)),
        fusion_alpha_min=float(get("fusion", "alpha_min", 0.01)),
        fusion_alpha_max=float(get("fusion", "alpha_max", 0.49)),
        fusion_alpha_initial=float(get("fusion", "alpha_initial", 0.055)),
        fusion_rms_epsilon=float(get("fusion", "rms_epsilon", 1.0e-6)),
        epochs=int(get("training", "epochs", 50)),
        batch_size=int(get("training", "batch_size", 8)),
        num_workers=int(get("training", "num_workers", 2)),
        learning_rate=float(get("training", "learning_rate", 5.0e-5)),
        phase_learning_rate=float(get("training", "phase_learning_rate", 6.0e-3)),
        router_learning_rate=float(get("training", "router_learning_rate", 1.0e-3)),
        optical_router_phase_learning_rate=float(
            get("training", "optical_router_phase_learning_rate", 1.0e-2)
        ),
        weight_decay=float(get("training", "weight_decay", 1.0e-4)),
        ranking_loss_weight=float(get("loss", "ranking_weight", 0.20)),
        router_balance_weight=float(get("loss", "router_balance_weight", 0.02)),
        router_importance_weight=float(get("loss", "router_importance_weight", 0.002)),
        test_interval_epochs=int(get("training", "test_interval_epochs", 5)),
        prompt=str(get("task", "prompt", default_prompt)),
        synthetic=bool(synthetic),
    )
    if value.output_dir is None:
        raise ValueError("output_dir is required")
    value.validate()
    return value


def resolved_dict(settings: ExperimentSettings) -> dict[str, Any]:
    value = asdict(settings)
    for key in (
        "config_path",
        "output_dir",
        "manifest_path",
        "cache_path",
        "source_feature_cache",
        "source_language_cache",
        "optional_sister_checkpoint",
        "qwen_model_path",
    ):
        value[key] = None if value[key] is None else str(value[key])
    value["architecture_label"] = settings.architecture_label
    value["alignment_target_enabled"] = False
    value["fusion_equation"] = "F=rE*((1-alpha)*E/rE+alpha*O/rO)/rms(mixture)"
    value["fusion_scale_statistics_detached"] = True
    return value


__all__ = [
    "ALLOWED_ROUTERS",
    "ALLOWED_TARGETS",
    "ExperimentSettings",
    "OpticalGeometry",
    "load_settings",
    "resolved_dict",
]
