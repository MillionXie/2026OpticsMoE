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
FEATURE_CONTRACT = "qwen3vl_front_patch_position_16f_784_mean2x2_196_pool7_49x1024_v1"
LANGUAGE_CONTRACT = "qwen3vl_front_chat_template_embed_tokens_2048_v1"
QUALITY_CONTRACT = "fixed_quality14_16f_adaptive_pool7x7_v1"


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

    The parallel plane is a 4x4 frame layout.  Every 114-pixel lane has a
    two-pixel outer edge on the 478 plane and contains a 2x2 layout of
    54-pixel experts separated by six pixels.  The later serial MoE4 retains
    the measured 109/123 geometry used by the physical setup.
    """

    canvas_size: int = 518
    active_size: int = 478
    lane_grid: int = 4
    lane_size: int = 114
    lane_pitch: int = 120
    parallel_expert_size: int = 54
    parallel_expert_pitch: int = 60
    serial_expert_size: int = 109
    serial_expert_pitch: int = 123

    @property
    def active_margin(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def lane_origins(self) -> tuple[tuple[int, int], ...]:
        axis = tuple(2 + index * self.lane_pitch for index in range(self.lane_grid))
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
        if min(asdict(self).values()) <= 0:
            raise ValueError("All geometry dimensions must be positive")
        if self.canvas_size < self.active_size or (self.canvas_size - self.active_size) % 2:
            raise ValueError("The active plane must be centered on the canvas")
        axis = sorted({top for top, _ in self.lane_origins})
        if len(axis) != self.lane_grid or axis[0] != 2:
            raise ValueError("Parallel lanes must use the audited two-pixel edge")
        if axis[-1] + self.lane_size != self.active_size - 2:
            raise ValueError("The 4x4 parallel lanes must end at active_size-2")
        if self.parallel_expert_pitch + self.parallel_expert_size != self.lane_size:
            raise ValueError("Two 54-pixel experts plus the six-pixel gap must fill a lane")
        if 2 * self.serial_expert_pitch + self.serial_expert_size > self.active_size:
            raise ValueError("Serial MoE4 experts exceed the active plane")
        formal_values = {
            "canvas_size": 518,
            "active_size": 478,
            "lane_grid": 4,
            "lane_size": 114,
            "lane_pitch": 120,
            "parallel_expert_size": 54,
            "parallel_expert_pitch": 60,
            "serial_expert_size": 109,
            "serial_expert_pitch": 123,
        }
        if formal and asdict(self) != formal_values:
            raise ValueError(f"Formal hardware geometry is locked to {formal_values}")


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
    training_soft_targets_path: Path | None = None
    initialization_checkpoint: Path | None = None
    qwen_model_path: Path | None = None
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
    top_k: int = 2
    router_temperature: float = 1.0
    router_noise_std: float = 0.03
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
    alpha_min: float = 0.50
    alpha_initial: float = 0.57
    alpha_max: float = 0.90
    fusion_epsilon: float = 1.0e-6
    head_width: int = 256
    dropout: float = 0.15
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 2.0e-4
    phase_learning_rate: float = 8.0e-3
    router_phase_learning_rate: float = 1.2e-2
    weight_decay: float = 1.0e-4
    ranking_weight: float = 0.20
    correlation_weight: float = 0.30
    optical_alignment_weight: float = 0.05
    router_balance_weight: float = 0.02
    router_importance_weight: float = 0.002
    router_capture_weight: float = 0.02
    soft_target_weight: float = 0.0
    test_interval_epochs: int = 5
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
        return (
            f"qwenfront_{self.target_name}_o2_16f49_vexpert54_"
            f"alpha{round(self.alpha_min * 100):02d}_no_attention_v1"
        )

    def validate(self) -> None:
        self.geometry.validate(formal=not self.synthetic)
        if self.target_name not in TARGET_PROMPTS:
            raise ValueError(f"task.target_name must be one of {sorted(TARGET_PROMPTS)}")
        if self.prompt.strip() != TARGET_PROMPTS[self.target_name]:
            raise ValueError(
                f"The {self.target_name} checkpoint must use its exact target-specific "
                "five-level prompt; cross-target prompt reuse is forbidden"
            )
        if self.frame_count != 16:
            raise ValueError("This experiment requires exactly 16 stratified video frames")
        if self.token_grid != 7 or self.token_count != 49:
            raise ValueError("Qwen front tokens are fixed to a 7x7=49 grid per frame")
        if (
            self.vision_input_width,
            self.quality_input_width,
            self.language_input_width,
            self.model_width,
        ) != (1024, 14, 2048, 192):
            raise ValueError("Formal widths are locked to Vision 1024, quality 14, Language 2048, model 192")
        if not self.frame_count < self.maximum_language_tokens <= self.geometry.serial_expert_size:
            raise ValueError(
                "maximum_language_tokens must leave room for 16 frame tokens and fit the 109-row serial field"
            )
        if self.detector_projection_size <= 0:
            raise ValueError("detector_projection_size must be positive")
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
        if not self.synthetic and self.alpha_min < 0.50:
            raise ValueError("Formal runs require optical alpha >= 0.50")
        if self.fusion_epsilon <= 0.0:
            raise ValueError("fusion_epsilon must be positive")
        if self.distance_m != 0.10 and not self.synthetic:
            raise ValueError("Formal propagation distance is 10 cm")
        if self.pixel_pitch_um != 17.0 and not self.synthetic:
            raise ValueError("Formal logical pixel pitch is 17 um")
        if self.wavelength_nm != 532.0 and not self.synthetic:
            raise ValueError("Formal wavelength is 532 nm")
        if min(self.epochs, self.batch_size, self.num_workers + 1, self.test_interval_epochs) <= 0:
            raise ValueError("Training counts must be positive (num_workers may be zero)")
        if self.soft_target_weight < 0.0:
            raise ValueError("soft_target_weight must be nonnegative")
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
        training_soft_targets_path=_path(get("data", "training_soft_targets"), config_path),
        initialization_checkpoint=_path(get("training", "initialization_checkpoint"), config_path),
        qwen_model_path=_path(get("initialization", "qwen_model_path"), config_path),
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
        head_width=int(get("model", "head_width", 256)),
        dropout=float(get("model", "dropout", 0.15)),
        top_k=int(get("router", "top_k", 2)),
        router_temperature=float(get("router", "temperature", 1.0)),
        router_noise_std=float(get("router", "noise_std", 0.03)),
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
        learning_rate=float(get("training", "learning_rate", 2.0e-4)),
        phase_learning_rate=float(get("training", "phase_learning_rate", 8.0e-3)),
        router_phase_learning_rate=float(get("training", "router_phase_learning_rate", 1.2e-2)),
        weight_decay=float(get("training", "weight_decay", 1.0e-4)),
        ranking_weight=float(get("loss", "ranking_weight", 0.20)),
        correlation_weight=float(get("loss", "correlation_weight", 0.30)),
        optical_alignment_weight=float(get("loss", "optical_alignment_weight", 0.05)),
        router_balance_weight=float(get("loss", "router_balance_weight", 0.02)),
        router_importance_weight=float(get("loss", "router_importance_weight", 0.002)),
        router_capture_weight=float(get("loss", "router_capture_weight", 0.02)),
        soft_target_weight=float(get("loss", "soft_target_weight", 0.0)),
        test_interval_epochs=int(get("training", "test_interval_epochs", 5)),
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
        "training_soft_targets_path",
        "initialization_checkpoint",
        "qwen_model_path",
    ):
        result[key] = None if result[key] is None else str(result[key])
    result.update(
        {
            "token_count": settings.token_count,
            "serial_vision_token_count": settings.serial_vision_token_count,
            "architecture_label": settings.architecture_label,
            "feature_contract": FEATURE_CONTRACT,
            "language_contract": LANGUAGE_CONTRACT,
            "quality_contract": QUALITY_CONTRACT,
        }
    )
    return result


__all__ = [
    "ExperimentSettings",
    "FEATURE_CONTRACT",
    "Geometry",
    "LANGUAGE_CONTRACT",
    "QUALITY_CONTRACT",
    "OpticalGeometry",
    "TARGET_PROMPTS",
    "load_settings",
    "resolved_dict",
]
