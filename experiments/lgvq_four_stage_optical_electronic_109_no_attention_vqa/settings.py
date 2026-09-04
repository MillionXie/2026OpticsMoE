from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
class OpticalGeometry:
    canvas_size: int = 518
    active_size: int = 478
    quadrant_size: int = 232
    expert_size: int = 109
    expert_pitch: int = 123

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

    def validate(self, *, formal: bool) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("All geometry dimensions must be positive")
        if self.canvas_size < self.active_size or (self.canvas_size - self.active_size) % 2:
            raise ValueError("The 478 active region must be centered in the canvas")
        if self.expert_pitch + self.expert_size != self.quadrant_size:
            raise ValueError("expert_pitch + expert_size must equal quadrant_size")
        if self.lane_gap < 0:
            raise ValueError("Four frame lanes do not fit the active region")
        if formal and asdict(self) != {
            "canvas_size": 518,
            "active_size": 478,
            "quadrant_size": 232,
            "expert_size": 109,
            "expert_pitch": 123,
        }:
            raise ValueError("Formal geometry is locked to 518/478/232/109/123")


@dataclass
class ExperimentSettings:
    config_path: Path
    output_dir: Path
    dataset_root: Path | None
    manifest_path: Path | None
    frame_cache_path: Path | None
    training_soft_targets_path: Path | None = None
    initialization_checkpoint: Path | None = None
    device: str = "cuda"
    random_seed: int = 42
    geometry: OpticalGeometry = field(default_factory=OpticalGeometry)
    frame_count: int = 4
    frame_size: int = 224
    crop_fraction: float = 0.65
    token_grid: int = 14
    width: int = 192
    bridge_pool: int = 4
    top_k: int = 2
    router_temperature: float = 1.0
    router_noise_std: float = 0.03
    parallel_detector_intervals: tuple[tuple[int, int], tuple[int, int]] = ((79, 108), (124, 153))
    serial_detector_intervals: tuple[tuple[int, int], tuple[int, int]] = ((164, 223), (255, 314))
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
    detector_projection_size: int = 196
    ccd_relative_clip: float = 8.0
    ccd_log_compression: float = 1.0
    alpha_min: float = 0.50
    alpha_max: float = 0.90
    alpha_initial: float = 0.57
    fusion_epsilon: float = 1.0e-6
    head_width: int = 256
    dropout: float = 0.15
    spatial_statistics_pooling: bool = False
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 2.0e-4
    phase_learning_rate: float = 8.0e-3
    router_phase_learning_rate: float = 1.2e-2
    weight_decay: float = 1.0e-4
    ranking_weight: float = 0.20
    correlation_weight: float = 0.30
    spatial_target_weight: float = 1.0
    temporal_target_weight: float = 1.0
    optical_alignment_weight: float = 0.05
    router_balance_weight: float = 0.02
    router_importance_weight: float = 0.002
    router_capture_weight: float = 0.02
    soft_target_weight: float = 0.0
    test_interval_epochs: int = 5
    synthetic: bool = False

    @property
    def token_count(self) -> int:
        return self.token_grid * self.token_grid

    @property
    def serial_token_count(self) -> int:
        return self.frame_count * self.bridge_pool * self.bridge_pool + self.frame_count

    @property
    def architecture_label(self) -> str:
        alpha_floor = round(100.0 * self.alpha_min)
        version = "v4stats" if self.spatial_statistics_pooling else "v3"
        return f"lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha{alpha_floor:02d}_{version}"

    def validate(self) -> None:
        self.geometry.validate(formal=not self.synthetic)
        if self.frame_count != 4:
            raise ValueError("LGVQ sampling uses exactly four frames")
        if self.frame_size != self.token_grid * 16:
            raise ValueError("The three-convolution stem requires frame_size=16*token_grid")
        if not 0.0 < self.crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be within (0,1]")
        if not 0 < self.bridge_pool <= self.token_grid:
            raise ValueError("bridge_pool must be within the electronic token grid")
        if self.serial_token_count > self.geometry.expert_size:
            raise ValueError("Serial tokens must fit the optical input rows")
        if self.width != 192 and not self.synthetic:
            raise ValueError("Formal feature width is 192")
        if self.top_k != 2:
            raise ValueError("The only formal routing contract is optical Top-2")
        if not 0.0 <= self.alpha_min < self.alpha_initial < self.alpha_max < 1.0:
            raise ValueError("Fusion requires min < initial < max within [0,1)")
        if not self.synthetic and self.alpha_min < 0.50:
            raise ValueError("Formal runs require optical alpha >= 0.50")
        if self.phase_dropout_cell_size <= 0 or not 0.0 <= self.phase_dropout_p < 1.0:
            raise ValueError("Invalid phase dropout settings")
        if self.input_shift_pixels < 0 or self.phase_shift_pixels < 0 or self.ccd_shift_pixels < 0:
            raise ValueError("Optical shifts must be nonnegative")
        if self.distance_m != 0.10 and not self.synthetic:
            raise ValueError("Formal propagation distance is 10 cm")
        if self.pixel_pitch_um != 17.0 and not self.synthetic:
            raise ValueError("Formal logical pixel pitch is 17 um")
        if self.wavelength_nm != 532.0 and not self.synthetic:
            raise ValueError("Formal wavelength is 532 nm")
        for intervals, limit in (
            (self.parallel_detector_intervals, self.geometry.quadrant_size),
            (self.serial_detector_intervals, self.geometry.active_size),
        ):
            if len(intervals) != 2 or any(not 0 <= a < b <= limit for a, b in intervals):
                raise ValueError("Detector intervals are invalid")
        if min(self.epochs, self.batch_size, self.test_interval_epochs) <= 0:
            raise ValueError("Training counts must be positive")
        if self.soft_target_weight < 0.0:
            raise ValueError("soft_target_weight must be nonnegative")
        if self.spatial_target_weight <= 0.0 or self.temporal_target_weight <= 0.0:
            raise ValueError("Spatial and temporal target weights must be positive")
        if self.soft_target_weight > 0.0 and self.training_soft_targets_path is None:
            raise ValueError("A positive soft_target_weight requires data.training_soft_targets")


def load_settings(path: str | Path, *, synthetic: bool = False) -> ExperimentSettings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_layered(config_path)
    get = lambda section, key, default=None: raw.get(section, {}).get(key, default)
    geometry = OpticalGeometry(**raw.get("geometry", {}))
    settings = ExperimentSettings(
        config_path=config_path,
        output_dir=_path(raw.get("output_dir", "../../runs/formal"), config_path),
        dataset_root=_path(get("data", "dataset_root"), config_path),
        manifest_path=_path(get("data", "manifest"), config_path),
        frame_cache_path=_path(get("data", "frame_cache"), config_path),
        training_soft_targets_path=_path(get("data", "training_soft_targets"), config_path),
        initialization_checkpoint=_path(get("training", "initialization_checkpoint"), config_path),
        device=str(raw.get("device", "cuda")),
        random_seed=int(raw.get("random_seed", 42)),
        geometry=geometry,
        frame_count=int(get("model", "frame_count", 4)),
        frame_size=int(get("model", "frame_size", 224)),
        crop_fraction=float(get("model", "crop_fraction", 0.65)),
        token_grid=int(get("model", "token_grid", 14)),
        width=int(get("model", "width", 192)),
        bridge_pool=int(get("model", "bridge_pool", 4)),
        top_k=int(get("router", "top_k", 2)),
        router_temperature=float(get("router", "temperature", 1.0)),
        router_noise_std=float(get("router", "noise_std", 0.03)),
        parallel_detector_intervals=tuple(tuple(map(int, x)) for x in get("router", "parallel_detector_intervals", [[79, 108], [124, 153]])),
        serial_detector_intervals=tuple(tuple(map(int, x)) for x in get("router", "serial_detector_intervals", [[164, 223], [255, 314]])),
        wavelength_nm=float(get("optics", "wavelength_nm", 532.0)),
        pixel_pitch_um=float(get("optics", "pixel_pitch_um", 17.0)),
        distance_m=float(get("optics", "distance_m", 0.10)),
        k_space_enabled=bool(get("optics", "k_space_enabled", True)),
        theta_max_deg=float(get("optics", "theta_max_deg", 1.0)),
        input_shift_pixels=int(get("robustness", "input_shift_pixels", 8)),
        phase_shift_pixels=int(get("robustness", "phase_shift_pixels", 8)),
        ccd_shift_pixels=int(get("robustness", "ccd_shift_pixels", 8)),
        phase_dropout_p=float(get("robustness", "phase_dropout_p", 0.05)),
        phase_dropout_cell_size=int(get("robustness", "phase_dropout_cell_size", 4)),
        phase_init_std=float(get("optics", "phase_init_std", 0.25)),
        detector_projection_size=int(get("optics", "detector_projection_size", 196)),
        ccd_relative_clip=float(get("optics", "ccd_relative_clip", 8.0)),
        ccd_log_compression=float(get("optics", "ccd_log_compression", 1.0)),
        alpha_min=float(get("fusion", "alpha_min", 0.50)),
        alpha_max=float(get("fusion", "alpha_max", 0.90)),
        alpha_initial=float(get("fusion", "alpha_initial", 0.57)),
        fusion_epsilon=float(get("fusion", "epsilon", 1.0e-6)),
        head_width=int(get("model", "head_width", 256)),
        dropout=float(get("model", "dropout", 0.15)),
        spatial_statistics_pooling=bool(get("model", "spatial_statistics_pooling", False)),
        epochs=int(get("training", "epochs", 50)),
        batch_size=int(get("training", "batch_size", 8)),
        num_workers=int(get("training", "num_workers", 2)),
        learning_rate=float(get("training", "learning_rate", 2.0e-4)),
        phase_learning_rate=float(get("training", "phase_learning_rate", 8.0e-3)),
        router_phase_learning_rate=float(get("training", "router_phase_learning_rate", 1.2e-2)),
        weight_decay=float(get("training", "weight_decay", 1.0e-4)),
        ranking_weight=float(get("loss", "ranking_weight", 0.20)),
        correlation_weight=float(get("loss", "correlation_weight", 0.30)),
        spatial_target_weight=float(get("loss", "spatial_target_weight", 1.0)),
        temporal_target_weight=float(get("loss", "temporal_target_weight", 1.0)),
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
        "config_path", "output_dir", "dataset_root", "manifest_path", "frame_cache_path",
        "training_soft_targets_path", "initialization_checkpoint",
    ):
        result[key] = None if result[key] is None else str(result[key])
    result["token_count"] = settings.token_count
    result["serial_token_count"] = settings.serial_token_count
    result["architecture_label"] = settings.architecture_label
    result["inference_ablation"] = "same checkpoint; optical stages bypassed"
    return result


__all__ = ["ExperimentSettings", "OpticalGeometry", "load_settings", "resolved_dict"]
