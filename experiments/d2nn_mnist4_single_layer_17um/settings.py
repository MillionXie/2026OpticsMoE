from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config reference involving {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    parent = raw.pop("base_config", None)
    if parent is None:
        return raw
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_config(parent_path, seen), raw)


def _nested(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


@dataclass
class Settings:
    config_path: Path
    dataset_root: Path
    output_dir: Path
    classes: tuple[int, ...]
    download: bool
    val_fraction: float
    train_limit: int | None
    val_limit: int | None
    test_limit: int | None

    wavelength_nm: float
    logical_pixel_pitch_um: float
    canvas_size: int
    active_size: int
    input_size: int
    detector_distance_m: float
    phase_parameterization: str
    phase_init: str
    detector_size: int
    loss_eps: float

    optimizer: str
    phase_learning_rate: float
    min_learning_rate: float
    epochs: int
    batch_size: int
    inference_batch_size: int
    num_workers: int
    random_seed: int
    gradient_clip_norm: float
    log_interval_batches: int
    device: str

    amplitude_slm_size_wh: tuple[int, int]
    amplitude_slm_pixel_pitch_um: float
    amplitude_slm_center_xy: tuple[float, float]
    amplitude_invert_before_export: bool
    phase_slm_size_wh: tuple[int, int]
    phase_slm_pixel_pitch_um: float
    phase_slm_center_xy: tuple[float, float]
    phase_flip_vertical: bool
    phase_flip_horizontal: bool
    ccd_target_size: int
    export_samples_per_class: int

    @property
    def canvas_guard(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def input_guard(self) -> int:
        return (self.active_size - self.input_size) // 2

    def detector_bounds(self) -> tuple[tuple[int, int, int, int], ...]:
        tile = self.active_size // 2
        inset = (tile - self.detector_size) // 2
        result = []
        for row in range(2):
            for column in range(2):
                left = column * tile + inset
                top = row * tile + inset
                result.append(
                    (left, top, left + self.detector_size, top + self.detector_size)
                )
        return tuple(result)

    def validate(self) -> None:
        if self.classes != (0, 1, 2, 3):
            raise ValueError("This experiment requires MNIST classes [0,1,2,3]")
        if (self.canvas_size, self.active_size) != (518, 478):
            raise ValueError("Formal hardware geometry is canvas518/active478")
        if self.canvas_size < self.active_size or (
            self.canvas_size - self.active_size
        ) % 2:
            raise ValueError("canvas_size-active_size must be nonnegative and even")
        if self.active_size < self.input_size or (
            self.active_size - self.input_size
        ) % 2:
            raise ValueError("input_size must be centered inside active_size")
        if self.active_size % 2:
            raise ValueError("active_size must split evenly into a 2x2 detector grid")
        if self.detector_size <= 0 or self.detector_size >= self.active_size // 2:
            raise ValueError("detector_size must fit inside one detector quadrant")
        if self.detector_size % 2 != 1:
            raise ValueError("detector_size must be odd for half-pixel quadrant centers")
        if self.phase_parameterization != "sigmoid" or self.phase_init != "zeros":
            raise ValueError("Formal model requires raw=0 and phase=2pi*sigmoid(raw)")
        if abs(self.wavelength_nm - 532.0) > 1.0e-9:
            raise ValueError("Formal wavelength is 532 nm")
        if abs(self.logical_pixel_pitch_um - 17.0) > 1.0e-9:
            raise ValueError("Formal logical sampling is 17 um")
        if abs(self.detector_distance_m - 0.05) > 1.0e-12:
            raise ValueError("Formal phase-to-CCD propagation is 5 cm")
        if abs(self.amplitude_slm_pixel_pitch_um - 17.0) > 1.0e-9:
            raise ValueError("Amplitude SLM pitch must be 17 um")
        if abs(self.phase_slm_pixel_pitch_um - 8.0) > 1.0e-9:
            raise ValueError("Phase SLM pitch must be 8 um")
        if self.ccd_target_size != self.active_size:
            raise ValueError("CCD logical ROI must be 478x478")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be in (0,0.5)")
        if min(
            self.phase_learning_rate,
            self.min_learning_rate,
            self.loss_eps,
            self.gradient_clip_norm,
        ) <= 0:
            raise ValueError("Learning rates, eps and gradient clip must be positive")
        if self.min_learning_rate > self.phase_learning_rate:
            raise ValueError("min_learning_rate cannot exceed phase_learning_rate")
        if self.optimizer != "adam":
            raise ValueError("Formal configuration uses Adam for raw_phase only")
        if min(
            self.epochs,
            self.batch_size,
            self.inference_batch_size,
            self.export_samples_per_class,
        ) <= 0:
            raise ValueError("Training/export counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
        result["detector_bounds_xyxy"] = [list(value) for value in self.detector_bounds()]
        return result


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_config(config_path)
    base = config_path.parent
    d = lambda key, default=None: _nested(raw, key, default)
    settings = Settings(
        config_path=config_path,
        dataset_root=_resolve(d("dataset.root", "../../cache/mnist"), base),
        output_dir=_resolve(d("output_dir", "../../runs/mnist4_single_layer_17um_5cm"), base),
        classes=tuple(int(value) for value in d("dataset.classes", [0, 1, 2, 3])),
        download=bool(d("dataset.download", True)),
        val_fraction=float(d("dataset.val_fraction", 0.1)),
        train_limit=d("dataset.train_limit"),
        val_limit=d("dataset.val_limit"),
        test_limit=d("dataset.test_limit"),
        wavelength_nm=float(d("optics.wavelength_nm", 532.0)),
        logical_pixel_pitch_um=float(d("optics.logical_pixel_pitch_um", 17.0)),
        canvas_size=int(d("optics.canvas_size", 518)),
        active_size=int(d("optics.active_size", 478)),
        input_size=int(d("optics.input_size", 400)),
        detector_distance_m=float(d("optics.detector_distance_m", 0.05)),
        phase_parameterization=str(d("optics.phase.parameterization", "sigmoid")),
        phase_init=str(d("optics.phase.init", "zeros")),
        detector_size=int(d("detector.size", 49)),
        loss_eps=float(d("loss.eps", 1.0e-8)),
        optimizer=str(d("training.optimizer", "adam")),
        phase_learning_rate=float(d("training.phase_learning_rate", 0.02)),
        min_learning_rate=float(d("training.min_learning_rate", 0.002)),
        epochs=int(d("training.epochs", 60)),
        batch_size=int(d("training.batch_size", 256)),
        inference_batch_size=int(d("training.inference_batch_size", 256)),
        num_workers=int(d("training.num_workers", 4)),
        random_seed=int(d("training.random_seed", 42)),
        gradient_clip_norm=float(d("training.gradient_clip_norm", 5.0)),
        log_interval_batches=int(d("training.log_interval_batches", 20)),
        device=str(d("device", "cuda")),
        amplitude_slm_size_wh=tuple(int(v) for v in d("hardware.amplitude_slm.size_wh", [1024, 1024])),
        amplitude_slm_pixel_pitch_um=float(d("hardware.amplitude_slm.pixel_pitch_um", 17.0)),
        amplitude_slm_center_xy=tuple(float(v) for v in d("hardware.amplitude_slm.center_xy", [512.0, 512.0])),
        amplitude_invert_before_export=bool(d("hardware.amplitude_slm.invert_before_export", True)),
        phase_slm_size_wh=tuple(int(v) for v in d("hardware.phase_slm.size_wh", [1920, 1200])),
        phase_slm_pixel_pitch_um=float(d("hardware.phase_slm.pixel_pitch_um", 8.0)),
        phase_slm_center_xy=tuple(float(v) for v in d("hardware.phase_slm.center_xy", [980.0, 590.0])),
        phase_flip_vertical=bool(d("hardware.phase_slm.flip_vertical", True)),
        phase_flip_horizontal=bool(d("hardware.phase_slm.flip_horizontal", False)),
        ccd_target_size=int(d("hardware.ccd.target_size", 478)),
        export_samples_per_class=int(d("hardware.export_samples_per_class", 10)),
    )
    settings.validate()
    return settings

