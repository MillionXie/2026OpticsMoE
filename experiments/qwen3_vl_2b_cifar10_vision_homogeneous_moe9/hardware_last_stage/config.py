from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class HardwareSettings:
    source_run_dir: Path
    output_dir: Path
    source_config: Path | None = None
    checkpoint_tag: str = "best"
    device: str = "cuda"
    cache_dir: Path | None = None
    local_files_only: bool | None = None
    selection_batch_size: int = 8
    capture_batch_size: int = 1
    num_workers: int = 8
    correct_samples_per_class: int = 20
    correct_candidate_multiplier: int = 4
    random_test_samples: int = 100
    random_exclude_selected_correct: bool = True
    seed: int = 42
    amplitude_slm_width: int = 1920
    amplitude_slm_height: int = 1080
    phase_slm_width: int = 1920
    phase_slm_height: int = 1200
    hardware_pixel_pitch_um: float = 8.0
    copy_student_checkpoints: bool = True
    save_raw_tensors: bool = True
    ccd_manifest: Path | None = None
    ccd_output_dir: Path | None = None
    ccd_image_size: int = 480
    ccd_crop_xywh: list[int] | None = None
    ccd_rotate_quadrants: int = 0
    ccd_flip_horizontal: bool = False
    ccd_flip_vertical: bool = False
    ccd_background_image: Path | None = None
    ccd_epochs: int = 50
    ccd_batch_size: int = 32
    ccd_learning_rate: float = 1e-4
    ccd_weight_decay: float = 5e-4
    ccd_validation_fraction: float = 0.2
    ccd_train_output_adapter: bool = True
    ccd_train_head: bool = True

    def validate(self) -> None:
        if self.checkpoint_tag not in {"best", "last"} and not self.checkpoint_tag.startswith("epoch_"):
            raise ValueError("checkpoint_tag must be best, last, or epoch_XXXX")
        for name in (
            "selection_batch_size", "capture_batch_size", "correct_samples_per_class",
            "correct_candidate_multiplier", "amplitude_slm_width", "amplitude_slm_height",
            "phase_slm_width", "phase_slm_height", "ccd_image_size", "ccd_epochs", "ccd_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or self.random_test_samples < 0:
            raise ValueError("num_workers and random_test_samples must be non-negative")
        if self.hardware_pixel_pitch_um <= 0 or self.ccd_learning_rate <= 0 or self.ccd_weight_decay < 0:
            raise ValueError("Invalid hardware pixel pitch or CCD optimizer settings")
        if not 0.0 < self.ccd_validation_fraction < 1.0:
            raise ValueError("ccd_validation_fraction must be in (0,1)")
        if not self.ccd_train_output_adapter and not self.ccd_train_head:
            raise ValueError("At least one CCD electronic component must be trainable")
        if self.ccd_crop_xywh is not None and (len(self.ccd_crop_xywh) != 4 or any(v < 0 for v in self.ccd_crop_xywh)):
            raise ValueError("ccd_crop_xywh must be [x,y,width,height] with non-negative values")


PATH_FIELDS = {
    "source_run_dir", "output_dir", "source_config", "cache_dir", "ccd_manifest",
    "ccd_output_dir", "ccd_background_image",
}


def load_hardware_settings(path: str | Path) -> HardwareSettings:
    config_path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(HardwareSettings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown hardware config keys: {unknown}")
    for name in PATH_FIELDS:
        value = raw.get(name)
        if value is None:
            continue
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        unresolved = [piece for piece in ("$", "%") if piece in expanded]
        if unresolved:
            raise ValueError(f"{name} contains an unresolved environment variable: {value}")
        candidate = Path(expanded)
        raw[name] = (candidate if candidate.is_absolute() else config_path.parent / candidate).resolve()
    settings = HardwareSettings(**raw)
    if settings.source_config is None:
        settings.source_config = settings.source_run_dir / "config_resolved.json"
    if settings.ccd_output_dir is None:
        settings.ccd_output_dir = settings.output_dir / "ccd_readout"
    settings.validate()
    return settings


def to_jsonable(settings: HardwareSettings) -> dict[str, Any]:
    return {item.name: str(value) if isinstance(value, Path) else value for item in fields(settings) if (value := getattr(settings, item.name)) is not None}

