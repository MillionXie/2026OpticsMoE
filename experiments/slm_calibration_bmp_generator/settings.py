from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    config_path: Path
    output_dir: Path
    active_size: int
    pixel_pitch_um: float
    wavelength_nm: float
    amplitude_size_wh: tuple[int, int]
    phase_size_wh: tuple[int, int]
    phase_flip_vertical: bool
    checkerboard_block_px: int
    lens_focal_lengths_cm: tuple[float, ...]
    letter: str

    def validate(self) -> None:
        if self.active_size <= 0 or self.checkerboard_block_px <= 0:
            raise ValueError("active_size and checkerboard_block_px must be positive")
        if self.pixel_pitch_um <= 0 or self.wavelength_nm <= 0:
            raise ValueError("pixel_pitch_um and wavelength_nm must be positive")
        for width, height in (self.amplitude_size_wh, self.phase_size_wh):
            if self.active_size > width or self.active_size > height:
                raise ValueError(
                    f"active_size={self.active_size} does not fit canvas {width}x{height}"
                )
        if not self.lens_focal_lengths_cm or any(v <= 0 for v in self.lens_focal_lengths_cm):
            raise ValueError("lens_focal_lengths_cm must contain positive values")
        if len(self.letter) != 1:
            raise ValueError("letter must contain exactly one character")


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Calibration configuration must be a YAML mapping")
    slm = raw.get("slm", {})
    patterns = raw.get("patterns", {})
    settings = Settings(
        config_path=config_path,
        output_dir=_resolve(str(raw["output_dir"]), config_path.parent),
        active_size=int(raw.get("active_size", 956)),
        pixel_pitch_um=float(raw.get("pixel_pitch_um", 8.0)),
        wavelength_nm=float(raw.get("wavelength_nm", 532.0)),
        amplitude_size_wh=tuple(int(v) for v in slm.get("amplitude_size_wh", [1920, 1080])),
        phase_size_wh=tuple(int(v) for v in slm.get("phase_size_wh", [1920, 1200])),
        phase_flip_vertical=bool(slm.get("phase_flip_vertical", True)),
        checkerboard_block_px=int(patterns.get("checkerboard_block_px", 32)),
        lens_focal_lengths_cm=tuple(float(v) for v in patterns.get("lens_focal_lengths_cm", [5.0, 10.0])),
        letter=str(patterns.get("letter", "A")),
    )
    settings.validate()
    return settings

