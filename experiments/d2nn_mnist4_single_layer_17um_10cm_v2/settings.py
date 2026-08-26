from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from experiments.d2nn_mnist4_single_layer_17um_10cm.settings import (
    Settings as BaseSettings,
)
from experiments.d2nn_mnist4_single_layer_17um_10cm.settings import (
    _nested,
    _read_config,
)
from experiments.d2nn_mnist4_single_layer_17um_10cm.settings import (
    load_settings as load_base_settings,
)


@dataclass
class V2Settings(BaseSettings):
    input_content_size: int
    detector_reference_grid_size: int
    detector_reference_intervals: tuple[tuple[int, int], ...]

    k_space_enabled: bool
    k_space_theta_max_deg: float

    robustness_enabled: bool
    robustness_probability: float
    input_shift_max_px: int
    phase_shift_max_px: int
    pre_ccd_shift_max_px: int

    target_region_mse_weight: float
    background_mse_weight: float
    ccd_postprocess: str

    @staticmethod
    def _scale_edge(edge: int, source: int, target: int) -> int:
        # Explicit round-half-up; Python's banker rounding is not an optical
        # coordinate contract.
        return int(math.floor(float(edge) * float(target) / float(source) + 0.5))

    def detector_bounds(self) -> tuple[tuple[int, int, int, int], ...]:
        intervals = tuple(
            (
                self._scale_edge(left, self.detector_reference_grid_size, self.active_size),
                self._scale_edge(right, self.detector_reference_grid_size, self.active_size),
            )
            for left, right in self.detector_reference_intervals
        )
        if len(intervals) != 2:
            raise ValueError("MNIST-4 v2 requires two reference intervals per axis")
        return tuple(
            (left, top, right, bottom)
            for top, bottom in intervals
            for left, right in intervals
        )

    def validate(self) -> None:
        super().validate()
        if self.input_size != 400 or self.input_content_size != 336:
            raise ValueError(
                "The audited notebook input is Resize(336) then zero-pad to 400"
            )
        if self.input_size < self.input_content_size or (
            self.input_size - self.input_content_size
        ) % 2:
            raise ValueError("input_size must center input_content_size exactly")
        if self.detector_reference_grid_size != 400:
            raise ValueError("The audited notebook detector reference grid is 400")
        if self.detector_reference_intervals != ((75, 125), (275, 325)):
            raise ValueError(
                "The audited notebook detector intervals are [75,125) and [275,325)"
            )
        widths = {right - left for left, _, right, _ in self.detector_bounds()}
        heights = {bottom - top for _, top, _, bottom in self.detector_bounds()}
        if widths != {self.detector_size} or heights != {self.detector_size}:
            raise ValueError(
                "detector.size must equal the proportionally mapped notebook ROI"
            )
        if not self.k_space_enabled:
            raise ValueError("The formal v2 contract requires k-space filtering")
        if not 0.0 < self.k_space_theta_max_deg < 90.0:
            raise ValueError("k_space.theta_max_deg must be in (0,90)")
        if not 0.0 <= self.robustness_probability <= 1.0:
            raise ValueError("robustness.probability must be in [0,1]")
        if min(
            self.input_shift_max_px,
            self.phase_shift_max_px,
            self.pre_ccd_shift_max_px,
        ) < 0:
            raise ValueError("Robustness shift limits must be nonnegative")
        if self.robustness_enabled and max(
            self.input_shift_max_px,
            self.phase_shift_max_px,
            self.pre_ccd_shift_max_px,
        ) == 0:
            raise ValueError("Enabled robustness requires at least one nonzero shift")
        if min(self.target_region_mse_weight, self.background_mse_weight) <= 0.0:
            raise ValueError("Both raw-CCD loss weights must be positive")
        if self.ccd_postprocess != "none_raw_linear":
            raise ValueError(
                "CCD postprocess must be none_raw_linear: no normalization, "
                "activation, log compression, or background subtraction"
            )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["detector_reference_intervals"] = [
            list(value) for value in self.detector_reference_intervals
        ]
        result["detector_bounds_xyxy"] = [
            list(value) for value in self.detector_bounds()
        ]
        return result


def load_settings(path: str | Path) -> V2Settings:
    config_path = Path(path).expanduser().resolve()
    base = load_base_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)
    base_values = {field.name: getattr(base, field.name) for field in fields(BaseSettings)}
    intervals = tuple(
        (int(value[0]), int(value[1]))
        for value in d("detector.reference_intervals", [[75, 125], [275, 325]])
    )
    settings = V2Settings(
        **base_values,
        input_content_size=int(d("optics.input_content_size", 336)),
        detector_reference_grid_size=int(d("detector.reference_grid_size", 400)),
        detector_reference_intervals=intervals,
        k_space_enabled=bool(d("optics.k_space.enabled", True)),
        k_space_theta_max_deg=float(d("optics.k_space.theta_max_deg", 0.65)),
        robustness_enabled=bool(d("robustness.enabled", True)),
        robustness_probability=float(d("robustness.probability", 0.75)),
        input_shift_max_px=int(d("robustness.input_shift_max_px", 2)),
        phase_shift_max_px=int(d("robustness.phase_shift_max_px", 2)),
        pre_ccd_shift_max_px=int(d("robustness.pre_ccd_shift_max_px", 2)),
        target_region_mse_weight=float(
            d("loss.target_region_mse_weight", 1.0)
        ),
        background_mse_weight=float(d("loss.background_mse_weight", 0.5)),
        ccd_postprocess=str(d("hardware.ccd.postprocess", "none_raw_linear")),
    )
    settings.validate()
    return settings
