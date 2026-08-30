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
    detector_mapping_mode: str
    detector_reference_grid_size: int
    detector_reference_intervals: tuple[tuple[int, int], ...]
    detector_reference_pixel_pitch_um: float
    detector_reference_distance_m: float

    k_space_enabled: bool
    k_space_theta_max_deg: float

    robustness_enabled: bool
    robustness_probability: float
    robustness_warmup_epochs: int
    input_shift_max_px: int
    phase_shift_max_px: int
    pre_ccd_shift_max_px: int

    phase_dropout_p: float
    phase_dropout_block_size: int
    zero_order_enabled: bool
    amplitude_zero_order_intensity_min: float
    amplitude_zero_order_intensity_max: float
    phase_zero_order_intensity_min: float
    phase_zero_order_intensity_max: float
    zero_order_random_relative_phase: bool
    detector_gain_min: float
    detector_gain_max: float
    ccd_noise_distribution: str
    ccd_noise_mean_fraction: float
    ccd_noise_std_fraction: float
    ccd_noise_min_fraction: float
    ccd_noise_max_fraction: float
    robust_validation_trials: int
    require_robust_update_for_selection: bool
    continuation_checkpoint: Path | None

    loss_mode: str
    notebook_full_plane_mse_scale: float
    target_region_mse_weight: float
    background_mse_weight: float
    ccd_postprocess: str

    @staticmethod
    def _scale_edge(edge: int, source: int, target: int) -> int:
        # Explicit round-half-up; Python's banker rounding is not an optical
        # coordinate contract.
        return int(math.floor(float(edge) * float(target) / float(source) + 0.5))

    def detector_bounds(self) -> tuple[tuple[int, int, int, int], ...]:
        if self.detector_mapping_mode == "proportional_active_plane":
            intervals = tuple(
                (
                    self._scale_edge(
                        left, self.detector_reference_grid_size, self.active_size
                    ),
                    self._scale_edge(
                        right, self.detector_reference_grid_size, self.active_size
                    ),
                )
                for left, right in self.detector_reference_intervals
            )
        elif self.detector_mapping_mode == "preserve_reference_angle":
            # The notebook used z=20 cm.  Keeping its coordinates after moving
            # the detector to z=10 cm would require roughly twice the steering
            # angle and places the four ROIs outside the useful 17 um sampled
            # k-space.  Preserve each notebook ROI centre's propagation angle,
            # while keeping the explicitly requested current-plane ROI size.
            reference_center = 0.5 * float(self.detector_reference_grid_size)
            target_center = 0.5 * float(self.active_size)
            offset_scale = (
                self.detector_reference_pixel_pitch_um
                * self.detector_distance_m
                / (
                    self.logical_pixel_pitch_um
                    * self.detector_reference_distance_m
                )
            )
            intervals_list: list[tuple[int, int]] = []
            for left, right in self.detector_reference_intervals:
                source_center = 0.5 * float(left + right)
                mapped_center = target_center + (
                    source_center - reference_center
                ) * offset_scale
                mapped_left = int(
                    math.floor(mapped_center - 0.5 * self.detector_size + 0.5)
                )
                intervals_list.append(
                    (mapped_left, mapped_left + int(self.detector_size))
                )
            intervals = tuple(intervals_list)
        else:  # validation should make this unreachable
            raise RuntimeError(
                f"Unsupported detector mapping mode: {self.detector_mapping_mode}"
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
        if self.detector_mapping_mode not in {
            "proportional_active_plane",
            "preserve_reference_angle",
        }:
            raise ValueError(
                "detector.mapping_mode must be proportional_active_plane or "
                "preserve_reference_angle"
            )
        if min(
            self.detector_reference_pixel_pitch_um,
            self.detector_reference_distance_m,
        ) <= 0.0:
            raise ValueError("Notebook detector reference geometry must be positive")
        widths = {right - left for left, _, right, _ in self.detector_bounds()}
        heights = {bottom - top for _, top, _, bottom in self.detector_bounds()}
        if widths != {self.detector_size} or heights != {self.detector_size}:
            raise ValueError(
                "detector.size must equal the mapped notebook ROI width"
            )
        for left, top, right, bottom in self.detector_bounds():
            if not (0 <= left < right <= self.active_size):
                raise ValueError("Mapped detector x bounds leave the active CCD")
            if not (0 <= top < bottom <= self.active_size):
                raise ValueError("Mapped detector y bounds leave the active CCD")
        if not self.k_space_enabled:
            raise ValueError("The formal v2 contract requires k-space filtering")
        if not 0.0 < self.k_space_theta_max_deg < 90.0:
            raise ValueError("k_space.theta_max_deg must be in (0,90)")
        if not 0.0 <= self.robustness_probability <= 1.0:
            raise ValueError("robustness.probability must be in [0,1]")
        if not 0 <= self.robustness_warmup_epochs < self.epochs:
            raise ValueError("robustness.warmup_epochs must be in [0, epochs)")
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
        if not 0.0 <= self.phase_dropout_p < 1.0:
            raise ValueError("robustness.phase_dropout.p must lie in [0,1)")
        if self.phase_dropout_block_size <= 0:
            raise ValueError("robustness.phase_dropout.block_size must be positive")
        for name, minimum, maximum in (
            (
                "amplitude zero-order",
                self.amplitude_zero_order_intensity_min,
                self.amplitude_zero_order_intensity_max,
            ),
            (
                "phase zero-order",
                self.phase_zero_order_intensity_min,
                self.phase_zero_order_intensity_max,
            ),
        ):
            if not 0.0 <= minimum <= maximum < 1.0:
                raise ValueError(f"{name} intensity fractions must satisfy 0<=min<=max<1")
        if not 0.0 < self.detector_gain_min <= self.detector_gain_max:
            raise ValueError("detector gain range must be positive and ordered")
        if self.ccd_noise_distribution not in {"none", "truncated_biased_gaussian"}:
            raise ValueError(
                "robustness.ccd_noise.distribution must be none or "
                "truncated_biased_gaussian"
            )
        if self.ccd_noise_std_fraction < 0.0:
            raise ValueError("CCD noise standard deviation must be nonnegative")
        if self.ccd_noise_min_fraction > self.ccd_noise_max_fraction:
            raise ValueError("CCD noise truncation bounds are reversed")
        if self.robust_validation_trials <= 0:
            raise ValueError("training.robust_validation_trials must be positive")
        if self.loss_mode not in {
            "notebook_full_plane_mse",
            "legacy_balanced_region_mse",
        }:
            raise ValueError(
                "loss.mode must be notebook_full_plane_mse or "
                "legacy_balanced_region_mse"
            )
        if self.notebook_full_plane_mse_scale <= 0.0:
            raise ValueError("loss.notebook_full_plane_mse_scale must be positive")
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
        result["continuation_checkpoint"] = (
            None
            if self.continuation_checkpoint is None
            else str(self.continuation_checkpoint)
        )
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
    checkpoint_value = d("continuation.checkpoint", None)
    continuation_checkpoint = None
    if checkpoint_value is not None:
        continuation_checkpoint = Path(str(checkpoint_value)).expanduser()
        if not continuation_checkpoint.is_absolute():
            continuation_checkpoint = (config_path.parent / continuation_checkpoint).resolve()
    settings = V2Settings(
        **base_values,
        input_content_size=int(d("optics.input_content_size", 336)),
        detector_mapping_mode=str(
            d("detector.mapping_mode", "proportional_active_plane")
        ),
        detector_reference_grid_size=int(d("detector.reference_grid_size", 400)),
        detector_reference_intervals=intervals,
        detector_reference_pixel_pitch_um=float(
            d("detector.reference_pixel_pitch_um", 16.0)
        ),
        detector_reference_distance_m=float(
            d("detector.reference_distance_m", 0.20)
        ),
        k_space_enabled=bool(d("optics.k_space.enabled", True)),
        k_space_theta_max_deg=float(d("optics.k_space.theta_max_deg", 0.65)),
        robustness_enabled=bool(d("robustness.enabled", True)),
        robustness_probability=float(d("robustness.probability", 0.75)),
        robustness_warmup_epochs=int(d("robustness.warmup_epochs", 0)),
        input_shift_max_px=int(d("robustness.input_shift_max_px", 2)),
        phase_shift_max_px=int(d("robustness.phase_shift_max_px", 2)),
        pre_ccd_shift_max_px=int(d("robustness.pre_ccd_shift_max_px", 2)),
        phase_dropout_p=float(d("robustness.phase_dropout.p", 0.0)),
        phase_dropout_block_size=int(
            d("robustness.phase_dropout.block_size", 8)
        ),
        zero_order_enabled=bool(d("robustness.zero_order.enabled", False)),
        amplitude_zero_order_intensity_min=float(
            d("robustness.zero_order.amplitude_intensity_fraction_min", 0.0)
        ),
        amplitude_zero_order_intensity_max=float(
            d("robustness.zero_order.amplitude_intensity_fraction_max", 0.0)
        ),
        phase_zero_order_intensity_min=float(
            d("robustness.zero_order.phase_intensity_fraction_min", 0.0)
        ),
        phase_zero_order_intensity_max=float(
            d("robustness.zero_order.phase_intensity_fraction_max", 0.0)
        ),
        zero_order_random_relative_phase=bool(
            d("robustness.zero_order.random_relative_phase", True)
        ),
        detector_gain_min=float(d("robustness.detector_gain_min", 1.0)),
        detector_gain_max=float(d("robustness.detector_gain_max", 1.0)),
        ccd_noise_distribution=str(
            d("robustness.ccd_noise.distribution", "none")
        ),
        ccd_noise_mean_fraction=float(
            d("robustness.ccd_noise.mean_fraction", 0.0)
        ),
        ccd_noise_std_fraction=float(
            d("robustness.ccd_noise.std_fraction", 0.0)
        ),
        ccd_noise_min_fraction=float(
            d("robustness.ccd_noise.min_fraction", 0.0)
        ),
        ccd_noise_max_fraction=float(
            d("robustness.ccd_noise.max_fraction", 0.0)
        ),
        robust_validation_trials=int(d("training.robust_validation_trials", 3)),
        require_robust_update_for_selection=bool(
            d("training.require_robust_update_for_selection", False)
        ),
        continuation_checkpoint=continuation_checkpoint,
        loss_mode=str(d("loss.mode", "legacy_balanced_region_mse")),
        notebook_full_plane_mse_scale=float(
            d("loss.notebook_full_plane_mse_scale", 100.0)
        ),
        target_region_mse_weight=float(
            d("loss.target_region_mse_weight", 1.0)
        ),
        background_mse_weight=float(d("loss.background_mse_weight", 0.5)),
        ccd_postprocess=str(d("hardware.ccd.postprocess", "none_raw_linear")),
    )
    settings.validate()
    return settings
