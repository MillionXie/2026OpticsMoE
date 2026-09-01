from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    AngularSpectrumPropagator,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    QwenStemSlimMixerOpticalImageNetBackbone,
)


class QwenStemDualScaleOpticalImageNetBackbone(
    QwenStemSlimMixerOpticalImageNetBackbone
):
    """P10: four serial local-to-global optical macro blocks.

    The eight phase planes and all electronic modules remain identical to P09.
    Only the fixed propagation kernel alternates between a short-distance local
    operator and the original long-distance broad-receptive-field operator.
    """

    def __init__(self, stem_checkpoint: str | Path, config: dict[str, Any]) -> None:
        super().__init__(stem_checkpoint, config)
        local_distance = float(config.get("local_propagation_distance_m", 0.005))
        global_distance = float(config.get("global_propagation_distance_m", 0.05))
        if not 0.0 < local_distance < global_distance:
            raise ValueError(
                "Dual-scale optics requires 0 < local distance < global distance"
            )
        wavelength = float(config.get("wavelength_m", 5.32e-7))
        pixel_size = float(config.get("pixel_size_m", 1.6e-5))
        schedule: list[dict[str, float | str | int]] = []
        for index, stage in enumerate(self.stages):
            role = "local" if index % 2 == 0 else "global"
            distance = local_distance if role == "local" else global_distance
            stage.propagator = AngularSpectrumPropagator(
                self.canvas_size,
                wavelength,
                pixel_size,
                distance,
            )
            stage.optical_mixing_role = role
            stage.propagation_distance_m = distance
            schedule.append(
                {
                    "stage": index + 1,
                    "role": role,
                    "distance_m": distance,
                }
            )
        self.local_propagation_distance_m = local_distance
        self.global_propagation_distance_m = global_distance
        self.propagation_schedule = schedule
        # A unique persistent key forces strict checkpoint loading to reject
        # P09/P11 weights whose transfer buffers otherwise have identical names
        # and shapes but different physical meanings.
        self.register_buffer(
            "p10_dual_scale_architecture_signature",
            torch.tensor([10, 5, 50, 4], dtype=torch.int64),
            persistent=True,
        )

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "optical_mixer_variant": "dual_scale_serial_local_global",
                "optical_macro_blocks": self.num_stages // 2,
                "local_propagation_distance_m": self.local_propagation_distance_m,
                "global_propagation_distance_m": self.global_propagation_distance_m,
                "propagation_schedule": self.propagation_schedule,
                "adds_trainable_parameters_over_p09": 0,
            }
        )
        return report
