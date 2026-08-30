from __future__ import annotations

from typing import Any

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.modeling import (
    FourLayerOpticalReplacement,
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    LanguageTwoBlockOpticalReplacement,
    VisionTwoBlockOpticalReplacement,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.modeling import (
    STAGE_ARCHITECTURES,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)


class EarlyRobustTradeoffReplacement(FourLayerOpticalReplacement):
    """Same checkpoint topology with earlier coherent-leakage vaccination."""

    training_architecture_label = (
        "vision2_language2_moe4_10cm_early_robust_coherent_zero_order"
    )
    # The shared loader validates this field before restoring any tensor.  The
    # continuation starts from the audited Stage-A checkpoint and preserves
    # exactly the same state topology, so its source contract remains valid.
    checkpoint_architecture = STAGE_ARCHITECTURES["optical_calibration"]

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        self.tradeoff_settings = settings
        super().__init__(*args, settings=settings, **kwargs)

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        settings = self.tradeoff_settings
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "initialization": "strict_resume_from_warmstart5_stage_a_epoch4",
                "tradeoff_variant": settings.tradeoff_variant,
                "minimum_optical_fusion_coefficient": (
                    settings.optical_fusion_minimum
                ),
                "coherent_zero_order_training": {
                    "amplitude_slm_intensity_fraction": [
                        settings.language_optical_amplitude_zero_order_intensity_min,
                        settings.language_optical_amplitude_zero_order_intensity_max,
                    ],
                    "phase_slm_intensity_fraction": [
                        settings.language_optical_phase_zero_order_intensity_min,
                        settings.language_optical_phase_zero_order_intensity_max,
                    ],
                    "random_relative_phase": (
                        settings.language_optical_zero_order_random_relative_phase
                    ),
                    "evaluation_perturbation": False,
                },
            }
        )
        return report


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[EarlyRobustTradeoffReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionTwoBlockOpticalReplacement(
        settings.vision_hidden_size, settings
    ).to(loaded.device)
    language = LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    ).to(loaded.device)
    replacement = EarlyRobustTradeoffReplacement(
        loaded.model, vision, language, settings=settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


__all__ = [
    "EarlyRobustTradeoffReplacement",
    "build_hybrid_student",
    "load_backbone",
]
