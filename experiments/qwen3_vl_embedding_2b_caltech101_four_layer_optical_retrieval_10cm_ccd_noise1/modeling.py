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
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)


SOURCE_ARCHITECTURE = "vision2_language2_moe4_10cm_warmstart5_stage_b_v1"


class CCDNoiseRobustReplacement(FourLayerOpticalReplacement):
    """Same tensors as warmstart5 Stage B, with a new training-time noise law."""

    training_architecture_label = (
        "vision2_language2_moe4_10cm_ccd_truncated_gaussian_noise1"
    )
    # Keeping the exact checkpoint label is intentional: the tensor topology is
    # unchanged, so the audited 81% warm-start checkpoint can be loaded strictly.
    checkpoint_architecture = SOURCE_ARCHITECTURE

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        self.noise_settings = settings
        super().__init__(*args, settings=settings, **kwargs)

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        settings = self.noise_settings
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "initialization": "strict_resume_from_81pct_warmstart5_stage_b",
                "minimum_optical_fusion_coefficient": settings.optical_fusion_minimum,
                "constructed_fusion_before_resume": settings.optical_fusion_initial,
                "ccd_training_noise": {
                    "distribution": settings.language_optical_ccd_noise_distribution,
                    "mean_fraction": settings.language_optical_ccd_noise_mean_fraction,
                    "std_fraction": settings.language_optical_ccd_noise_std_fraction,
                    "min_fraction": settings.language_optical_ccd_noise_min_fraction,
                    "max_fraction": settings.language_optical_ccd_noise_max_fraction,
                    "reference": "per-frame clean mean intensity",
                    "evaluation_noise": False,
                },
                "phase_engagement": {
                    "learning_rate": settings.phase_learning_rate,
                    "phase_dc_weight": settings.lambda_phase_dc,
                    "phase_focus_enabled": settings.phase_focus_enabled,
                },
            }
        )
        return report


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[CCDNoiseRobustReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionTwoBlockOpticalReplacement(
        settings.vision_hidden_size, settings
    ).to(loaded.device)
    language = LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    ).to(loaded.device)
    replacement = CCDNoiseRobustReplacement(
        loaded.model, vision, language, settings=settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


__all__ = [
    "CCDNoiseRobustReplacement",
    "SOURCE_ARCHITECTURE",
    "build_hybrid_student",
    "load_backbone",
]

