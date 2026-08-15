from __future__ import annotations

from typing import Any

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.replacement import (
    DeepStackMultimodalReplacement,
)


class RobustDeepStackMultimodalReplacement(DeepStackMultimodalReplacement):
    """Baseline Qwen hook replacement with robust training perturbations enabled."""

    def __init__(self, model: Any, vision: Any, language: Any, settings: Any) -> None:
        super().__init__(model, vision, language, settings)
        self.phase_dropout_training_enabled = (
            settings.phase_dropout_mode != "none" and settings.phase_dropout_p > 0.0
        )

    def set_student_train_mode(self) -> None:
        super().set_student_train_mode()
        self.set_phase_dropout_active(self.phase_dropout_training_enabled)

    def alignment_specification(self) -> dict[str, Any]:
        result = super().alignment_specification()
        result.update(
            {
                "learnable_optical_residuals": True,
                "residual_identity_scale_trainable": True,
                "phase_dropout_training_enabled": self.phase_dropout_training_enabled,
                "equation": (
                    "E1 = Refine(g1*Input + (1-g1)*CCD(ExpertPhase(Input))); "
                    "E2 = Refine(g2*Input + (1-g2)*CCD(GlobalPhase(E1)))"
                ),
            }
        )
        return result
