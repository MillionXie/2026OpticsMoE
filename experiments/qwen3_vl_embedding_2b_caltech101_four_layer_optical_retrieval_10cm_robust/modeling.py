from __future__ import annotations

from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicDeepStackReplacement,
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    load_backbone,
)

from .optical_blocks import (
    LanguageTwoBlockOpticalReplacement,
    VisionTwoBlockOpticalReplacement,
)


class FourLayerOpticalReplacement(ElectronicDeepStackReplacement):
    """Joint electronic/optical student with four physical CCD boundaries."""

    has_optical_phases = True
    training_architecture_label = "vision2_language2_moe4_10cm_robust_bounded_fusion"
    checkpoint_architecture = "vision2_language2_moe4_10cm_robust_bounded_fusion_v2"

    def configure_student_trainability(self) -> None:
        # Original Qwen stays frozen. Every compact electronic and optical
        # component used by retrieval trains jointly from the first optimizer
        # step. The base class deliberately freezes Language's final hidden
        # adapter because retrieval consumes detector features before it.
        super().configure_student_trainability()
        self.language_surrogate.core.residual_logit.requires_grad_(False)

    def set_student_train_mode(self) -> None:
        """Enable the configured phase vaccination during joint training."""

        super().set_student_train_mode()
        self.set_phase_dropout_active(True)

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        values = [
            self.vision_surrogate.core.optical_branch.current_operating_loss,
            self.language_surrogate.core.optical_branch.current_operating_loss,
        ]
        present = [value for value in values if value is not None]
        if not present:
            zero = self.language_surrogate.core.block2_optical_fusion_logit.new_zeros(())
            return {"ccd_operating_point": zero}
        return {"ccd_operating_point": torch.stack(present).mean()}

    def router_parameters(self) -> list[torch.nn.Parameter]:
        return [
            *self.vision_surrogate.core.optical_branch.core.router.parameters(),
            *self.language_surrogate.core.optical_branch.core.router.parameters(),
        ]

    def phase_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        vision = self.vision_surrogate.core.optical_branch.core
        language = self.language_surrogate.core.optical_branch.core
        return {
            "vision_expert": [
                expert.raw_phase
                for layer in vision.expert_layers
                for expert in layer.experts
            ],
            "vision_global": [vision.global_phase.phase.raw_phase],
            "language_expert": [
                expert.raw_phase
                for layer in language.expert_layers
                for expert in layer.experts
            ],
            "language_global": [language.global_phase.phase.raw_phase],
        }

    def optical_artifact_cores(self) -> dict[str, torch.nn.Module]:
        """Expose nested physical cores to the shared phase artifact writer."""
        return {
            "vision": self.vision_surrogate.core.optical_branch.core,
            "language": self.language_surrogate.core.optical_branch.core,
        }

    def student_architecture_report(self) -> dict[str, Any]:
        vision = self.vision_surrogate.core
        language = self.language_surrogate.core
        return {
            "type": self.training_architecture_label,
            "checkpoint_architecture": self.checkpoint_architecture,
            "initialization": "joint_from_scratch_no_electronic_checkpoint",
            "original_qwen_frozen": True,
            "electronic_and_optical_jointly_trainable": True,
            "deepstack_enabled": False,
            "physical_stage_order": [
                "vision_expert",
                "vision_global",
                "language_expert",
                "language_global",
            ],
            "vision": {
                "input_hidden": vision.hidden_size,
                "width": vision.width,
                "electronic": "2x depthwise_conv2d residual MLP",
                "optical": "MoE4 expert CCD then global CCD",
                "fusion": (
                    "two learned gates with a hard minimum optical fraction "
                    f"of {vision.minimum_optical_fusion:.3f}"
                ),
            },
            "language": {
                "input_hidden": language.hidden_size,
                "width": language.width,
                "electronic": "2x causal depthwise_conv1d residual MLP",
                "optical": "MoE4 expert CCD then global CCD",
                "fusion": (
                    "two learned gates with a hard minimum optical fraction "
                    f"of {language.minimum_optical_fusion:.3f}"
                ),
                "qwen_replacement_indexes": list(self.language_optical_layer_indexes),
            },
            "ccd_readout": (
                "no background subtraction; frame-mean scaling, clip/log1p, "
                "478-to-224 pooling and independent per-stage readout"
            ),
        }


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[FourLayerOpticalReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionTwoBlockOpticalReplacement(
        settings.vision_hidden_size, settings
    ).to(loaded.device)
    language = LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    ).to(loaded.device)
    replacement = FourLayerOpticalReplacement(
        loaded.model, vision, language, settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


__all__ = [
    "FourLayerOpticalReplacement",
    "build_hybrid_student",
    "load_backbone",
]
