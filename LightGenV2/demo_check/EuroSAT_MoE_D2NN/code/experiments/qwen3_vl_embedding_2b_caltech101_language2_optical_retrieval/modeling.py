from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    VisionElectronicReplacement,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicDeepStackReplacement,
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    load_backbone,
)

from .optical_blocks import LanguageTwoBlockOpticalReplacement


class Language2OpticalReplacement(ElectronicDeepStackReplacement):
    # The legacy Vision+Language artifact writer cannot describe this asymmetric
    # Language-only MoE4 experiment. Hardware artifacts come from hardware_bridge.
    has_optical_phases = False
    training_architecture_label = "language_two_block_moe4_dual_fusion"
    checkpoint_architecture = "language_two_block_moe4_topk2_dual_fusion_v2"

    def __init__(self, *args: Any, freeze_electronic: bool, **kwargs: Any) -> None:
        self.freeze_electronic = bool(freeze_electronic)
        super().__init__(*args, **kwargs)

    def configure_student_trainability(self) -> None:
        super().configure_student_trainability()
        if not self.freeze_electronic:
            return
        self.vision_surrogate.requires_grad_(False)
        core = self.language_surrogate.core
        core.requires_grad_(False)
        core.optical_branch.requires_grad_(True)
        core.block1_optical_fusion_logit.requires_grad_(True)
        core.block2_optical_fusion_logit.requires_grad_(True)

    def set_student_train_mode(self) -> None:
        if not self.freeze_electronic:
            super().set_student_train_mode()
            return
        self.vision_surrogate.eval()
        self.language_surrogate.eval()
        self.language_surrogate.core.optical_branch.train()
        self.language_surrogate.core.optical_branch.set_phase_dropout_active(True)

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        value = self.language_surrogate.core.optical_branch.current_operating_loss
        if value is None:
            value = self.language_surrogate.core.block2_optical_fusion_logit.new_zeros(())
        return {"ccd_operating_point": value}

    def router_parameters(self) -> list[torch.nn.Parameter]:
        """Expose the real MoE4 router nested in the Language optical branch."""
        return list(
            self.language_surrogate.core.optical_branch.core.router.parameters()
        )

    def phase_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Report real nested Language phases while keeping Vision electronic."""
        vision = self.vision_surrogate.core
        optical = self.language_surrogate.core.optical_branch.core
        return {
            "vision_expert": [
                expert.raw_phase
                for layer in vision.expert_layers
                for expert in layer.experts
            ],
            "vision_global": [vision.global_phase.phase.raw_phase],
            "language_expert": [
                expert.raw_phase
                for layer in optical.expert_layers
                for expert in layer.experts
            ],
            "language_global": [optical.global_phase.phase.raw_phase],
        }

    def student_architecture_report(self) -> dict[str, Any]:
        core = self.language_surrogate.core
        return {
            "type": "vision2d_language2_moe4_optical_residual",
            "optical_enabled": True,
            "optical_location": (
                "MoE4 experts in Language Block 1; global phase/CCD in Block 2"
            ),
            "router_enabled": True,
            "router_layout": "2x2 experts, top-k=2",
            "router_loss_enabled": False,
            "deepstack_enabled": False,
            "vision_mixer": "2x depthwise_conv2d_residual_mlp",
            "language_block1": "depthwise_conv1d_residual_mlp",
            "language_block1_optical_path": (
                "Linear(192,224)->Softplus/RMS->MoE4 router(top-k=2)->"
                "2x2 expert phase(224 each)->ASM->CCD478->expert readout->192"
            ),
            "language_block1_fusion": (
                "electronic1 + sigmoid(gate1) * expert_ccd_delta"
            ),
            "language_block2_electronic_path": "depthwise_conv1d_residual_mlp",
            "language_block2_optical_path": (
                "re-encode fused Block1 result->global phase(478 active)->ASM->CCD478->"
                "robust_norm->pool224->Linear(224,192)"
            ),
            "language_block2_fusion": (
                "LN(electronic2 + sigmoid(gate2) * global_ccd_delta)"
            ),
            "fusion_initial": {
                "block1": float(core.block1_optical_fusion.detach()),
                "block2": float(core.block2_optical_fusion.detach()),
            },
            "electronic_frozen": self.freeze_electronic,
            "ccd_normalization": (
                "per-frame mean division (no background subtraction) -> clamp/log1p -> "
                "478-to-224 area pooling -> row LayerNorm; same architecture but "
                "independent readout weights in the two blocks"
            ),
        }


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[Language2OpticalReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionElectronicReplacement(settings.vision_hidden_size, settings).to(
        loaded.device
    )
    language = LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    ).to(loaded.device)
    replacement = Language2OpticalReplacement(
        loaded.model,
        vision,
        language,
        settings,
        freeze_electronic=settings.hybrid_freeze_electronic,
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


def load_electronic_initialization(
    path: str | Path,
    replacement: Language2OpticalReplacement,
    readout: ElectronicRetrievalReadout,
) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"2D electronic initialization checkpoint is missing: {checkpoint}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    replacement.vision_surrogate.load_state_dict(payload["vision_optical"], strict=True)
    missing, unexpected = replacement.language_surrogate.load_state_dict(
        payload["language_optical"], strict=False
    )
    allowed_missing = {
        name
        for name in replacement.language_surrogate.state_dict()
        if name.startswith("core.optical_branch.")
        or name in {
            "core.block1_optical_fusion_logit",
            "core.block2_optical_fusion_logit",
        }
    }
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            "Electronic checkpoint is not compatible with the Language-2 hybrid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    return {
        "path": str(checkpoint),
        "source_epoch": int(payload.get("epoch", -1)),
        "source_train_loss": float(payload.get("train_loss", float("nan"))),
        "new_optical_tensors": sorted(allowed_missing),
    }


__all__ = [
    "Language2OpticalReplacement",
    "build_hybrid_student",
    "load_backbone",
    "load_electronic_initialization",
]
