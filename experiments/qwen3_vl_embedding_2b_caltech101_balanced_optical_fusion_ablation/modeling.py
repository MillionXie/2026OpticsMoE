from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.modeling import (
    FourLayerOpticalReplacement as RobustFourLayerOpticalReplacement,
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    LanguageTwoBlockOpticalCore,
    LanguageTwoBlockOpticalReplacement,
    VisionTwoBlockOpticalCore,
    VisionTwoBlockOpticalReplacement,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)


SOURCE_ARCHITECTURES = {
    "vision2_language2_moe4_10cm_robust_bounded_fusion_v2",
    "vision2_language2_moe4_10cm_warmstart5_stage_b_v1",
}
ABLATION_MODES = {"none", "remove_optical", "remove_electronic"}


def balanced_checkpoint_architecture(
    fusion_mode: str,
    alpha_minimum: float,
    alpha_maximum: float,
    rms_epsilon: float,
) -> str:
    """Return the architecture contract used to encode/decode gate logits."""
    range_label = (
        f"{float(alpha_minimum):.4f}_{float(alpha_maximum):.4f}".replace(".", "p")
    )
    epsilon_label = f"{float(rms_epsilon):.0e}".replace("-", "m").replace("+", "p")
    return (
        f"caltech101_scale_matched_convex_{fusion_mode}_{range_label}_"
        f"eps{epsilon_label}_v1"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _range_logit(initial: float, minimum: float, maximum: float) -> torch.Tensor:
    position = (float(initial) - float(minimum)) / (
        float(maximum) - float(minimum)
    )
    if not 0.0 < position < 1.0:
        raise ValueError("Initial alpha must be strictly inside its configured range")
    return torch.logit(torch.tensor(position))


def _range_gate(raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return float(minimum) + (float(maximum) - float(minimum)) * torch.sigmoid(raw)


def _dummy_routing(reference: torch.Tensor, expert_count: int) -> dict[str, torch.Tensor]:
    batch = int(reference.shape[0])
    return {
        "selected_mask": torch.zeros(
            batch, expert_count, dtype=torch.bool, device=reference.device
        ),
        "importance": torch.zeros(expert_count, device=reference.device),
        "normalized_entropy": reference.new_zeros(()),
        "balance_loss": reference.new_zeros(()),
        "importance_loss": reference.new_zeros(()),
    }


class _ScaleMatchedFusionMixin:
    """Parameter-free branch calibration with an auditable convex coefficient.

    RMS values are deliberately detached. Consequently a branch cannot reduce
    its effective coefficient merely by inflating or shrinking its norm. The
    exact forward equation (per sample, over every valid token and channel) is

        En = E / stopgrad(rms(E)); On = O / stopgrad(rms(O))
        M = (1-a) En + a On
        F = stopgrad(rms(E)) * M / stopgrad(rms(M)).

    The final common rescale preserves the legacy electronic RMS and prevents
    cancellation/correlation from changing the next block's operating scale.
    At a=0 this reduces algebraically to E. No affine branch calibration is
    learned, and token-to-token magnitude differences are preserved.
    """

    fusion_mode: str
    fusion_alpha_min: float
    fusion_alpha_max: float
    fusion_rms_epsilon: float
    fusion_ablation_mode: str
    last_fusion_diagnostics: dict[str, dict[str, torch.Tensor]]

    def _configure_balanced_fusion(self, settings: Any) -> None:
        self.fusion_mode = str(settings.fusion_mode)
        self.fusion_alpha_min = float(settings.fusion_alpha_min)
        self.fusion_alpha_max = float(settings.fusion_alpha_max)
        # Keep inherited diagnostics truthful; the overridden properties use
        # the full [min,max] interval below.
        self.minimum_optical_fusion = self.fusion_alpha_min
        self.fusion_rms_epsilon = float(settings.fusion_rms_epsilon)
        self.fusion_ablation_mode = "none"
        self.last_fusion_diagnostics = {}
        initial = _range_logit(
            settings.fusion_alpha_initial,
            self.fusion_alpha_min,
            self.fusion_alpha_max,
        )
        with torch.no_grad():
            self.block1_optical_fusion_logit.copy_(initial)
            self.block2_optical_fusion_logit.copy_(initial)

    @property
    def block1_optical_fusion(self) -> torch.Tensor:
        return _range_gate(
            self.block1_optical_fusion_logit,
            self.fusion_alpha_min,
            self.fusion_alpha_max,
        )

    @property
    def block2_optical_fusion(self) -> torch.Tensor:
        return _range_gate(
            self.block2_optical_fusion_logit,
            self.fusion_alpha_min,
            self.fusion_alpha_max,
        )

    def reset_fusion_logits(self, initial: float) -> None:
        value = _range_logit(
            initial, self.fusion_alpha_min, self.fusion_alpha_max
        ).to(self.block1_optical_fusion_logit)
        with torch.no_grad():
            self.block1_optical_fusion_logit.copy_(value)
            self.block2_optical_fusion_logit.copy_(value)

    def set_fusion_ablation(self, mode: str) -> None:
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown fusion ablation mode {mode!r}")
        self.fusion_ablation_mode = mode

    def _fuse(
        self,
        electronic: torch.Tensor,
        optical: torch.Tensor | None,
        alpha: torch.Tensor,
        padding_mask: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        valid = ~padding_mask

        def sample_rms(value: torch.Tensor) -> torch.Tensor:
            mask = valid.unsqueeze(-1).to(value.dtype)
            denominator = (
                valid.sum(dim=1, keepdim=True).to(value.dtype)
                * value.shape[-1]
            ).clamp_min(1.0)
            squared_sum = (value.square() * mask).sum(
                dim=(1, 2), keepdim=False
            ).unsqueeze(-1).unsqueeze(-1)
            return (squared_sum / denominator.unsqueeze(-1)).sqrt().clamp_min(
                self.fusion_rms_epsilon
            )

        electronic32 = electronic.float()
        electronic_rms = sample_rms(electronic32)
        if (
            self.fusion_mode == "electronic_only"
            or self.fusion_ablation_mode == "remove_optical"
        ):
            fused = electronic
            self.last_fusion_diagnostics[stage] = {
                "alpha": alpha.detach(),
                "electronic_coefficient": alpha.detach().new_tensor(1.0),
                "optical_coefficient": alpha.detach().new_tensor(0.0),
                "electronic_rms_before": electronic_rms.mean().detach(),
                "optical_rms_before": alpha.detach().new_zeros(()),
                "pre_optical_to_electronic_rms_ratio": alpha.detach().new_zeros(()),
                "post_optical_to_electronic_rms_ratio": alpha.detach().new_zeros(()),
                "fused_to_electronic_rms_ratio": alpha.detach().new_tensor(1.0),
                "branch_cosine": alpha.detach().new_zeros(()),
            }
            return fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if optical is None or optical.shape != electronic.shape:
            raise RuntimeError("Scale-matched fusion requires shape-matched E and O")

        optical32 = optical.float()
        epsilon = self.fusion_rms_epsilon
        optical_rms = sample_rms(optical32)
        # Detached scale statistics make alpha the only trainable scalar that
        # controls relative branch scale. They are calibration values, not a
        # path through which either branch may game its nominal contribution.
        electronic_scale = electronic_rms.detach()
        optical_scale = optical_rms.detach()
        normalized_electronic = electronic32 / electronic_scale
        normalized_optical = optical32 / optical_scale
        remove_electronic = self.fusion_ablation_mode == "remove_electronic"
        if remove_electronic:
            mixture = normalized_optical
        else:
            mixture = (
                (1.0 - alpha) * normalized_electronic
                + alpha * normalized_optical
            )
        mixture_rms = sample_rms(mixture).detach()
        fused32 = electronic_scale * mixture / mixture_rms
        # The epsilon floor is required for numerical safety, but without this
        # explicit limiting case it would amplify extremely small E when the
        # caller intentionally asks for alpha=0. Formal trainable ranges are
        # strictly positive, so this branch only affects the documented
        # counterfactual endpoint and makes its identity contract exact.
        if not remove_electronic:
            fused32 = torch.where(alpha.detach() == 0, electronic32, fused32)
        fused = fused32.to(electronic.dtype).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )

        mask = valid.unsqueeze(-1).to(electronic32.dtype)
        dot = (electronic32 * optical32 * mask).sum(dim=(1, 2))
        electronic_norm = (electronic32.square() * mask).sum(dim=(1, 2)).sqrt()
        optical_norm = (optical32.square() * mask).sum(dim=(1, 2)).sqrt()
        cosine = dot / (electronic_norm * optical_norm).clamp_min(epsilon)
        fused_rms = sample_rms(fused32)
        electronic_coefficient = (
            alpha.detach().new_zeros(())
            if remove_electronic
            else (1.0 - alpha).detach()
        )
        optical_coefficient = (
            alpha.detach().new_ones(())
            if remove_electronic
            else alpha.detach()
        )
        self.last_fusion_diagnostics[stage] = {
            "alpha": alpha.detach(),
            "electronic_coefficient": electronic_coefficient,
            "optical_coefficient": optical_coefficient,
            "electronic_rms_before": electronic_rms.mean().detach(),
            "optical_rms_before": optical_rms.mean().detach(),
            "pre_optical_to_electronic_rms_ratio": (
                optical_rms / electronic_rms
            ).mean().detach(),
            "post_optical_to_electronic_rms_ratio": alpha.detach().new_tensor(1.0),
            "fused_to_electronic_rms_ratio": (
                fused_rms / electronic_rms
            ).mean().detach(),
            "branch_cosine": cosine.mean().detach(),
        }
        return fused

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        return {
            stage: {
                key: float(value.detach().float().cpu())
                for key, value in values.items()
            }
            for stage, values in self.last_fusion_diagnostics.items()
        }

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        report.update(
            {
                "fusion_mode": self.fusion_mode,
                "fusion_parameterization": (
                    "alpha_min+(alpha_max-alpha_min)*sigmoid(raw_gate)"
                ),
                "fusion_equation": (
                    "rE*((1-alpha)*E/rE+alpha*O/rO)/rms(mixture)"
                ),
                "fusion_rms_scope": "per_sample_all_valid_tokens_and_channels",
                "fusion_scale_statistics_detached": True,
                "fusion_alpha_range": [
                    self.fusion_alpha_min,
                    self.fusion_alpha_max,
                ],
            }
        )
        return report


class BalancedVisionCore(_ScaleMatchedFusionMixin, VisionTwoBlockOpticalCore):
    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(hidden_size, max_tokens, settings)
        self._configure_balanced_fusion(settings)

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        *,
        causal: bool,
        spatial_shapes: list[tuple[int, int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if causal or spatial_shapes is None:
            raise RuntimeError("Vision balanced fusion requires non-causal 2D shapes")
        padded, padding_mask, lengths = self._pad_groups(groups)
        input_latent = self.input_norm(self.input_adapter(padded.float()))
        electronic1 = self.blocks[0](
            input_latent,
            padding_mask=padding_mask,
            causal=False,
            spatial_shapes=spatial_shapes,
        )
        optical_disabled = (
            self.fusion_mode == "electronic_only"
            or self.fusion_ablation_mode == "remove_optical"
        )
        if optical_disabled:
            expert_delta = None
            routing = _dummy_routing(input_latent, 4)
        else:
            expert_delta, routing, optical_lengths = self.optical_branch.run_expert_block(
                input_latent, padding_mask
            )
        fused1 = self._fuse(
            electronic1,
            expert_delta,
            self.block1_optical_fusion,
            padding_mask,
            "block1",
        )
        electronic2 = self.blocks[1](
            fused1,
            padding_mask=padding_mask,
            causal=False,
            spatial_shapes=spatial_shapes,
        )
        if optical_disabled:
            global_delta = None
        else:
            global_input = self.optical_branch.encode_global_input(
                fused1, padding_mask, routing
            )
            global_delta = self.optical_branch.run_global_block(
                global_input,
                optical_lengths,
                padding_mask,
                fused1.dtype,
            )
        latent = self.output_norm(
            self._fuse(
                electronic2,
                global_delta,
                self.block2_optical_fusion,
                padding_mask,
                "block2",
            )
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        residual_gate = torch.sigmoid(self.residual_logit)
        output = padded.float() + residual_gate * self.output_adapter(latent)
        output = output.to(groups[0].dtype)
        self.last_latent_groups = [
            latent[index, :length] for index, length in enumerate(lengths)
        ]
        self.last_routing = routing
        packed = torch.cat(
            [output[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        return packed, latent


class BalancedLanguageCore(_ScaleMatchedFusionMixin, LanguageTwoBlockOpticalCore):
    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(hidden_size, max_tokens, settings)
        self._configure_balanced_fusion(settings)

    def forward_stage_groups(
        self,
        stage: int,
        groups: list[torch.Tensor],
        *,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not causal:
            raise RuntimeError("Language balanced fusion expects causal mixing")
        padded, padding_mask, lengths = self._pad_groups(groups)
        gate = torch.sigmoid(self.residual_logit)
        if stage == 0:
            input_latent = self.input_norm(self.input_adapter(padded.float()))
            electronic1 = self.blocks[0](
                input_latent, padding_mask=padding_mask, causal=True
            )
            optical_disabled = (
                self.fusion_mode == "electronic_only"
                or self.fusion_ablation_mode == "remove_optical"
            )
            if optical_disabled:
                expert_delta = None
                routing = _dummy_routing(input_latent, 4)
                self._stage1_global_input = None
                optical_lengths = lengths
            else:
                expert_delta, routing, optical_lengths = self.optical_branch.run_expert_block(
                    input_latent, padding_mask
                )
            fused1 = self._fuse(
                electronic1,
                expert_delta,
                self.block1_optical_fusion,
                padding_mask,
                "block1",
            )
            if not optical_disabled:
                self._stage1_global_input = self.optical_branch.encode_global_input(
                    fused1, padding_mask, routing
                )
            self._stage1_latent = fused1
            self._stage1_lengths = optical_lengths
            self._stage1_padding_mask = padding_mask
            self.last_routing = routing
            stage1_output = padded.float() + gate * self.output_adapter(fused1)
            stage1_output = stage1_output.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            ).to(groups[0].dtype)
            return self._pack(stage1_output, lengths), fused1
        if stage != 1:
            raise RuntimeError("Language balanced fusion has exactly two stages")
        if (
            self._stage1_latent is None
            or self._stage1_padding_mask is None
            or lengths != self._stage1_lengths
        ):
            raise RuntimeError("Language expert stage must precede global stage")
        if tuple(padding_mask.shape) != tuple(self._stage1_padding_mask.shape):
            raise RuntimeError("Language token layout changed between stages")
        block2_input = self._stage1_latent
        electronic2 = self.blocks[1](
            block2_input, padding_mask=padding_mask, causal=True
        )
        optical_disabled = (
            self.fusion_mode == "electronic_only"
            or self.fusion_ablation_mode == "remove_optical"
        )
        if optical_disabled:
            global_delta = None
        else:
            if self._stage1_global_input is None:
                raise RuntimeError("Optical global input is missing")
            global_delta = self.optical_branch.run_global_block(
                self._stage1_global_input,
                self._stage1_lengths,
                padding_mask,
                block2_input.dtype,
            )
        latent = self.output_norm(
            self._fuse(
                electronic2,
                global_delta,
                self.block2_optical_fusion,
                padding_mask,
                "block2",
            )
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        output = padded.float() + gate * self.output_adapter(latent)
        output = output.to(groups[0].dtype)
        self.last_latent_groups = [
            latent[index, :length] for index, length in enumerate(lengths)
        ]
        self.last_block2_input_groups = [
            block2_input[index, :length].detach()
            for index, length in enumerate(lengths)
        ]
        self.last_electronic_block2_groups = [
            electronic2[index, :length].detach()
            for index, length in enumerate(lengths)
        ]
        return self._pack(output, lengths), latent

    def detector_features_from_cached(
        self, electronic: torch.Tensor, ccd: torch.Tensor
    ) -> torch.Tensor:
        if electronic.ndim != 2 or electronic.shape[-1] != self.width:
            raise ValueError("Cached electronic output must be [L,width]")
        length = electronic.shape[0]
        if length <= 0 or length > self.max_tokens:
            raise ValueError(f"Invalid cached token length {length}")
        mask = torch.ones(1, self.max_tokens, dtype=torch.bool, device=ccd.device)
        mask[:, :length] = False
        electronic_on_device = electronic.to(ccd.device).float()
        if (
            self.fusion_mode == "electronic_only"
            or self.fusion_ablation_mode == "remove_optical"
        ):
            delta = None
        else:
            delta = self.optical_branch.decode_measured_ccd(
                ccd.unsqueeze(0), mask, electronic_on_device.dtype
            )[0, :length]
        padded_e = electronic_on_device.unsqueeze(0)
        padded_o = None if delta is None else delta.unsqueeze(0)
        latent = self.output_norm(
            self._fuse(
                padded_e,
                padded_o,
                self.block2_optical_fusion,
                mask[:, :length],
                "block2_hardware",
            )[0]
        )
        return torch.cat((latent.mean(dim=0), latent.amax(dim=0)), dim=0)

    def detector_features_from_block2_inputs(
        self, block2_input_groups: list[torch.Tensor], ccd: torch.Tensor
    ) -> torch.Tensor:
        if not block2_input_groups or any(
            group.ndim != 2 or group.shape[-1] != self.width
            for group in block2_input_groups
        ):
            raise ValueError(
                "block2_input_groups must be non-empty [L,width] tensors"
            )
        device = ccd.device
        lengths = [len(group) for group in block2_input_groups]
        if any(length <= 0 or length > self.max_tokens for length in lengths):
            raise ValueError(f"Invalid cached token lengths {lengths}")
        if ccd.ndim != 3 or len(ccd) != len(block2_input_groups):
            raise ValueError("Measured CCD batch must match cached Language groups")
        max_length = max(lengths)
        padded = torch.zeros(
            len(block2_input_groups),
            max_length,
            self.width,
            device=device,
            dtype=torch.float32,
        )
        padding_mask = torch.ones(
            len(block2_input_groups), max_length, dtype=torch.bool, device=device
        )
        for index, group in enumerate(block2_input_groups):
            padded[index, : len(group)] = group.to(device=device, dtype=torch.float32)
            padding_mask[index, : len(group)] = False
        electronic = self.blocks[1](
            padded, padding_mask=padding_mask, causal=True
        )
        if (
            self.fusion_mode == "electronic_only"
            or self.fusion_ablation_mode == "remove_optical"
        ):
            delta = None
        else:
            delta = self.optical_branch.decode_measured_ccd(
                ccd, padding_mask, electronic.dtype
            )
        latent = self.output_norm(
            self._fuse(
                electronic,
                delta,
                self.block2_optical_fusion,
                padding_mask,
                "block2_hardware",
            )
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return torch.stack(
            [
                torch.cat(
                    (
                        latent[index, :length].mean(dim=0),
                        latent[index, :length].amax(dim=0),
                    ),
                    dim=0,
                )
                for index, length in enumerate(lengths)
            ],
            dim=0,
        )


class BalancedVisionReplacement(VisionTwoBlockOpticalReplacement):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        nn.Module.__init__(self)
        self.core = BalancedVisionCore(
            hidden_size, settings.max_visual_tokens, settings
        )
        self.tap_stages = ()
        self.tap_outputs = []
        self.last_output = None
        self.spatial_shapes = None


class BalancedLanguageReplacement(LanguageTwoBlockOpticalReplacement):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        nn.Module.__init__(self)
        self.valid_mask = None
        self.lengths = []
        self.deepstack_injection_count = 0
        self.pooling = str(settings.electronic_pooling)
        self.core = BalancedLanguageCore(
            hidden_size, settings.max_language_tokens, settings
        )


class BalancedFusionReplacement(RobustFourLayerOpticalReplacement):
    training_architecture_label = "caltech101_scale_matched_convex_fusion"

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        self.fusion_mode = settings.fusion_mode
        self.fusion_alpha_min = settings.fusion_alpha_min
        self.fusion_alpha_max = settings.fusion_alpha_max
        self.fusion_alpha_initial = settings.fusion_alpha_initial
        super().__init__(*args, settings=settings, **kwargs)
        self.checkpoint_architecture = balanced_checkpoint_architecture(
            self.fusion_mode,
            self.fusion_alpha_min,
            self.fusion_alpha_max,
            settings.fusion_rms_epsilon,
        )
        self.has_optical_phases = self.fusion_mode != "electronic_only"

    def configure_student_trainability(self) -> None:
        super().configure_student_trainability()
        if self.fusion_mode == "electronic_only":
            for core in (
                self.vision_surrogate.core,
                self.language_surrogate.core,
            ):
                core.optical_branch.requires_grad_(False)
                core.block1_optical_fusion_logit.requires_grad_(False)
                core.block2_optical_fusion_logit.requires_grad_(False)

    def router_parameters(self) -> list[torch.nn.Parameter]:
        if self.fusion_mode == "electronic_only":
            return []
        return super().router_parameters()

    def phase_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        if self.fusion_mode == "electronic_only":
            return {}
        return super().phase_parameter_groups()

    def router_losses(self) -> dict[str, torch.Tensor]:
        if self.fusion_mode != "electronic_only":
            return super().router_losses()
        zero = self.vision_surrogate.core.block1_optical_fusion_logit.new_zeros(())
        return {
            "vision_balance": zero,
            "vision_importance": zero,
            "language_balance": zero,
            "language_importance": zero,
        }

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        if self.fusion_mode == "electronic_only":
            zero = self.vision_surrogate.core.block1_optical_fusion_logit.new_zeros(())
            return {"ccd_operating_point": zero}
        return super().auxiliary_losses()

    def set_fusion_ablation(self, mode: str) -> None:
        if self.fusion_mode == "electronic_only" and mode != "none":
            raise ValueError("A trained electronic-only model has no optical branch to ablate")
        self.vision_surrogate.core.set_fusion_ablation(mode)
        self.language_surrogate.core.set_fusion_ablation(mode)

    def reset_fusion_logits(self) -> None:
        for core in (self.vision_surrogate.core, self.language_surrogate.core):
            core.reset_fusion_logits(self.fusion_alpha_initial)

    def fusion_diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.fusion_mode,
            "ablation": self.vision_surrogate.core.fusion_ablation_mode,
            "alpha_range": [self.fusion_alpha_min, self.fusion_alpha_max],
            "vision": self.vision_surrogate.core.fusion_diagnostics(),
            "language": self.language_surrogate.core.fusion_diagnostics(),
        }

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "fusion_mode": self.fusion_mode,
                "fusion_equation": (
                    "En=E/stopgrad(rE); On=O/stopgrad(rO); "
                    "M=(1-alpha)En+alpha*On; "
                    "F=stopgrad(rE)*M/stopgrad(rms(M))"
                ),
                "alpha_range": [self.fusion_alpha_min, self.fusion_alpha_max],
                "alpha_initial": self.fusion_alpha_initial,
                "scale_statistics_detached": True,
                "scale_matching_scope": (
                    "per_sample_over_all_valid_tokens_and_channels"
                ),
                "alpha_zero_is_exact_electronic_identity": True,
            }
        )
        for modality in ("vision", "language"):
            report[modality]["fusion"] = (
                "scale-matched convex fusion with explicit coefficients "
                "(1-alpha, alpha) and common post-fusion RMS restoration"
            )
            report[modality]["alpha_range"] = [
                self.fusion_alpha_min,
                self.fusion_alpha_max,
            ]
        return report


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[BalancedFusionReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = BalancedVisionReplacement(settings.vision_hidden_size, settings).to(
        loaded.device
    )
    language = BalancedLanguageReplacement(settings.text_hidden_size, settings).to(
        loaded.device
    )
    replacement = BalancedFusionReplacement(
        loaded.model, vision, language, settings=settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


def _strict_load_state(
    name: str,
    module: nn.Module,
    source: Mapping[str, torch.Tensor],
) -> None:
    target = module.state_dict()
    if set(target) != set(source):
        raise RuntimeError(
            f"{name} state mismatch: missing={sorted(set(target)-set(source))} "
            f"unexpected={sorted(set(source)-set(target))}"
        )
    for key, value in target.items():
        if tuple(value.shape) != tuple(source[key].shape):
            raise RuntimeError(f"{name}.{key} tensor shape mismatch")
    module.load_state_dict(source, strict=True)


def load_fair_initialization(
    settings: Any,
    replacement: BalancedFusionReplacement,
    readout: ElectronicRetrievalReadout,
) -> dict[str, Any]:
    path = settings.initialization_checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Fair initialization checkpoint is missing: {path}")
    digest = _sha256(path)
    if digest != settings.initialization_sha256:
        raise RuntimeError(
            "Fair initialization SHA-256 mismatch: "
            f"expected={settings.initialization_sha256} actual={digest}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    architecture = str(metadata.get("optical_architecture", ""))
    if architecture not in SOURCE_ARCHITECTURES:
        raise RuntimeError(
            f"Unsupported initialization architecture {architecture!r}; "
            f"expected one of {sorted(SOURCE_ARCHITECTURES)}"
        )
    _strict_load_state(
        "vision", replacement.vision_surrogate, payload["vision_optical"]
    )
    _strict_load_state(
        "language", replacement.language_surrogate, payload["language_optical"]
    )
    _strict_load_state("readout", readout, payload["retrieval_readout"])
    replacement.reset_fusion_logits()
    replacement.configure_student_trainability()
    return {
        "path": str(path),
        "sha256": digest,
        "source_architecture": architecture,
        "source_epoch": int(payload["epoch"]),
        "source_train_loss": float(payload["train_loss"]),
        "source_weight_variant": metadata.get("weight_variant"),
        "loaded_tensor_policy": "strict exact keys and shapes",
        "four_fusion_logits_reset": True,
        "alpha_initial": settings.fusion_alpha_initial,
        "alpha_range": [settings.fusion_alpha_min, settings.fusion_alpha_max],
        "optimizer_state_loaded": False,
    }


__all__ = [
    "ABLATION_MODES",
    "BalancedFusionReplacement",
    "BalancedLanguageCore",
    "BalancedVisionCore",
    "build_hybrid_student",
    "load_backbone",
    "load_fair_initialization",
]
