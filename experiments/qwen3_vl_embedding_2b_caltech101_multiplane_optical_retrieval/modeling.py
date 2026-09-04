from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    lengths_from_cu,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.replacement import (
    DeepStackMultimodalReplacement,
)

from .optical_cores import D2NNFivePlaneCore, MultiplaneMoECore, build_physical_core
from .artifacts import save_preview, save_snapshot


class MultiplaneVisionReplacement(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.core = build_physical_core(
            hidden_size, settings.max_visual_tokens, settings
        )
        self.tap_stages: tuple[int, ...] = ()
        self.tap_outputs: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None
        self.spatial_shapes: list[tuple[int, int, int]] | None = None

    def set_image_grid_thw(self, image_grid_thw: torch.Tensor | None) -> None:
        if image_grid_thw is None:
            self.spatial_shapes = None
            return
        self.spatial_shapes = [
            (1, int(height), int(width))
            for frames, height, width in image_grid_thw.detach().cpu().long().tolist()
            for _ in range(int(frames))
        ]

    def compute(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        residual_base: torch.Tensor | None = None,
    ) -> None:
        if residual_base is not None:
            raise RuntimeError("Multiplane Vision explicitly forbids electronic residuals")
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        packed = self.core.forward_groups(list(hidden_states.split(lengths)))
        if packed.shape != hidden_states.shape:
            raise RuntimeError(
                f"Vision multiplane output {tuple(packed.shape)} does not match "
                f"Qwen hidden {tuple(hidden_states.shape)}"
            )
        self.tap_outputs = [packed]
        self.last_output = packed

    def output_for_slot(self, slot: int) -> torch.Tensor:
        if slot == 0 and self.last_output is not None:
            return self.last_output
        raise RuntimeError("Vision multiplane output is unavailable")

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.set_phase_dropout_active(active)

    def set_intermediate_field_capture(self, enabled: bool, sample_count: int = 1) -> None:
        self.core.set_intermediate_field_capture(enabled, sample_count)

    def parameter_breakdown(self) -> dict[str, Any]:
        phases = sum(
            parameter.numel()
            for parameter in self.core.parameters()
            if parameter is not None and parameter.ndim == 2
        )
        return {
            "variant": type(self.core).__name__,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "approximate_matrix_parameters": phases,
        }


class MultiplaneLanguageReplacement(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.core = build_physical_core(
            hidden_size, settings.max_language_tokens, settings
        )
        self.valid_mask: torch.Tensor | None = None
        self.lengths: list[int] = []
        self.deepstack_injection_count = 0

    def set_attention_mask(self, mask: torch.Tensor) -> None:
        self.valid_mask = mask.bool()
        self.lengths = [int(value) for value in mask.long().sum(1).tolist()]
        if not self.lengths or max(self.lengths) > self.core.max_tokens:
            raise RuntimeError(
                f"Language token lengths {self.lengths} exceed optical limit "
                f"{self.core.max_tokens}"
            )

    def set_deepstack_injection_count(self, count: int) -> None:
        self.deepstack_injection_count = int(count)
        if count:
            raise RuntimeError("Multiplane comparison uses one main visual injection only")

    def forward_stage(
        self,
        stage: int,
        hidden_states: torch.Tensor,
        optical_input: torch.Tensor | None = None,
        residual_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual_base is not None:
            raise RuntimeError(
                "Multiplane Language forbids electronic residuals"
            )
        if self.valid_mask is None or self.valid_mask.shape != hidden_states.shape[:2]:
            raise RuntimeError("Call prepare_student_batch before Language optics")
        mask = self.valid_mask.to(hidden_states.device)
        if stage == 0:
            # The shared replacement hook always passes ``optical_input`` at
            # stage 0.  With native_pre_attention disabled this is exactly the
            # incoming Qwen hidden state, not an extra electronic operation.
            source = hidden_states if optical_input is None else optical_input
            if source.shape != hidden_states.shape:
                raise RuntimeError(
                    f"Language optical input {tuple(source.shape)} does not match "
                    f"hidden state {tuple(hidden_states.shape)}"
                )
            groups = [source[index, mask[index]] for index in range(len(mask))]
            self.core.start_staged(groups, ~mask)
        packed = self.core.forward_staged_plane(stage, hidden_states.dtype)
        if packed is None:
            # Identity is only the Qwen hook contract while the complex field
            # travels to the next physical plane; it is not fused into optics.
            return hidden_states
        output = torch.zeros_like(hidden_states)
        output[mask] = packed
        return output

    def retrieval_detector_features(self) -> torch.Tensor:
        readout = self.core.current_detector_readout
        if readout is None:
            raise RuntimeError("Final Language CCD readout is unavailable")
        if readout.ndim != 3 or len(readout) != len(self.lengths):
            raise RuntimeError(f"Unexpected Language CCD shape {tuple(readout.shape)}")
        features = []
        for sample_index, length in enumerate(self.lengths):
            tokens = readout[sample_index, :length]
            features.append(torch.cat((tokens.mean(dim=0), tokens.amax(dim=0))))
        value = torch.stack(features)
        if not torch.isfinite(value).all():
            raise RuntimeError("Language retrieval features contain NaN or Inf")
        return value

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.set_phase_dropout_active(active)

    def set_intermediate_field_capture(self, enabled: bool, sample_count: int = 1) -> None:
        self.core.set_intermediate_field_capture(enabled, sample_count)

    def parameter_breakdown(self) -> dict[str, Any]:
        return {
            "variant": type(self.core).__name__,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


class MultiplaneOpticalReplacement(DeepStackMultimodalReplacement):
    has_optical_phases = True

    def __init__(
        self,
        model: nn.Module,
        vision: MultiplaneVisionReplacement,
        language: MultiplaneLanguageReplacement,
        settings: Any,
    ) -> None:
        self.variant = str(settings.multiplane_variant)
        self.training_architecture_label = f"caltech101_{self.variant}"
        self.checkpoint_architecture = f"caltech101_multiplane_{self.variant}_v1"
        super().__init__(model, vision, language, settings)

    def configure_student_trainability(self) -> None:
        self.vision_surrogate.requires_grad_(True)
        self.language_surrogate.requires_grad_(True)
        # Retrieval reads the final Language CCD directly.  The decoded
        # hidden sequence only satisfies Qwen's layer-hook contract and is
        # deliberately not consumed by the retrieval loss.
        self.language_surrogate.core.output_adapter.requires_grad_(False)
        for module in (self.vision_pre_attention, self.language_pre_attention):
            if module is not None:
                module.requires_grad_(False)

    def router_parameters(self) -> list[nn.Parameter]:
        return [
            *self.vision_surrogate.core.all_router_parameters(),
            *self.language_surrogate.core.all_router_parameters(),
        ]

    def phase_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        for name, surrogate in (
            ("vision", self.vision_surrogate),
            ("language", self.language_surrogate),
        ):
            core = surrogate.core
            if isinstance(core, D2NNFivePlaneCore):
                # The shared trainer has stable CSV columns named
                # ``*_expert`` and ``*_global``.  For D2NN these are logging
                # compatibility slots only: planes 1--4 and plane 5,
                # respectively.  The architecture report remains explicit
                # that all five are ordinary same-aperture D2NN planes.
                groups[f"{name}_expert"] = [
                    layer.raw_phase for layer in core.expert_layers[:-1]
                ]
                groups[f"{name}_global"] = [
                    core.expert_layers[-1].raw_phase
                ]
            elif isinstance(core, MultiplaneMoECore):
                groups[f"{name}_expert"] = [
                    expert.raw_phase
                    for layer in core.expert_layers
                    for expert in layer.experts
                ]
                groups[f"{name}_global"] = [core.global_phase.phase.raw_phase]
            else:
                raise TypeError(f"Unsupported core {type(core).__name__}")
        return groups

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        reference = next(self.vision_surrogate.parameters())
        return {"ccd_operating_point": reference.new_zeros(())}

    def router_response_consistency_loss(self) -> torch.Tensor:
        reference = next(self.vision_surrogate.parameters())
        return reference.new_zeros(())

    def student_architecture_report(self) -> dict[str, Any]:
        is_d2nn = self.variant.startswith("d2nn")
        oeo = "oeo" in self.variant
        dynamic = self.variant == "moe_oeo_dynamic_router"
        return {
            "type": self.training_architecture_label,
            "checkpoint_architecture": self.checkpoint_architecture,
            "original_qwen_frozen": True,
            "qwen_tokenizer_and_embeddings_retained": True,
            "deepstack_enabled": False,
            "electronic_residuals": False,
            "optical_bypass_present": False,
            "task_loss_requires_optical_path": True,
            "explicit_optical_contribution_constraint": False,
            "electronic_interfaces": [
                "frozen Qwen patch/token embeddings",
                "trainable hidden-to-amplitude dimensional interface",
                *([] if is_d2nn else ["trainable electronic Top-k router"]),
                "final CCD conditioning and retrieval readout",
            ],
            "optical_family": "D2NN" if is_d2nn else "MoE4",
            "phase_planes_per_modality": 5,
            "expert_planes_per_modality": 0 if is_d2nn else 4,
            "global_planes_per_modality": 0 if is_d2nn else 1,
            "intermediate_square_law_boundaries": 4 if oeo else 0,
            "final_ccd_boundaries": 1,
            "router_calls_per_modality": 0 if is_d2nn else 4 if dynamic else 1,
            "router_policy": (
                "none"
                if is_d2nn
                else "independent_per_expert_plane"
                if dynamic
                else "fixed_from_input"
            ),
            "oeo_transfer": (
                "square -> full-aperture non-affine normalization -> sigmoid -> zero-phase reload"
                if self.variant == "d2nn_oeo_sigmoid"
                else "square -> per-expert non-affine normalization -> sigmoid -> routing-weighted zero-phase reload"
                if oeo
                else "none"
            ),
        }

    def save_multiplane_phase_snapshot(
        self,
        output_dir: Any,
        *,
        epoch: int,
        train_loss: float,
        weight_variant: str,
    ) -> dict[str, Any]:
        return save_snapshot(
            self,
            output_dir,
            epoch=epoch,
            train_loss=train_loss,
            weight_variant=weight_variant,
        )

    def save_multiplane_phase_preview(self, path: Any, *, title: str) -> None:
        save_preview(self, path, title=title)


def build_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[MultiplaneOpticalReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = MultiplaneVisionReplacement(
        settings.vision_hidden_size, settings
    ).to(loaded.device)
    language = MultiplaneLanguageReplacement(
        settings.text_hidden_size, settings
    ).to(loaded.device)
    replacement = MultiplaneOpticalReplacement(
        loaded.model, vision, language, settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


__all__ = [
    "MultiplaneLanguageReplacement",
    "MultiplaneOpticalReplacement",
    "MultiplaneVisionReplacement",
    "build_student",
    "load_backbone",
]
