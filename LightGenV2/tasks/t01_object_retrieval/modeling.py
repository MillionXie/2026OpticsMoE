"""Caltech101 optical-Router MoE and active-budget-matched D2NN graphs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import (
    BalancedFusionReplacement,
    BalancedLanguageReplacement,
    BalancedVisionReplacement,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.modeling import (
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    MoE4LanguageTwoBlockOpticalPath,
    _translate_with_fill,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.modeling import (
    STAGE_ARCHITECTURES,
    _load_payload,
    _validate_metadata,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.artifacts import (
    save_phase_preview as save_router_phase_preview,
    save_phase_snapshot as save_router_phase_snapshot,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.modeling import (
    load_warmstart5_initialization,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    OpticalDetectorTopKRouter,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    PhaseLayer,
)


def checkpoint_architecture(settings: Any) -> str:
    alpha = (
        f"{settings.fusion_alpha_min:.3f}_{settings.fusion_alpha_max:.3f}"
        .replace(".", "p")
    )
    return (
        f"lightgen_t01_{settings.lightgen_model_variant}_10cm_17um_"
        f"scale_matched_{alpha}_c{settings.router_contract_sha256[:12]}_v1"
    )


def _install_optical_router(surrogate: nn.Module, settings: Any) -> None:
    core = surrogate.core.optical_branch.core
    core.router = OpticalDetectorTopKRouter(core.geometry, settings)


def hard_topk_load_balance_loss(
    routing: Mapping[str, torch.Tensor], *, num_experts: int, top_k: int
) -> torch.Tensor:
    """Penalize deterministic Top-k collapse with a soft-gradient surrogate.

    The forward value is computed from the actual hard expert selections.  Its
    gradient follows the optical detector probabilities, so it can train the
    phase-only Router even though ``topk`` itself is discrete.
    """

    selected = routing["selected_mask"].float()
    probabilities = routing["probabilities"].float()
    hard_load = selected.mean(dim=0) / float(top_k)
    soft_load = probabilities.mean(dim=0)
    load_st = hard_load + soft_load - soft_load.detach()
    return float(num_experts) * load_st.square().sum() - 1.0


class OpticalRouterScaleMatchedReplacement(BalancedFusionReplacement):
    training_architecture_label = "lightgen_t01_optical_router_scale_matched_moe"

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        self.router_backend = "optical"
        self.router_top_k = int(settings.top_k)
        self.router_weight_normalization = str(settings.router_weight_normalization)
        self.router_straight_through = bool(settings.router_straight_through)
        super().__init__(*args, settings=settings, **kwargs)
        self.checkpoint_architecture = checkpoint_architecture(settings)

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        router_parameters = sum(p.numel() for p in self.router_parameters())
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "initialization": (
                    "strict warmstart5 Stage-B EMA body; fresh optical Router "
                    "phases; fusion gates reset"
                ),
                "router": {
                    "backend": "optical_detector_energy",
                    "top_k": self.router_top_k,
                    "weight_normalization": self.router_weight_normalization,
                    "straight_through_gradient": self.router_straight_through,
                    "trainable_phase_parameters": router_parameters,
                    "extra_ccd_exposures": 2,
                },
                "physical_feature_stage_count": 4,
                "physical_capture_count_with_router": 6,
            }
        )
        return report

    def save_multiplane_phase_snapshot(
        self,
        output_dir: Path,
        *,
        epoch: int,
        train_loss: float,
        weight_variant: str,
    ) -> dict[str, Any]:
        return save_router_phase_snapshot(
            self,
            output_dir,
            epoch=epoch,
            train_loss=train_loss,
            weight_variant=weight_variant,
        )

    def save_multiplane_phase_preview(self, path: Path, *, title: str) -> None:
        save_router_phase_preview(self, path, title=title)

    def router_hard_load_balance_loss(self) -> dict[str, torch.Tensor]:
        return {
            name: hard_topk_load_balance_loss(
                surrogate.core.last_routing,
                num_experts=4,
                top_k=self.router_top_k,
            )
            for name, surrogate in (
                ("vision", self.vision_surrogate),
                ("language", self.language_surrogate),
            )
        }


class DensePhasePlane(PhaseLayer):
    """One fully trainable 224x224 phase-only plane."""

    def __init__(self, size: int, settings: Any) -> None:
        # The dense baseline owns a different modulation implementation, but
        # remains a PhaseLayer for the shared phase-DC loss and diagnostics.
        nn.Module.__init__(self)
        self.size = int(size)
        self.parameterization = "sigmoid"
        std = float(settings.language_optical_phase_init_std)
        self.raw_phase = nn.Parameter(torch.empty(self.size, self.size))
        nn.init.normal_(self.raw_phase, mean=0.0, std=std)
        self.dropout_p = float(settings.language_optical_phase_dropout_p)
        self.dropout_block_size = int(
            settings.language_optical_phase_dropout_block_size
        )
        self.dropout_active = False

    def physical_phase(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sigmoid(self.raw_phase)

    def phase(self) -> torch.Tensor:
        return self.physical_phase()

    def modulation(self, batch: int) -> torch.Tensor:
        phase = self.physical_phase().unsqueeze(0).expand(batch, -1, -1)
        if self.training and self.dropout_active and self.dropout_p > 0.0:
            block = self.dropout_block_size
            height = math.ceil(self.size / block)
            width = math.ceil(self.size / block)
            bypass = torch.rand(
                batch, 1, height, width, device=phase.device
            ).lt(self.dropout_p)
            bypass = torch.nn.functional.interpolate(
                bypass.float(), size=(self.size, self.size), mode="nearest"
            )[:, 0].bool()
            phase = torch.where(bypass, torch.zeros_like(phase), phase)
        return torch.exp(1j * phase).to(torch.complex64)

    def set_phase_dropout_active(self, active: bool) -> None:
        self.dropout_active = bool(active)


def _dense_routing(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = int(reference.shape[0])
    device = reference.device
    return {
        "selected_mask": torch.ones(batch, 1, dtype=torch.bool, device=device),
        "weights": torch.ones(batch, 1, device=device),
        "importance": torch.ones(1, device=device),
        "load": torch.ones(1, device=device),
        "normalized_entropy": reference.new_zeros(()),
        "balance_loss": reference.new_zeros(()),
        "importance_loss": reference.new_zeros(()),
    }


class DenseTwoStageOpticalPath(MoE4LanguageTwoBlockOpticalPath):
    """Two O/E/O stages without routing or spatial expert fan-out.

    Each stage reloads the 224x224 token amplitude at the centre of the same
    518 numerical canvas, applies one 224x224 phase mask, propagates 10 cm and
    reads the canonical 478x478 CCD.  This preserves the main graph's two
    feature boundaries while using exactly two active expert masks' worth of
    trainable phase per modality.
    """

    def __init__(
        self, width: int, settings: Any, *, max_tokens: int | None = None
    ) -> None:
        super().__init__(width, settings, max_tokens=max_tokens)
        # Keep one parameter-free compatibility slot because the shared
        # checkpoint schema records ``expert_stages_per_stack=1``.  The slot
        # is never executed: this class owns both dense phase stages below.
        self.core.expert_layers = nn.ModuleList([nn.Identity()])
        self.core.global_phase = nn.Identity()
        self.core.router = nn.Identity()
        self.phase1 = DensePhasePlane(settings.d2nn_phase_size, settings)
        self.phase2 = DensePhasePlane(settings.d2nn_phase_size, settings)

    def _center_field(self, input_fields: torch.Tensor) -> torch.Tensor:
        batch, height, width = input_fields.shape
        if (height, width) != (self.phase1.size, self.phase1.size):
            raise RuntimeError(
                f"D2NN input must be {self.phase1.size}x{self.phase1.size}"
            )
        canvas = input_fields.new_zeros(
            batch, self.core.geometry.canvas_size, self.core.geometry.canvas_size
        )
        y0 = (canvas.shape[-2] - height) // 2
        x0 = (canvas.shape[-1] - width) // 2
        canvas[:, y0 : y0 + height, x0 : x0 + width] = input_fields
        return torch.complex(canvas, torch.zeros_like(canvas))

    def _phase_modulation(
        self, field: torch.Tensor, phase: DensePhasePlane
    ) -> torch.Tensor:
        patch = phase.modulation(len(field))
        canvas = torch.ones_like(field, dtype=torch.complex64)
        y0 = (field.shape[-2] - phase.size) // 2
        x0 = (field.shape[-1] - phase.size) // 2
        canvas[:, y0 : y0 + phase.size, x0 : x0 + phase.size] = patch
        return canvas

    def _phase_support_mask(
        self, value: torch.Tensor, phase_support: str | None
    ) -> torch.Tensor:
        if phase_support in {"d2nn_phase1", "d2nn_phase2"}:
            mask = torch.zeros_like(value.real, dtype=torch.bool)
            size = self.phase1.size
            y0 = (value.shape[-2] - size) // 2
            x0 = (value.shape[-1] - size) // 2
            mask[:, y0 : y0 + size, x0 : x0 + size] = True
            return mask
        return super()._phase_support_mask(value, phase_support)

    def _run_encoded_stage(
        self,
        input_fields: torch.Tensor,
        lengths: list[int],
        padding_mask: torch.Tensor,
        phase: DensePhasePlane,
        stage: str,
        *,
        final: bool,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[int]]:
        field = self._center_field(input_fields)
        shifts = self._draw_stage_shifts("global" if final else "expert")
        input_y, input_x = shifts["input"]
        field = _translate_with_fill(field, input_y, input_x, fill_value=0.0)
        active = self.core.geometry.active_aperture
        amplitude = field[
            :, active.y0 : active.y1, active.x0 : active.x1
        ].abs().detach()
        if final:
            self.last_global_input_amplitude = amplitude
        else:
            self.last_expert_input_amplitude = amplitude
        raw_ccd = self._simulate_detector_roi(
            field,
            self._phase_modulation(field, phase),
            shifts,
            phase_support=stage,
        )
        operating = self._operating_loss(raw_ccd)
        if final:
            self.current_global_operating_loss = operating
            expert = self.current_expert_operating_loss
            self.current_operating_loss = (
                operating if expert is None else 0.5 * (expert + operating)
            )
            self.last_raw_ccd = raw_ccd.detach()
        else:
            self.current_expert_operating_loss = operating
            self.current_operating_loss = operating
            self.last_raw_expert_ccd = raw_ccd.detach()
        packed = self._readout_delta(
            self._perturb_ccd(raw_ccd), lengths, dtype, final=final
        )
        return self._scatter_delta(packed, padding_mask, dtype), lengths

    def run_expert_block(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[int]]:
        input_fields, lengths = self._encode_input_fields(latent, padding_mask)
        delta, lengths = self._run_encoded_stage(
            input_fields,
            lengths,
            padding_mask,
            self.phase1,
            "d2nn_phase1",
            final=False,
            dtype=latent.dtype,
        )
        routing = _dense_routing(latent)
        self.core.last_routing = routing
        return delta, routing, lengths

    def encode_global_input(
        self,
        fused_latent: torch.Tensor,
        padding_mask: torch.Tensor,
        routing: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del routing
        fields, _ = self._encode_input_fields(fused_latent, padding_mask)
        return fields

    def run_global_block(
        self,
        field: torch.Tensor,
        lengths: list[int],
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # ``field`` is the already normalized 224x224 amplitude returned by
        # ``encode_global_input``; do not silently re-encode or resize it.
        delta, _ = self._run_encoded_stage(
            field,
            lengths,
            padding_mask,
            self.phase2,
            "d2nn_phase2",
            final=True,
            dtype=dtype,
        )
        return delta

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase1.set_phase_dropout_active(active)
        self.phase2.set_phase_dropout_active(active)


class D2NNVisionReplacement(BalancedVisionReplacement):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__(hidden_size, settings)
        self.core.optical_branch = DenseTwoStageOpticalPath(
            self.core.width, settings, max_tokens=settings.max_visual_tokens
        )

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        phase = sum(
            parameter.numel()
            for plane in (
                self.core.optical_branch.phase1,
                self.core.optical_branch.phase2,
            )
            for parameter in plane.parameters()
        )
        report.update(
            {
                "implementation": "dense_d2nn_two_oeo_stages_no_router",
                "optical_phase_parameters": phase,
                "router_parameters": 0,
                "optical_parameters": sum(
                    parameter.numel()
                    for parameter in self.core.optical_branch.parameters()
                ),
            }
        )
        return report


class D2NNLanguageReplacement(BalancedLanguageReplacement):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__(hidden_size, settings)
        self.core.optical_branch = DenseTwoStageOpticalPath(
            self.core.width, settings, max_tokens=settings.max_language_tokens
        )

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        phase = sum(
            parameter.numel()
            for plane in (
                self.core.optical_branch.phase1,
                self.core.optical_branch.phase2,
            )
            for parameter in plane.parameters()
        )
        report.update(
            {
                "implementation": "dense_d2nn_two_oeo_stages_no_router",
                "optical_phase_parameters": phase,
                "router_parameters": 0,
                "optical_parameters": sum(
                    parameter.numel()
                    for parameter in self.core.optical_branch.parameters()
                ),
            }
        )
        return report


class D2NNMatchedReplacement(BalancedFusionReplacement):
    training_architecture_label = "lightgen_t01_d2nn_active_expert_matched"

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        super().__init__(*args, settings=settings, **kwargs)
        self.checkpoint_architecture = checkpoint_architecture(settings)

    def router_parameters(self) -> list[nn.Parameter]:
        return []

    def router_losses(self) -> dict[str, torch.Tensor]:
        zero = self.vision_surrogate.core.block1_optical_fusion_logit.new_zeros(())
        return {
            "vision_balance": zero,
            "vision_importance": zero,
            "language_balance": zero,
            "language_importance": zero,
        }

    def router_hard_load_balance_loss(self) -> dict[str, torch.Tensor]:
        zero = self.vision_surrogate.core.block1_optical_fusion_logit.new_zeros(())
        return {"vision": zero, "language": zero}

    def phase_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        vision = self.vision_surrogate.core.optical_branch
        language = self.language_surrogate.core.optical_branch
        return {
            "vision_expert": [vision.phase1.raw_phase],
            "vision_global": [vision.phase2.raw_phase],
            "language_expert": [language.phase1.raw_phase],
            "language_global": [language.phase2.raw_phase],
        }

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        per_plane = self.vision_surrogate.core.optical_branch.phase1.raw_phase.numel()
        optical_description = (
            "two dense 224x224 phase stages with CCD/electronic reload; "
            "no Router or experts"
        )
        for modality in ("vision", "language"):
            if isinstance(report.get(modality), dict):
                report[modality]["optical"] = optical_description
                report[modality]["router"] = "none"
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "initialization": (
                    "strict warmstart5 common electronics/readouts; fresh dense "
                    "D2NN phases; fusion gates reset"
                ),
                "router": {"backend": "none", "trainable_parameters": 0},
                "d2nn": {
                    "phase_layers_per_modality": 2,
                    "phase_shape": [224, 224],
                    "phase_parameters_per_layer": per_plane,
                    "phase_parameters_per_modality": 2 * per_plane,
                    "phase_parameters_total": 4 * per_plane,
                    "matching_target": "MoE Top-2 activated expert parameters",
                    "intermediate_boundary": "CCD normalization and electronic reload",
                },
                "physical_feature_stage_count": 4,
                "physical_capture_count": 4,
            }
        )
        return report

    def save_multiplane_phase_snapshot(
        self,
        output_dir: Path,
        *,
        epoch: int,
        train_loss: float,
        weight_variant: str,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        phases = {
            name: [2.0 * math.pi * torch.sigmoid(value.detach().cpu().float()) for value in values]
            for name, values in self.phase_parameter_groups().items()
        }
        destination = output_dir / "phase_parameters.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema_version": 1,
                "architecture": self.checkpoint_architecture,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "weight_variant": str(weight_variant),
                "physical_phase_rad": phases,
            },
            temporary,
        )
        temporary.replace(destination)
        report = {
            "schema_version": 1,
            "phase_parameters": str(destination),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "weight_variant": str(weight_variant),
            "shapes": {name: [list(item.shape) for item in values] for name, values in phases.items()},
        }
        (output_dir / "phase_parameters.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def save_multiplane_phase_preview(self, path: Path, *, title: str) -> None:
        import matplotlib.pyplot as plt

        phases = {
            name: 2.0 * math.pi * torch.sigmoid(values[0].detach().cpu().float())
            for name, values in self.phase_parameter_groups().items()
        }
        figure, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        image = None
        for axis, (name, phase) in zip(axes.ravel(), phases.items()):
            mean = torch.angle(torch.exp(1j * phase).mean())
            residual = torch.remainder(phase - mean + math.pi, 2.0 * math.pi) - math.pi
            image = axis.imshow(residual.numpy(), cmap="RdBu_r", vmin=-math.pi, vmax=math.pi)
            axis.set_title(f"{name}; std={float(residual.std(unbiased=False)):.3f} rad")
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="relative phase (rad)")
        figure.suptitle(title)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)


def build_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[BalancedFusionReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    if settings.lightgen_model_variant == "optical_router_scale_matched_moe":
        vision = BalancedVisionReplacement(settings.vision_hidden_size, settings)
        language = BalancedLanguageReplacement(settings.text_hidden_size, settings)
        _install_optical_router(vision, settings)
        _install_optical_router(language, settings)
        replacement: BalancedFusionReplacement = OpticalRouterScaleMatchedReplacement(
            loaded.model,
            vision.to(loaded.device),
            language.to(loaded.device),
            settings=settings,
        )
    elif settings.lightgen_model_variant == "d2nn_active_expert_matched":
        replacement = D2NNMatchedReplacement(
            loaded.model,
            D2NNVisionReplacement(settings.vision_hidden_size, settings).to(loaded.device),
            D2NNLanguageReplacement(settings.text_hidden_size, settings).to(loaded.device),
            settings=settings,
        )
    else:
        raise ValueError("Frozen Qwen baseline has no trainable student graph")
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


def _transplant_common_state(
    target: Mapping[str, torch.Tensor], source: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output = {key: value.detach().clone() for key, value in target.items()}
    fresh = {
        key
        for key in target
        if ".optical_branch.phase1.raw_phase" in key
        or ".optical_branch.phase2.raw_phase" in key
    }
    loaded: list[str] = []
    for key, value in target.items():
        if key in fresh:
            continue
        source_value = source.get(key)
        if source_value is None:
            raise RuntimeError(f"D2NN common warm-start tensor is missing: {key}")
        if tuple(source_value.shape) != tuple(value.shape):
            raise RuntimeError(f"D2NN common warm-start shape mismatch: {key}")
        output[key] = source_value.detach().clone()
        loaded.append(key)
    allowed_unused = (
        "core.optical_branch.core.expert_layers.",
        "core.optical_branch.core.router.",
        "core.optical_branch.core.global_phase.",
    )
    unexpected = sorted(
        key
        for key in set(source).difference(loaded)
        if not key.startswith(allowed_unused)
    )
    if unexpected:
        raise RuntimeError(
            "D2NN transplant left unexpected source tensors: "
            + ", ".join(unexpected[:8])
        )
    return output, {
        "loaded_common_tensor_count": len(loaded),
        "fresh_d2nn_phase_tensor_count": len(fresh),
        "discarded_moe_router_global_tensor_count": len(source) - len(loaded),
    }


def initialize_student(
    settings: Any,
    replacement: BalancedFusionReplacement,
    readout: ElectronicRetrievalReadout,
) -> dict[str, Any]:
    if isinstance(replacement, OpticalRouterScaleMatchedReplacement):
        report = load_warmstart5_initialization(settings, replacement, readout)
        replacement.reset_fusion_logits()
        report["scale_matched_fusion_logits_reset"] = True
        report["alpha_initial"] = settings.fusion_alpha_initial
        return report

    payload, digest = _load_payload(
        settings.router_source_checkpoint, settings.router_source_sha256
    )
    _validate_metadata(
        payload, settings, expected_architecture=STAGE_ARCHITECTURES["joint"]
    )
    reports = {}
    for name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        state, report = _transplant_common_state(
            surrogate.state_dict(), payload[f"{name}_optical"]
        )
        surrogate.load_state_dict(state, strict=True)
        reports[name] = report
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    replacement.reset_fusion_logits()
    replacement.configure_student_trainability()
    return {
        "mode": "warmstart5_common_electronics_fresh_d2nn_phases",
        "path": str(settings.router_source_checkpoint),
        "sha256": digest,
        "source_epoch": int(payload["epoch"]),
        "source_architecture": STAGE_ARCHITECTURES["joint"],
        "target_architecture": replacement.checkpoint_architecture,
        "scale_matched_fusion_logits_reset": True,
        "alpha_initial": settings.fusion_alpha_initial,
        "surrogates": reports,
    }


def parameter_fairness_contract(settings: Any) -> dict[str, Any]:
    per_expert = int(settings.expert_size) ** 2
    active_expert_per_modality = int(settings.top_k) * per_expert
    d2nn_per_modality = (
        int(settings.d2nn_phase_layers_per_modality)
        * int(settings.d2nn_phase_size) ** 2
    )
    return {
        "expert_shape": [settings.expert_size, settings.expert_size],
        "parameters_per_expert": per_expert,
        "moe_top_k": settings.top_k,
        "moe_active_expert_parameters_per_modality": active_expert_per_modality,
        "moe_active_expert_parameters_vision_language": 2
        * active_expert_per_modality,
        "d2nn_phase_layers_per_modality": settings.d2nn_phase_layers_per_modality,
        "d2nn_phase_parameters_per_modality": d2nn_per_modality,
        "d2nn_phase_parameters_vision_language": 2 * d2nn_per_modality,
        "active_expert_parameter_match_exact": (
            active_expert_per_modality == d2nn_per_modality
        ),
        "router_parameters_excluded_from_match": True,
        "moe_global_phase_parameters_excluded_from_match": True,
        "warning": (
            "The requested equality concerns activated expert masks only; total "
            "stored phase parameters and Router overhead are reported separately."
        ),
    }


__all__ = [
    "DensePhasePlane",
    "D2NNMatchedReplacement",
    "OpticalRouterScaleMatchedReplacement",
    "build_student",
    "initialize_student",
    "load_backbone",
    "parameter_fairness_contract",
]
