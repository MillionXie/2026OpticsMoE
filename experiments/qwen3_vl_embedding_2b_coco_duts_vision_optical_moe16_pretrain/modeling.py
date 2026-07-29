from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import (
    LightweightSegmentationHead,
    LoadedVisionBackbone,
    load_vision_backbone,
    preprocess_vision,
    restore_detector_spatial,
    restore_packed_spatial,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
    lengths_from_cu,
)

from .io_utils import torch_load


class NativeVisionFeatureExtractor(nn.Module):
    """Frozen Qwen Vision final block output before the native merger."""

    def __init__(self, loaded: LoadedVisionBackbone) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.device = loaded.device
        self._captured: torch.Tensor | None = None
        if not hasattr(self.visual, "blocks") or not len(self.visual.blocks):
            raise RuntimeError("Qwen visual module has no native Transformer blocks")
        self._hook = self.visual.blocks[-1].register_forward_hook(self._capture)
        self.visual.requires_grad_(False).eval()

    def _capture(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(value) or value.ndim != 2:
            raise RuntimeError(
                "Final Qwen Vision block did not return packed [tokens,hidden]"
            )
        self._captured = value

    @torch.no_grad()
    def extract_packed(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        self.visual.eval()
        self._captured = None
        visual_dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(visual_dtype), grid_thw=image_grid_thw)
        if self._captured is None:
            raise RuntimeError("Failed to capture final Qwen Vision hidden")
        grids = image_grid_thw.detach().cpu().long().tolist()
        lengths = [int(temporal * height * width) for temporal, height, width in grids]
        if any(temporal != 1 for temporal, _, _ in grids):
            raise RuntimeError(
                f"Only still images are supported, got image_grid_thw={grids}"
            )
        if sum(lengths) != self._captured.shape[0]:
            raise RuntimeError(
                f"Teacher hidden has {self._captured.shape[0]} tokens but runtime "
                f"grid requires {sum(lengths)}"
            )
        return self._captured.detach(), lengths

    def close(self) -> None:
        self._hook.remove()


class CCDResidualRecombiner(nn.Module):
    """Shared 224D electronic restore layer retained by every downstream task.

    The base branch is the validated nonnegative CCD readout. The modulated
    branch can be signed:

        output = ccd + alpha * Linear(LayerNorm(ccd))
    """

    def __init__(
        self,
        dimension: int = 224,
        *,
        alpha_init: float = 0.1,
        alpha_trainable: bool = True,
        layernorm_affine: bool = True,
        layernorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.norm = nn.LayerNorm(
            self.dimension,
            eps=float(layernorm_eps),
            elementwise_affine=bool(layernorm_affine),
        )
        self.linear = nn.Linear(self.dimension, self.dimension)
        alpha = torch.tensor(float(alpha_init), dtype=torch.float32)
        if alpha_trainable:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    def forward(self, ccd_features: torch.Tensor) -> torch.Tensor:
        if ccd_features.shape[-1] != self.dimension:
            raise RuntimeError(
                f"CCD recombiner expects final dim {self.dimension}, got "
                f"{tuple(ccd_features.shape)}"
            )
        boundary_dtype = ccd_features.dtype
        base = ccd_features.float()
        delta = self.linear(self.norm(base))
        output = base + self.alpha.float() * delta
        if not torch.isfinite(output).all():
            raise RuntimeError("CCD residual recombiner produced NaN or Inf")
        return output.to(boundary_dtype)

    def specification(self) -> dict[str, Any]:
        return {
            "type": "ccd_residual_linear_recombiner",
            "formula": "Fout = Fccd + alpha * Linear(LayerNorm(Fccd))",
            "dimension": self.dimension,
            "alpha": float(self.alpha.detach()),
            "alpha_trainable": bool(self.alpha.requires_grad),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


class _VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class _ThreeStageOpticalCapture(nn.Module):
    """Replace native Vision blocks with one complete three-stage optical path."""

    def __init__(
        self,
        core: HomogeneousMoEOpticalCore,
        recombiner: CCDResidualRecombiner,
    ) -> None:
        super().__init__()
        self.core = core
        self.recombiner = recombiner
        self.token_counts: list[int] = []
        self.current_packed_features: torch.Tensor | None = None
        self.current_ccd_features: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        if max(lengths) > self.core.max_tokens:
            raise RuntimeError(
                f"visual token count {max(lengths)} exceeds optical field rows="
                f"{self.core.max_tokens}; lower processor_max_pixels. No crop, "
                "truncate, pooling, or token remapping is allowed."
            )
        self.token_counts = lengths
        input_fields = self.core.encode_groups(list(hidden_states.split(lengths)))
        field, routing = self.core.begin(input_fields)
        for stage_index in range(len(self.core.expert_layers)):
            field = self.core.run_stage(stage_index, field, routing)
        # The third OEO reload and global phase are co-planar. Only the global
        # phase-to-CCD 10 cm propagation remains after global modulation.
        field = self.core.propagator(self.core.global_phase(field))
        ccd, detector_intensity = self.core.readout(field)
        if not torch.isfinite(ccd).all() or torch.any(ccd < 0):
            raise RuntimeError("Physical CCD readout must be finite and nonnegative")
        recombined = self.recombiner(ccd)
        self.current_ccd_features = ccd
        self.current_packed_features = torch.cat(
            [
                recombined[sample_index, :length]
                for sample_index, length in enumerate(lengths)
            ],
            dim=0,
        )
        self.core.current_detector_readout = ccd
        if self.core.capture_intermediate_fields:
            count = min(self.core.capture_sample_count, len(field))
            self.core.last_detector_intensity = (
                detector_intensity[:count].detach().cpu()
            )
            self.core.last_detector_readout = ccd[:count].detach().cpu()
        # The native merger result is intentionally ignored by this experiment,
        # but returning the unmodified hidden preserves Qwen's visual call ABI.
        return hidden_states


class OpticalVisionBackbone(nn.Module):
    """Frozen Qwen patch/position stem plus trainable MoE16 and CCD recombiner."""

    def __init__(self, loaded: LoadedVisionBackbone, settings: Any) -> None:
        super().__init__()
        if settings.vision_hidden_size is None:
            raise RuntimeError("Qwen architecture must be resolved before student build")
        loaded.model.requires_grad_(False).eval()
        self.settings = settings
        self.visual = loaded.visual
        self.device = loaded.device
        self.original_blocks = list(self.visual.blocks)
        self.core = HomogeneousMoEOpticalCore(
            settings.vision_hidden_size,
            settings.max_visual_tokens,
            settings,
        ).to(self.device)
        # The generic multimodal core contains a 224->1024 hidden adapter.
        # This experiment never returns to 1024D: remove it from parameters,
        # checkpoints, and reports rather than merely freezing an unused layer.
        del self.core.output_adapter
        self.recombiner = CCDResidualRecombiner(
            settings.detector_output_size,
            alpha_init=settings.recombiner_alpha_init,
            alpha_trainable=settings.recombiner_alpha_trainable,
            layernorm_affine=settings.recombiner_layernorm_affine,
            layernorm_eps=settings.recombiner_layernorm_eps,
        ).to(self.device)
        self.capture_block = _ThreeStageOpticalCapture(
            self.core,
            self.recombiner,
        )
        self.student_blocks = nn.ModuleList(
            [self.capture_block]
            + [_VisionBypass() for _ in range(len(self.original_blocks) - 1)]
        )
        self._active = False
        self._native_released_to_cpu = False

    def activate(self, *, release_native_to_cpu: bool = False) -> None:
        for index, block in enumerate(self.student_blocks):
            self.visual.blocks[index] = block
        self._active = True
        if release_native_to_cpu and not self._native_released_to_cpu:
            for block in self.original_blocks:
                block.to("cpu")
            self._native_released_to_cpu = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def restore_native(self) -> None:
        if self._native_released_to_cpu:
            for block in self.original_blocks:
                block.to(self.device)
            self._native_released_to_cpu = False
        for index, block in enumerate(self.original_blocks):
            self.visual.blocks[index] = block
        self._active = False

    def train(self, mode: bool = True) -> "OpticalVisionBackbone":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.recombiner.train(mode)
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        if not self._active:
            self.activate()
        visual_dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(visual_dtype), grid_thw=image_grid_thw)
        packed = self.capture_block.current_packed_features
        ccd = self.capture_block.current_ccd_features
        if packed is None or ccd is None:
            raise RuntimeError("Optical Vision path did not produce CCD features")
        expected = sum(self.capture_block.token_counts)
        if packed.shape != (expected, self.settings.detector_output_size):
            raise RuntimeError(
                f"Packed optical feature shape {tuple(packed.shape)} != "
                f"({expected},{self.settings.detector_output_size})"
            )
        return packed, list(self.capture_block.token_counts), ccd

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "core_state_dict": self.core.state_dict(),
            "recombiner_state_dict": self.recombiner.state_dict(),
            "architecture": self.specification(),
        }

    def load_checkpoint(self, path: Path, *, strict: bool = True) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Optical backbone checkpoint is missing: {path}")
        payload = torch_load(path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid optical checkpoint payload: {path}")
        if "backbone" in payload:
            state = payload["backbone"]
        else:
            state = payload
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint has no backbone state: {path}")
        self.core.load_state_dict(state["core_state_dict"], strict=strict)
        self.recombiner.load_state_dict(
            state["recombiner_state_dict"],
            strict=strict,
        )
        return payload

    def specification(self) -> dict[str, Any]:
        return {
            "qwen_native_vision_blocks_executed": 0,
            "qwen_patch_position_stem_frozen": True,
            "expert_stages": len(self.core.expert_layers),
            "experts_per_stage": self.settings.num_experts,
            "top_k": self.settings.top_k,
            "expert_size": self.settings.expert_size,
            "active_size": self.settings.active_size,
            "canvas_size": self.settings.canvas_size,
            "ccd_shape": [self.settings.detector_output_size] * 2,
            "hidden_restore_224_to_1024_present": False,
            "pca_present_in_student": False,
            "recombiner": self.recombiner.specification(),
            "parameter_breakdown": optical_parameter_breakdown(self),
        }


class DUTSSaliencyModel(nn.Module):
    def __init__(
        self,
        backbone: OpticalVisionBackbone,
        settings: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.settings = settings
        self.head = LightweightSegmentationHead(
            input_dim=settings.detector_output_size,
            projection_dim=settings.segmentation_projection_dim,
            decoder_channels=settings.segmentation_channels,
            groupnorm_groups=settings.segmentation_groupnorm_groups,
            output_size=settings.image_size,
            refinement_enabled=False,
            progressive_refinement_enabled=False,
            detector_residual_enabled=False,
        ).to(backbone.device)

    def train(self, mode: bool = True) -> "DUTSSaliencyModel":
        super().train(mode)
        self.backbone.train(mode)
        self.head.train(mode)
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        detach_backbone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_backbone:
            with torch.no_grad():
                packed, _, ccd = self.backbone(pixel_values, image_grid_thw)
            packed = packed.detach()
        else:
            packed, _, ccd = self.backbone(pixel_values, image_grid_thw)
        spatial = restore_packed_spatial(packed, image_grid_thw)
        logits = self.head(spatial)
        return logits, spatial, ccd

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone.router_losses()


def build_optical_backbone(
    loaded: LoadedVisionBackbone,
    settings: Any,
    *,
    release_native_to_cpu: bool = True,
) -> OpticalVisionBackbone:
    backbone = OpticalVisionBackbone(loaded, settings)
    backbone.activate(release_native_to_cpu=release_native_to_cpu)
    return backbone


def build_duts_model(
    loaded: LoadedVisionBackbone,
    settings: Any,
    *,
    checkpoint: Path | None = None,
) -> DUTSSaliencyModel:
    backbone = build_optical_backbone(loaded, settings)
    if checkpoint is not None:
        backbone.load_checkpoint(checkpoint)
    return DUTSSaliencyModel(backbone, settings)


def optical_parameter_breakdown(backbone: OpticalVisionBackbone) -> dict[str, Any]:
    core = backbone.core
    expert_phase = sum(
        parameter.numel()
        for layer in core.expert_layers
        for parameter in layer.parameters()
    )
    global_phase = sum(
        parameter.numel() for parameter in core.global_phase.parameters()
    )
    router = sum(parameter.numel() for parameter in core.router.parameters())
    input_adapter = sum(
        parameter.numel() for parameter in core.input_adapter.parameters()
    )
    input_norm = sum(
        parameter.numel() for parameter in core.input_norm.parameters()
    )
    oeo = sum(
        parameter.numel()
        for module in core.interlayer_conversions
        for parameter in module.parameters()
    )
    recombiner = sum(
        parameter.numel() for parameter in backbone.recombiner.parameters()
    )
    total = sum(parameter.numel() for parameter in backbone.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )
    return {
        "input_adapter_parameters": input_adapter,
        "input_adapter_norm_parameters": input_norm,
        "router_parameters": router,
        "expert_phase_parameters": expert_phase,
        "global_phase_parameters": global_phase,
        "optical_phase_parameters": expert_phase + global_phase,
        "oeo_affine_parameters": oeo,
        "ccd_recombiner_parameters": recombiner,
        "removed_hidden_restore_224_to_1024_parameters": 0,
        "backbone_parameters_excluding_frozen_qwen": (
            input_adapter
            + input_norm
            + router
            + expert_phase
            + global_phase
            + oeo
            + recombiner
        ),
        "module_total_parameters_including_frozen_qwen_stem": total,
        "trainable_parameters": trainable,
    }


def trainable_parameter_report(
    module: nn.Module,
    *,
    prefix: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            rows.append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": list(parameter.shape),
                    "parameters": parameter.numel(),
                }
            )
            seen.add(id(parameter))
    return {
        "trainable_parameters": sum(row["parameters"] for row in rows),
        "trainable_tensors": len(rows),
        "trainable_parameter_list": rows,
    }
