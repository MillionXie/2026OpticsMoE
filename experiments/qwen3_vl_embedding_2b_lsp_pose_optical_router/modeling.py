from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    FairElectronicAmplitudeRouter,
    OpticalDetectorTopKRouter,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    VisionTwoBlockOpticalCore as RobustVisionTwoBlockOpticalCore,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    lengths_from_cu,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import (
    load_vision_backbone,
    preprocess_vision,
    trainable_parameter_report,
)
from experiments.vision2_hybrid_dense.modeling import (
    PoseHeatmapDecoder,
    restore_qwen_block_major_spatial,
)


ROUTER_PREFIX = "hybrid.optical_branch.core.router."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_label(settings: Any) -> str:
    return (
        "lsp_vision2_moe4_17um_10cm_router_"
        f"{settings.router_backend}_k{settings.top_k}_"
        f"{settings.router_weight_normalization}_ste_v1"
    )


def _new_router(settings: Any, geometry: Any) -> torch.nn.Module:
    if settings.router_backend == "electronic":
        return FairElectronicAmplitudeRouter(geometry, settings)
    if settings.router_backend == "optical":
        return OpticalDetectorTopKRouter(geometry, settings)
    raise ValueError(f"Unsupported Router backend {settings.router_backend!r}")


class _VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class RobustDenseVision2Core(nn.Module):
    """Dense-task adapter around the exact latest robust Caltech Vision core."""

    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.hybrid = RobustVisionTwoBlockOpticalCore(
            hidden_size, settings.max_visual_tokens, settings
        )

    @property
    def optical_branch(self) -> nn.Module:
        return self.hybrid.optical_branch

    @property
    def router(self) -> nn.Module:
        return self.optical_branch.core.router

    @property
    def expert_layers(self) -> nn.Module:
        return self.optical_branch.core.expert_layers

    @property
    def global_phase(self) -> nn.Module:
        return self.optical_branch.core.global_phase

    @property
    def last_routing(self) -> dict[str, torch.Tensor]:
        return self.hybrid.last_routing

    @property
    def last_latent_groups(self) -> list[torch.Tensor]:
        return self.hybrid.last_latent_groups

    @property
    def last_input_fields(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_input_fields

    @property
    def last_amplitude_slm_canvas(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_amplitude_slm_canvas

    @property
    def last_stage_fields(self) -> list[torch.Tensor]:
        return self.optical_branch.core.last_stage_fields

    @property
    def last_detector_intensity(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_detector_intensity

    @property
    def last_detector_readout(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_detector_readout

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hybrid.forward_groups(
            groups, causal=False, spatial_shapes=spatial_shapes
        )

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.optical_branch.core.router_losses()

    def operating_loss(self) -> torch.Tensor:
        value = self.optical_branch.current_operating_loss
        return (
            self.hybrid.block2_optical_fusion_logit.new_zeros(())
            if value is None
            else value
        )

    def parameter_breakdown(self) -> dict[str, Any]:
        report = self.hybrid.parameter_breakdown()
        report["dense_output"] = "robust_second_fusion_latent_192_before_qwen_merger"
        report["source_core"] = (
            "qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust"
        )
        return report

    def set_phase_dropout_active(self, active: bool) -> None:
        self.optical_branch.set_phase_dropout_active(active)

    def set_intermediate_field_capture(
        self, enabled: bool, sample_count: int = 1
    ) -> None:
        self.optical_branch.core.set_intermediate_field_capture(enabled, sample_count)


class _RobustCaptureBlock(nn.Module):
    def __init__(self, core: RobustDenseVision2Core) -> None:
        super().__init__()
        self.core = core
        self.spatial_shapes: list[tuple[int, int, int]] | None = None

    def set_grid(self, image_grid_thw: torch.Tensor) -> None:
        shapes: list[tuple[int, int, int]] = []
        for frames, height, width in image_grid_thw.detach().cpu().long().tolist():
            if frames != 1:
                raise RuntimeError("LSP pose supports still images only")
            shapes.append((1, int(height), int(width)))
        self.spatial_shapes = shapes

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if self.spatial_shapes is None:
            raise RuntimeError("Set image_grid_thw before the Vision forward")
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        if len(lengths) != len(self.spatial_shapes):
            raise RuntimeError("Qwen sequence boundaries do not match the LSP batch")
        self.core.forward_groups(
            list(hidden_states.split(lengths)), self.spatial_shapes
        )
        return hidden_states


class RobustVision2PoseStudent(nn.Module):
    """Frozen Qwen stem, robust Caltech Vision2 body, and LSP pose decoder."""

    def __init__(self, loaded: Any, settings: Any) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.original_blocks = list(self.visual.blocks)
        self.original_deepstack_indexes = tuple(
            int(value)
            for value in getattr(self.visual, "deepstack_visual_indexes", ())
        )
        self.core = RobustDenseVision2Core(
            settings.vision_hidden_size, settings
        ).to(loaded.device)
        self.capture_block = _RobustCaptureBlock(self.core)
        self.student_blocks = nn.ModuleList(
            [self.capture_block]
            + [_VisionBypass() for _ in self.original_blocks[1:]]
        )
        self.head = PoseHeatmapDecoder(
            input_dim=settings.electronic_width,
            heatmap_size=settings.heatmap_size,
            num_joints=14,
        ).to(loaded.device)
        self._active = False
        loaded.model.requires_grad_(False).eval()
        self.core.requires_grad_(True)
        # Dense head reads the 192-D fused latent, not Qwen's ignored merger.
        self.core.hybrid.output_adapter.requires_grad_(False)
        self.head.requires_grad_(True)

    def activate(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.visual.blocks[index] = block
        if hasattr(self.visual, "deepstack_visual_indexes"):
            self.visual.deepstack_visual_indexes = []
        self._active = True

    def restore_native(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.visual.blocks[index] = block
        if hasattr(self.visual, "deepstack_visual_indexes"):
            self.visual.deepstack_visual_indexes = list(
                self.original_deepstack_indexes
            )
        self._active = False

    def train(self, mode: bool = True) -> "RobustVision2PoseStudent":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.head.train(mode)
        return self

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._active:
            self.activate()
        self.capture_block.set_grid(image_grid_thw)
        dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
        groups = self.core.last_latent_groups
        if len(groups) != len(image_grid_thw):
            raise RuntimeError("Robust Vision2 core did not retain one group per image")
        spatial = restore_qwen_block_major_spatial(
            torch.cat(groups, dim=0), image_grid_thw
        )
        detector = self.core.optical_branch.core.current_detector_readout
        if detector is None:
            raise RuntimeError("Vision global CCD readout is unavailable")
        return self.head(spatial), spatial, detector

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def operating_loss(self) -> torch.Tensor:
        return self.core.operating_loss()


def build_router_student(loaded: Any, settings: Any) -> torch.nn.Module:
    settings.resolve_architecture(loaded.model)
    student = RobustVision2PoseStudent(loaded, settings)
    optical_core = student.core.optical_branch.core
    optical_core.router = _new_router(settings, optical_core.geometry).to(loaded.device)
    student.router_backend = settings.router_backend
    student.checkpoint_architecture = architecture_label(settings)
    return student


def _body_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
        if not key.startswith(ROUTER_PREFIX)
    }


def materialize_common_initialization(
    model: torch.nn.Module,
    settings: Any,
    output: Path,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite common random initialization: {output}"
        )
    body = _body_state(model.core.state_dict())
    payload = {
        "schema_version": 1,
        "type": "untrained_lsp_vision2_body_and_pose_head_without_router",
        "initialization_seed": int(settings.initialization_seed),
        "router_state_included": False,
        "qwen_backbone_trainable": False,
        "core_body": body,
        "head": {
            key: value.detach().cpu().clone()
            for key, value in model.head.state_dict().items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "body_tensors": len(body),
        "head_tensors": len(payload["head"]),
        "router_state_included": False,
    }


def load_common_initialization(
    model: torch.nn.Module,
    settings: Any,
) -> dict[str, Any]:
    path = settings.common_initialization_checkpoint
    if not path.is_file():
        raise FileNotFoundError(
            f"Common untrained initialization is missing: {path}; run "
            "--phase materialize_initialization first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("type") != "untrained_lsp_vision2_body_and_pose_head_without_router":
        raise RuntimeError("Initialization checkpoint has the wrong type")
    if int(payload.get("initialization_seed", -1)) != int(settings.initialization_seed):
        raise RuntimeError("Initialization seed does not match the experiment contract")
    target = model.core.state_dict()
    target_body = {key for key in target if not key.startswith(ROUTER_PREFIX)}
    source_body = set(payload["core_body"])
    if target_body != source_body:
        raise RuntimeError(
            "Common initialization body is incompatible: "
            f"missing={sorted(target_body - source_body)} "
            f"unexpected={sorted(source_body - target_body)}"
        )
    merged = {key: value.detach().clone() for key, value in target.items()}
    for key in sorted(target_body):
        source = payload["core_body"][key]
        if tuple(source.shape) != tuple(target[key].shape):
            raise RuntimeError(f"Initialization tensor shape mismatch for {key}")
        merged[key] = source.to(dtype=target[key].dtype)
    model.core.load_state_dict(merged, strict=True)
    model.head.load_state_dict(payload["head"], strict=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "initialization_seed": int(settings.initialization_seed),
        "body_tensors_loaded": len(target_body),
        "router_tensors_loaded": 0,
        "router_backend": settings.router_backend,
        "top_k": int(settings.top_k),
    }


def student_architecture_report(model: torch.nn.Module, settings: Any) -> dict[str, Any]:
    router = model.core.optical_branch.core.router
    return {
        "type": "lsp_pose_vision_only_router_ablation",
        "checkpoint_architecture": architecture_label(settings),
        "qwen": {
            "frozen": True,
            "deepstack_disabled": True,
            "executed_native_vision_blocks": 0,
            "patch_embedding_trainable": False,
            "position_handling": (
                "image_grid_thw preserves 2D token topology for the custom mixer; "
                "Qwen native rotary kwargs are not consumed by the bypass block"
            ),
        },
        "vision": {
            "hybrid_blocks": 2,
            "block_1": "electronic mixer || MoE4 expert optical path, learned fusion",
            "block_2": "electronic mixer || global optical path, learned fusion",
            "latent_width": 192,
        },
        "router": {
            "backend": settings.router_backend,
            "top_k": int(settings.top_k),
            "weight_normalization": settings.router_weight_normalization,
            "straight_through": bool(settings.router_straight_through),
            "trainable_parameters": sum(p.numel() for p in router.parameters() if p.requires_grad),
            "vision_global_reuses_expert_route": True,
            "optical_extra_capture_count": 1 if settings.router_backend == "optical" else 0,
        },
        "optics": {
            "canvas": [518, 518],
            "canonical_ccd": [478, 478],
            "expert_layout": "2x2 MoE4",
            "expert_tile": [224, 224],
            "pixel_pitch_um": float(settings.pixel_pitch_um),
            "distance_m": float(settings.global_to_detector_distance_m),
        },
        "task_head": model.head.specification(),
        "physical_feature_capture_count": 2,
        "physical_total_capture_count": 3 if settings.router_backend == "optical" else 2,
    }


__all__ = [
    "ROUTER_PREFIX",
    "architecture_label",
    "build_router_student",
    "load_common_initialization",
    "load_vision_backbone",
    "materialize_common_initialization",
    "preprocess_vision",
    "sha256_file",
    "student_architecture_report",
    "trainable_parameter_report",
]
