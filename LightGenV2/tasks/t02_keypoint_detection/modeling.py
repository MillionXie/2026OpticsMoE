"""LSP vision-only optical-Router MoE and phase-budget-matched D2NN."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from LightGenV2.tasks.t01_object_retrieval.modeling import DenseTwoStageOpticalPath
from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import (
    BalancedVisionCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    OpticalDetectorTopKRouter,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.modeling import (
    ROUTER_PREFIX,
    RobustDenseVision2Core,
    RobustVision2PoseStudent,
    _RobustCaptureBlock,
    _VisionBypass,
    load_vision_backbone,
    sha256_file,
    trainable_parameter_report,
)
from experiments.vision2_hybrid_dense.modeling import PoseHeatmapDecoder


def architecture_label(settings: Any) -> str:
    label = (
        f"lightgen_t02_{settings.lightgen_model_variant}_vision2_17um_10cm_"
        f"dc20_scale_matched_top2_v1"
    )
    lower, upper = settings.fusion_alpha_min, settings.fusion_alpha_max
    if (lower, upper) != (0.01, 0.95):
        label += f"_alpha{lower:.3f}_{upper:.3f}"
    return label


class LightGenDenseVision2Core(RobustDenseVision2Core):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        nn.Module.__init__(self)
        self.hybrid = BalancedVisionCore(
            hidden_size, settings.max_visual_tokens, settings
        )
        if settings.lightgen_model_variant == "d2nn_active_expert_matched":
            self.hybrid.optical_branch = DenseTwoStageOpticalPath(
                self.hybrid.width, settings, max_tokens=settings.max_visual_tokens
            )

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hybrid.optical_branch.__class__.__name__ == "DenseTwoStageOpticalPath":
            zero = self.hybrid.block1_optical_fusion_logit.new_zeros(())
            return zero, zero
        return super().router_losses()

    def phase_parameters(self) -> list[nn.Parameter]:
        branch = self.hybrid.optical_branch
        if isinstance(branch, DenseTwoStageOpticalPath):
            return [branch.phase1.raw_phase, branch.phase2.raw_phase]
        return [
            *branch.core.expert_layers.parameters(),
            *branch.core.global_phase.parameters(),
        ]


class LightGenVision2PoseStudent(RobustVision2PoseStudent):
    def __init__(self, loaded: Any, settings: Any) -> None:
        nn.Module.__init__(self)
        self.visual = loaded.visual
        self.original_blocks = list(self.visual.blocks)
        self.original_deepstack_indexes = tuple(
            int(value) for value in getattr(self.visual, "deepstack_visual_indexes", ())
        )
        self.core = LightGenDenseVision2Core(settings.vision_hidden_size, settings).to(
            loaded.device
        )
        self.capture_block = _RobustCaptureBlock(self.core)
        self.student_blocks = nn.ModuleList(
            [self.capture_block] + [_VisionBypass() for _ in self.original_blocks[1:]]
        )
        self.head = PoseHeatmapDecoder(
            input_dim=settings.electronic_width,
            heatmap_size=settings.heatmap_size,
            num_joints=14,
        ).to(loaded.device)
        self._active = False
        loaded.model.requires_grad_(False).eval()
        self.core.requires_grad_(True)
        self.core.hybrid.output_adapter.requires_grad_(False)
        self.head.requires_grad_(True)


def build_student(loaded: Any, settings: Any) -> LightGenVision2PoseStudent:
    settings.resolve_architecture(loaded.model)
    model = LightGenVision2PoseStudent(loaded, settings)
    if settings.lightgen_model_variant == "optical_router_scale_matched_moe":
        optical_core = model.core.optical_branch.core
        optical_core.router = OpticalDetectorTopKRouter(
            optical_core.geometry, settings
        ).to(loaded.device)
    model.router_backend = (
        "optical" if settings.lightgen_model_variant.startswith("optical_router") else "none"
    )
    model.checkpoint_architecture = architecture_label(settings)
    return model


def initialize_student(model: LightGenVision2PoseStudent, settings: Any) -> dict[str, Any]:
    path = settings.common_initialization_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mode = str(getattr(settings, "lightgen_initialization_mode", "shared_untrained"))
    if mode == "shared_untrained":
        if payload.get("type") != "untrained_lsp_vision2_body_and_pose_head_without_router":
            raise RuntimeError("LSP common initialization has the wrong type")
    elif mode == "compatible_trained_lsp":
        if not isinstance(payload.get("core"), dict) or not isinstance(payload.get("head"), dict):
            raise RuntimeError("Compatible trained LSP initialization requires core and head states")
    else:
        raise RuntimeError(f"Unsupported LSP initialization mode: {mode}")
    target = model.core.state_dict()
    source = payload["core_body"] if mode == "shared_untrained" else payload["core"]
    merged = {key: value.detach().clone() for key, value in target.items()}
    is_d2nn = settings.lightgen_model_variant == "d2nn_active_expert_matched"
    fresh = {
        key
        for key in target
        if is_d2nn and (
            ".optical_branch.phase1.raw_phase" in key
            or ".optical_branch.phase2.raw_phase" in key
        )
    }
    loaded_keys: list[str] = []
    for key, value in target.items():
        if key in fresh or key.startswith(ROUTER_PREFIX):
            continue
        source_value = source.get(key)
        if source_value is None:
            raise RuntimeError(f"Common LSP initialization tensor is missing: {key}")
        if tuple(source_value.shape) != tuple(value.shape):
            raise RuntimeError(f"Common LSP initialization shape mismatch: {key}")
        merged[key] = source_value.to(dtype=value.dtype)
        loaded_keys.append(key)
    model.core.load_state_dict(merged, strict=True)
    model.head.load_state_dict(payload["head"], strict=True)
    model.core.hybrid.reset_fusion_logits(settings.fusion_alpha_initial)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "mode": mode,
        "source_epoch": payload.get("epoch"),
        "loaded_common_tensors": len(loaded_keys),
        "fresh_phase_tensors": len(fresh),
        "fresh_router": not is_d2nn,
        "fusion_logits_reset": True,
    }


def optimizer(model: LightGenVision2PoseStudent, settings: Any) -> torch.optim.Optimizer:
    phase = model.core.phase_parameters()
    router = (
        list(model.core.router.parameters())
        if settings.lightgen_model_variant == "optical_router_scale_matched_moe"
        else []
    )
    head = list(model.head.parameters())
    phase_ids = {id(value) for value in phase}
    router_ids = {id(value) for value in router}
    head_ids = {id(value) for value in head}
    readout = [
        value
        for name, value in model.core.named_parameters()
        if value.requires_grad
        and ("readout" in name or "output_adapter" in name)
        and id(value) not in phase_ids | router_ids | head_ids
    ]
    readout_ids = {id(value) for value in readout}
    base = [
        value
        for value in model.parameters()
        if value.requires_grad
        and id(value) not in phase_ids | router_ids | head_ids | readout_ids
    ]
    groups = [
        {"params": base, "lr": settings.student_learning_rate, "name": "electronic"},
        {"params": phase, "lr": settings.phase_learning_rate, "name": "feature_phase"},
        {"params": readout, "lr": settings.dense_readout_learning_rate, "name": "ccd_readout"},
        {"params": head, "lr": settings.dense_head_learning_rate, "name": "pose_head"},
    ]
    if router:
        groups.insert(1, {"params": router, "lr": settings.router_learning_rate, "name": "router"})
    flat = [value for group in groups for value in group["params"]]
    expected = {id(value) for value in model.parameters() if value.requires_grad}
    if len(flat) != len({id(value) for value in flat}) or {id(value) for value in flat} != expected:
        raise RuntimeError("T02 optimizer groups do not exactly partition trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def architecture_report(model: LightGenVision2PoseStudent, settings: Any) -> dict[str, Any]:
    phase_parameters = sum(value.numel() for value in model.core.phase_parameters())
    is_d2nn = settings.lightgen_model_variant == "d2nn_active_expert_matched"
    return {
        "type": settings.lightgen_model_variant,
        "checkpoint_architecture": architecture_label(settings),
        "task": "LSP 14-keypoint heatmap regression",
        "qwen": {"frozen": True, "native_vision_blocks_executed": 0},
        "vision": {
            "hybrid_blocks": 2,
            "fusion": "scale-matched convex (1-alpha)E + alpha O",
            "alpha_range": [settings.fusion_alpha_min, settings.fusion_alpha_max],
            "latent_width": settings.electronic_width,
        },
        "router": {"backend": "none" if is_d2nn else "optical", "top_k": None if is_d2nn else 2},
        "optics": {
            "feature_captures": 2,
            "router_captures": 0 if is_d2nn else 1,
            "phase_parameters": phase_parameters,
            "active_expert_budget_target": 2 * settings.expert_size**2,
            "zero_order_intensity_range": [0.20, 0.30],
            "pixel_pitch_um": settings.pixel_pitch_um,
            "distance_m": settings.global_to_detector_distance_m,
        },
        "task_head": model.head.specification(),
    }


__all__ = [
    "architecture_label",
    "architecture_report",
    "build_student",
    "initialize_student",
    "load_vision_backbone",
    "optimizer",
    "sha256_file",
    "trainable_parameter_report",
]
