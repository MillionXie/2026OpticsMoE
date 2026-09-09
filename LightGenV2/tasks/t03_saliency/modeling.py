"""SALICON Vision2 optical-Router MoE and active-budget-matched D2NN."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from LightGenV2.tasks.t01_object_retrieval.modeling import (
    DenseTwoStageOpticalPath,
    hard_topk_load_balance_loss,
)
from LightGenV2.tasks.t02_keypoint_detection.modeling import (
    LightGenDenseVision2Core,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    OpticalDetectorTopKRouter,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.modeling import (
    ROUTER_PREFIX,
    RobustVision2PoseStudent,
    _RobustCaptureBlock,
    _VisionBypass,
    load_vision_backbone,
    sha256_file,
    trainable_parameter_report,
)
from experiments.vision2_hybrid_dense.modeling import SaliencyDensityDecoder


def architecture_label(settings: Any) -> str:
    label = (
        f"lightgen_t03_{settings.lightgen_model_variant}_vision2_17um_10cm_"
        "dc20_scale_matched_top2_v1"
    )
    if (settings.fusion_alpha_min, settings.fusion_alpha_max) != (0.01, 0.95):
        label += f"_alpha{settings.fusion_alpha_min:g}_{settings.fusion_alpha_max:g}"
    label += "_mean_only" if settings.ccd_normalization == "mean_only" else ""
    kernel = getattr(settings, "electronic_spatial_kernel_size", 3)
    return label + (f"_ek{kernel}" if kernel != 3 else "")


def configure_spatial_kernel(hybrid: nn.Module, kernel: int) -> None:
    """T03-only receptive-field change; no shared backend mutation."""
    if kernel == 3:
        return
    if kernel != 5:
        raise ValueError("Only the audited 3-to-5 expansion is supported")
    for block in hybrid.blocks:
        old = block.token_depthwise
        if old.kernel_size != (3, 3) or old.groups != old.in_channels or old.bias is not None:
            raise ValueError("Unexpected spatial mixer; refusing kernel replacement")
        new = nn.Conv2d(old.in_channels, old.out_channels, kernel, padding=kernel//2,
                        groups=old.groups, bias=False, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :, 1:4, 1:4].copy_(old.weight)
        block.token_depthwise = new


def expand_spatial_checkpoint(source: dict, target: dict) -> dict:
    """Zero-pad only the two named depthwise kernels; reject all other drift."""
    if source.keys() != target.keys():
        raise RuntimeError("Kernel transfer cannot add/remove checkpoint keys")
    expanded = dict(source)
    expected = {f"hybrid.blocks.{i}.token_depthwise.weight" for i in range(2)}
    changed = set()
    for name, value in source.items():
        shape = target[name].shape
        if value.shape == shape:
            continue
        if name not in expected or value.shape[-2:] != (3, 3) or shape[-2:] != (5, 5) or value.shape[:-2] != shape[:-2]:
            raise RuntimeError(f"Unexpected warmstart shape mismatch: {name}")
        expanded[name] = torch.nn.functional.pad(value, (1, 1, 1, 1))
        changed.add(name)
    if changed != expected:
        raise RuntimeError("Expected exactly two 3-to-5 depthwise kernel transfers")
    return expanded


class MeanOnlyCCDNormalizer(nn.Module):
    """Task-local intensity normalization: no log, gamma, or upper clipping."""
    def __init__(self, active_size: int) -> None:
        super().__init__()
        self.active_size = active_size

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != (self.active_size, self.active_size):
            raise ValueError("CCD geometry mismatch")
        if not torch.isfinite(intensity).all():
            raise ValueError("Nonfinite CCD intensity")
        # The simulator already produces nonnegative measured intensity.
        if torch.any(intensity < 0):
            raise ValueError("Expected nonnegative measured intensity")
        value = intensity.float()
        return value / value.mean((-2, -1), keepdim=True).clamp_min(1e-6)


class LightGenVision2SaliencyStudent(RobustVision2PoseStudent):
    def __init__(self, loaded: Any, settings: Any) -> None:
        nn.Module.__init__(self)
        self.visual = loaded.visual
        self.original_blocks = list(self.visual.blocks)
        self.original_deepstack_indexes = tuple(
            int(value)
            for value in getattr(self.visual, "deepstack_visual_indexes", ())
        )
        self.core = LightGenDenseVision2Core(
            settings.vision_hidden_size, settings
        ).to(loaded.device)
        configure_spatial_kernel(self.core.hybrid, getattr(settings, "electronic_spatial_kernel_size", 3))
        self.capture_block = _RobustCaptureBlock(self.core)
        self.student_blocks = nn.ModuleList(
            [self.capture_block]
            + [_VisionBypass() for _ in self.original_blocks[1:]]
        )
        self.head = SaliencyDensityDecoder(
            input_dim=settings.electronic_width,
            output_size=settings.image_size,
        ).to(loaded.device)
        self._active = False
        loaded.model.requires_grad_(False).eval()
        self.core.requires_grad_(True)
        self.core.hybrid.output_adapter.requires_grad_(False)
        self.head.requires_grad_(True)
        self._router_soft_weight = float(settings.router_balance_weight)
        self._router_hard_weight = float(settings.router_hard_load_balance_weight)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        soft, importance = self.core.router_losses()
        if self.router_backend != "optical" or self._router_soft_weight <= 0.0:
            return soft, importance
        # The legacy SALICON objective has one balance scalar.  Preserve its
        # soft term while adding the hard Top-k load term at the independently
        # configured coefficient.
        hard = self.router_hard_load_balance_loss()
        return soft + (self._router_hard_weight / self._router_soft_weight) * hard, importance

    def router_hard_load_balance_loss(self) -> torch.Tensor:
        if getattr(self, "router_backend", "none") != "optical":
            return next(self.parameters()).new_zeros(())
        return hard_topk_load_balance_loss(
            self.core.last_routing,
            num_experts=4,
            top_k=2,
        )


def build_student(loaded: Any, settings: Any) -> LightGenVision2SaliencyStudent:
    settings.resolve_architecture(loaded.model)
    model = LightGenVision2SaliencyStudent(loaded, settings)
    is_d2nn = settings.lightgen_model_variant == "d2nn_active_expert_matched"
    if is_d2nn:
        model.core.hybrid.optical_branch = DenseTwoStageOpticalPath(
            model.core.hybrid.width,
            settings,
            max_tokens=settings.max_visual_tokens,
        ).to(loaded.device)
    else:
        optical_core = model.core.optical_branch.core
        optical_core.router = OpticalDetectorTopKRouter(
            optical_core.geometry, settings
        ).to(loaded.device)
    model.router_backend = "none" if is_d2nn else "optical"
    if settings.ccd_normalization == "mean_only":
        model.core.hybrid.optical_branch.ccd_normalizer = MeanOnlyCCDNormalizer(settings.active_size)
    model.checkpoint_architecture = architecture_label(settings)
    return model


def _reset_head(model: LightGenVision2SaliencyStudent, seed: int) -> None:
    devices = [model.head.classifier.weight.device.index] if model.head.classifier.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for module in model.head.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()


def initialize_student(
    model: LightGenVision2SaliencyStudent, settings: Any
) -> dict[str, Any]:
    warmstart = getattr(settings, "initialization_checkpoint", None)
    if warmstart is not None:
        expected_sha = getattr(settings, "initialization_checkpoint_sha256", None)
        if expected_sha and sha256_file(warmstart) != expected_sha:
            raise RuntimeError("T03 warmstart SHA256 mismatch")
        payload = torch.load(warmstart, map_location="cpu", weights_only=False)
        allowed = {model.checkpoint_architecture}
        expand_kernel = getattr(settings, "expand_kernel_on_warmstart", False)
        source_ek3 = model.checkpoint_architecture.removesuffix("_ek5")
        if expand_kernel:
            allowed.add(source_ek3)
        if settings.ccd_normalization == "mean_only":
            # Explicit parameter-compatible transfer, not exact continuation.
            allowed.add(model.checkpoint_architecture.removesuffix("_mean_only"))
        if settings.reset_fusion_on_warmstart:
            allowed.add(f"lightgen_t03_{settings.lightgen_model_variant}_vision2_17um_10cm_dc20_scale_matched_top2_v1_mean_only")
        if payload.get("architecture") not in allowed:
            raise RuntimeError("T03 warmstart architecture mismatch")
        transfer = expand_kernel and payload.get("architecture") == source_ek3
        core_state = expand_spatial_checkpoint(payload["core"], model.core.state_dict()) if transfer else payload["core"]
        model.core.load_state_dict(core_state, strict=True)
        model.head.load_state_dict(payload["saliency_head"], strict=True)
        if settings.reset_fusion_on_warmstart:
            model.core.hybrid.reset_fusion_logits(settings.fusion_alpha_initial)
        return {"path": str(warmstart), "sha256": sha256_file(warmstart),
                "mode": "warmstart_weights_with_new_optimizer", "source_epoch": payload["epoch"],
                "source_architecture": payload["architecture"], "target_architecture": model.checkpoint_architecture,
                "kernel_transfer": "3x3 centered in zero 5x5; initial convolution function preserved" if transfer else "none",
                "fusion_reset": settings.reset_fusion_on_warmstart,
                "fusion_alpha_initial": settings.fusion_alpha_initial}
    path = settings.common_initialization_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("type") != "untrained_lsp_vision2_body_and_pose_head_without_router":
        raise RuntimeError("T03 common Vision2 initialization has the wrong type")
    target = model.core.state_dict()
    source = payload["core_body"]
    merged = {key: value.detach().clone() for key, value in target.items()}
    is_d2nn = settings.lightgen_model_variant == "d2nn_active_expert_matched"
    fresh = {
        key
        for key in target
        if is_d2nn
        and (
            ".optical_branch.phase1.raw_phase" in key
            or ".optical_branch.phase2.raw_phase" in key
        )
    }
    loaded: list[str] = []
    for key, value in target.items():
        if key in fresh or key.startswith(ROUTER_PREFIX):
            continue
        source_value = source.get(key)
        if source_value is None:
            raise RuntimeError(f"T03 common initialization tensor missing: {key}")
        if tuple(source_value.shape) != tuple(value.shape):
            raise RuntimeError(f"T03 common initialization shape mismatch: {key}")
        merged[key] = source_value.to(dtype=value.dtype)
        loaded.append(key)
    model.core.load_state_dict(merged, strict=True)
    model.core.hybrid.reset_fusion_logits(settings.fusion_alpha_initial)
    _reset_head(model, int(settings.initialization_seed) + 300)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "loaded_common_core_tensors": len(loaded),
        "fresh_phase_tensors": len(fresh),
        "fresh_optical_router": not is_d2nn,
        "saliency_head_seed": int(settings.initialization_seed) + 300,
        "fusion_logits_reset": True,
    }


def optimizer(
    model: LightGenVision2SaliencyStudent, settings: Any
) -> torch.optim.Optimizer:
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
    electronic = [
        value
        for value in model.parameters()
        if value.requires_grad
        and id(value) not in phase_ids | router_ids | head_ids | readout_ids
    ]
    groups = [
        {"params": electronic, "lr": settings.student_learning_rate, "name": "electronic"},
        {"params": phase, "lr": settings.phase_learning_rate, "name": "feature_phase"},
        {"params": readout, "lr": settings.dense_readout_learning_rate, "name": "ccd_readout"},
        {"params": head, "lr": settings.dense_head_learning_rate, "name": "saliency_head"},
    ]
    if router:
        groups.insert(1, {"params": router, "lr": settings.router_learning_rate, "name": "optical_router"})
    groups = [group for group in groups if group["params"]]
    for group in groups:
        if group["name"] in {"feature_phase", "optical_router"}:
            group["weight_decay"] = settings.phase_weight_decay
    flat = [value for group in groups for value in group["params"]]
    expected = {id(value) for value in model.parameters() if value.requires_grad}
    if len(flat) != len({id(value) for value in flat}) or {id(value) for value in flat} != expected:
        raise RuntimeError("T03 optimizer groups do not partition trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def architecture_report(model: LightGenVision2SaliencyStudent, settings: Any) -> dict[str, Any]:
    is_d2nn = settings.lightgen_model_variant == "d2nn_active_expert_matched"
    phase_parameters = sum(value.numel() for value in model.core.phase_parameters())
    return {
        "type": settings.lightgen_model_variant,
        "checkpoint_architecture": architecture_label(settings),
        "task": "SALICON fixation-density prediction",
        "ccd_normalization": settings.ccd_normalization,
        "qwen": {"frozen": True, "native_vision_blocks_executed": 0},
        "vision": {
            "hybrid_blocks": 2,
            "fusion": "scale-matched convex (1-alpha)E + alpha O",
            "alpha_range": [settings.fusion_alpha_min, settings.fusion_alpha_max],
            "latent_width": settings.electronic_width,
            "spatial_depthwise_kernel": getattr(settings, "electronic_spatial_kernel_size", 3),
            "extra_electronic_parameters_vs_kernel3": 2 * settings.electronic_width * (getattr(settings, "electronic_spatial_kernel_size", 3)**2 - 9),
        },
        "router": {
            "backend": "none" if is_d2nn else "optical",
            "top_k": None if is_d2nn else 2,
        },
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
    "trainable_parameter_report",
]
