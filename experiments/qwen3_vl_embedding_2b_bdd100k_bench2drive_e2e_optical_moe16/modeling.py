from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import (
    LoadedVisionBackbone,
    load_vision_backbone,
    preprocess_vision,
    restore_packed_spatial,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
    lengths_from_cu,
)

from .io_utils import torch_load


class NativeVisionFeatureExtractor(nn.Module):
    """Frozen final Qwen Vision block output immediately before the merger."""

    def __init__(self, loaded: LoadedVisionBackbone) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.device = loaded.device
        self._captured: torch.Tensor | None = None
        if not hasattr(self.visual, "blocks") or not len(self.visual.blocks):
            raise RuntimeError("Qwen visual module has no native transformer blocks")
        self._hook = self.visual.blocks[-1].register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(value) or value.ndim != 2:
            raise RuntimeError("Qwen Vision final block must return packed [tokens,D]")
        self._captured = value

    @torch.no_grad()
    def extract(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        self.visual.eval()
        self._captured = None
        dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
        if self._captured is None:
            raise RuntimeError("Failed to capture native Qwen Vision hidden")
        lengths = _grid_lengths(image_grid_thw)
        if sum(lengths) != self._captured.shape[0]:
            raise RuntimeError(
                f"Teacher hidden tokens={self._captured.shape[0]}, grid tokens={sum(lengths)}"
            )
        return self._captured.detach(), lengths

    def close(self) -> None:
        self._hook.remove()


class CCDLinearRecombiner(nn.Module):
    """Physical nonnegative CCD rows -> signed 224D token features."""

    def __init__(self, dimension: int = 224) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.norm = nn.LayerNorm(self.dimension)
        self.linear = nn.Linear(self.dimension, self.dimension)

    def forward(self, ccd: torch.Tensor) -> torch.Tensor:
        if ccd.shape[-1] != self.dimension:
            raise RuntimeError(
                f"CCD recombiner expected feature width {self.dimension}, got {ccd.shape[-1]}"
            )
        result = self.linear(self.norm(ccd.float()))
        if not torch.isfinite(result).all():
            raise RuntimeError("CCD LayerNorm/Linear produced NaN or Inf")
        return result.to(ccd.dtype)


class _Bypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class _SingleStageCapture(nn.Module):
    def __init__(
        self,
        core: HomogeneousMoEOpticalCore,
        recombiner: CCDLinearRecombiner,
    ) -> None:
        super().__init__()
        self.core = core
        self.recombiner = recombiner
        self.token_counts: list[int] = []
        self.current_packed: torch.Tensor | None = None
        self.current_ccd: torch.Tensor | None = None
        self.last_numeric_stage: str = "not-run"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        if max(lengths) > self.core.max_tokens:
            raise RuntimeError(
                f"visual token count {max(lengths)} exceeds optical rows="
                f"{self.core.max_tokens}. Lower processor_max_pixels; no crop, "
                "truncate, pooling, or remapping is allowed."
            )
        if len(self.core.expert_layers) != 1:
            raise RuntimeError("Driving backbone requires exactly one expert phase plane")
        self.token_counts = lengths
        _require_finite_optical("packed Qwen patch hidden", hidden_states)
        _require_phase_parameters_finite(self.core)
        input_fields = self.core.encode_groups(list(hidden_states.split(lengths)))
        _require_finite_optical("nonnegative optical input field", input_fields)
        field, routing = self.core.begin(input_fields)
        _require_finite_optical("amplitude-SLM field", field)
        _require_finite_optical("router weights", routing["weights"])

        # Expand the single generic run_stage call into its mathematically
        # identical operations so an error identifies the first bad boundary.
        # A bounded physical phase does not sanitize a NaN raw phase or an
        # already non-finite input field.
        field = self.core.expert_layers[0](field)
        _require_finite_optical("expert phase output", field)
        field = self.core.propagator(field)
        _require_finite_optical("expert propagation output", field)
        if self.core.interlayer_enabled:
            field = self.core.interlayer_conversions[0](
                field,
                selected_experts=(
                    routing["selected_mask"]
                    if self.core.interlayer_hard_route_mask
                    else None
                ),
                routing_weights=(
                    routing["weights"]
                    if self.core.interlayer_reapply_routing_weights
                    else None
                ),
            )
            _require_finite_optical("OEO reload field", field)
        if self.core.capture_intermediate_fields:
            count = min(self.core.capture_sample_count, len(field))
            self.core.last_stage_fields.append(field[:count].detach().cpu())
        field = self.core.global_phase(field)
        _require_finite_optical("global phase output", field)
        field = self.core.propagator(field)
        _require_finite_optical("global-to-CCD propagation output", field)
        ccd = raw_ccd_readout(field, self.core)
        if torch.any(ccd < 0) or not torch.isfinite(ccd).all():
            raise RuntimeError("Physical CCD intensity must be finite and nonnegative")
        signed = self.recombiner(ccd)
        self.current_ccd = ccd
        self.current_packed = torch.cat(
            [signed[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        self.core.current_detector_readout = ccd
        if self.core.capture_intermediate_fields:
            count = min(self.core.capture_sample_count, ccd.shape[0])
            self.core.last_detector_readout = ccd[:count].detach().cpu()
        return hidden_states


def _require_finite_optical(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    detached = tensor.detach()
    magnitude = detached.abs().float() if detached.is_complex() else detached.float().abs()
    finite = magnitude[torch.isfinite(magnitude)]
    maximum = float(finite.max()) if finite.numel() else float("nan")
    mean = float(finite.mean()) if finite.numel() else float("nan")
    raise RuntimeError(
        f"Non-finite {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite_abs_mean={mean:.6g} finite_abs_max={maximum:.6g}. "
        "The 0..2pi phase constraint only bounds finite raw_phase values; it "
        "cannot repair NaN optimizer parameters or non-finite input/OEO fields."
    )


def _require_phase_parameters_finite(core: HomogeneousMoEOpticalCore) -> None:
    bad = [
        name
        for name, parameter in core.named_parameters()
        if "raw_phase" in name and not torch.isfinite(parameter).all()
    ]
    if bad:
        raise RuntimeError(
            "Non-finite raw phase parameters before optical propagation: "
            f"{bad[:8]}. sigmoid(NaN) remains NaN; restore the last finite "
            "step checkpoint and reset the optimizer state."
        )


def raw_ccd_readout(
    field: torch.Tensor, core: HomogeneousMoEOpticalCore
) -> torch.Tensor:
    """Square-law CCD crop and pooling, with no electronic activation.

    The only post-CCD transformation in the deployable backbone is the explicit
    LayerNorm(224) -> Linear(224,224) recombiner.
    """
    intensity = field.to(torch.complex64).abs().square().float()
    aperture = core.geometry.detector_aperture
    roi = intensity[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
    ccd = F.adaptive_avg_pool2d(
        roi.unsqueeze(1),
        (core.geometry.expert_size, core.geometry.expert_size),
    ).squeeze(1)
    return ccd


class OpticalDrivingBackbone(nn.Module):
    """Frozen Qwen patch/position stem and deployable single-stage Optical MoE16."""

    def __init__(self, loaded: LoadedVisionBackbone, settings: Any) -> None:
        super().__init__()
        if settings.vision_hidden_size is None:
            raise RuntimeError("Resolve Qwen architecture before building the student")
        loaded.model.requires_grad_(False).eval()
        self.settings = settings
        self.visual = loaded.visual
        self.device = loaded.device
        self.original_blocks = list(self.visual.blocks)
        self.core = HomogeneousMoEOpticalCore(
            settings.vision_hidden_size, settings.max_visual_tokens, settings
        ).to(self.device)
        # The generic core's 224->1024 output adapter is not part of this model.
        del self.core.output_adapter
        self.recombiner = CCDLinearRecombiner(settings.detector_output_size).to(
            self.device
        )
        self.capture = _SingleStageCapture(self.core, self.recombiner)
        self.student_blocks = nn.ModuleList(
            [self.capture] + [_Bypass() for _ in range(len(self.original_blocks) - 1)]
        )
        self._active = False

    def activate(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.visual.blocks[index] = block
        self._active = True

    def restore_native(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.visual.blocks[index] = block
        self._active = False

    def train(self, mode: bool = True) -> "OpticalDrivingBackbone":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.recombiner.train(mode)
        return self

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        if not self._active:
            self.activate()
        dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
        packed = self.capture.current_packed
        ccd = self.capture.current_ccd
        if packed is None or ccd is None:
            raise RuntimeError("Optical Vision path did not produce CCD features")
        return packed, list(self.capture.token_counts), ccd

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "core_state_dict": self.core.state_dict(),
            "recombiner_state_dict": self.recombiner.state_dict(),
            "architecture": self.specification(),
        }

    def load_checkpoint(self, path: Path, strict: bool = True) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Pretrained BDD Optical Backbone is missing: {path}. "
                "Run fit_pca and bdd_pretrain first."
            )
        payload = torch_load(path)
        state = payload.get("backbone", payload)
        self.core.load_state_dict(state["core_state_dict"], strict=strict)
        self.recombiner.load_state_dict(
            state["recombiner_state_dict"], strict=strict
        )
        return payload

    def specification(self) -> dict[str, Any]:
        return {
            "data_flow": (
                "Frozen Qwen patch/position embedding -> one Optical MoE16 "
                "expert phase -> OEO reload -> one global phase -> 10cm -> "
                "square-law CCD 224x224 -> LayerNorm(224) -> Linear(224,224)"
            ),
            "native_qwen_vision_blocks_executed": 0,
            "expert_phase_layers": 1,
            "global_phase_layers": 1,
            "num_experts": 16,
            "top_k": 4,
            "expert_size": 224,
            "active_size": 986,
            "canvas_size": 1026,
            "ccd_shape": [224, 224],
            "parameter_breakdown": optical_parameter_breakdown(self),
        }


class RoadStructureAuxiliaryHead(nn.Module):
    """Training-only lightweight drivable/lane/participant predictor."""

    def __init__(self, input_dim: int = 224, hidden_dim: int = 64) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, spatial: torch.Tensor, output_size: int = 224) -> torch.Tensor:
        tokens = spatial.permute(0, 2, 3, 1)
        tokens = self.projection(self.token_norm(tokens)).permute(0, 3, 1, 2)
        logits = self.decoder(tokens)
        return F.interpolate(
            logits,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )


class BDDPretrainModel(nn.Module):
    def __init__(self, backbone: OpticalDrivingBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.auxiliary_head = RoadStructureAuxiliaryHead()

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed, _, ccd = self.backbone(pixel_values, image_grid_thw)
        spatial = restore_packed_spatial(packed, image_grid_thw)
        auxiliary_logits = self.auxiliary_head(spatial)
        return packed, auxiliary_logits, ccd


class DrivingActor(nn.Module):
    """Small electronic actor conditioned on optical features and route state."""

    def __init__(
        self,
        visual_dim: int = 224,
        num_commands: int = 6,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        input_dim = visual_dim + 1 + num_commands + 2
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(current, width), nn.GELU()])
            current = width
        self.trunk = nn.Sequential(*layers)
        self.control_head = nn.Linear(current, 3)
        self.log_std = nn.Parameter(torch.full((3,), -2.0))

    def forward_normalized(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.control_head(self.trunk(state.float())))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return decode_normalized_action(self.forward_normalized(state))

    def sample(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.control_head(self.trunk(state.float()))
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw) - torch.log(
            1.0 - action.square() + 1e-6
        )
        return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mean)


class OpticalDrivingPolicy(nn.Module):
    def __init__(
        self,
        backbone: OpticalDrivingBackbone,
        actor: DrivingActor,
        settings: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.actor = actor
        self.settings = settings

    def encode(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        packed, lengths, _ = self.backbone(pixel_values, image_grid_thw)
        return pool_packed_tokens(packed, lengths)

    def state(
        self,
        visual: torch.Tensor,
        speed: torch.Tensor,
        command: torch.Tensor,
        target_point: torch.Tensor,
    ) -> torch.Tensor:
        from .datasets_bench2drive import normalized_driving_state

        conditioning = normalized_driving_state(
            speed,
            command,
            target_point,
            speed_scale=self.settings.speed_normalization_mps,
            target_clip=self.settings.target_point_clip_m,
            num_commands=self.settings.num_commands,
        ).to(visual.device)
        return torch.cat([visual.float(), conditioning], dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        speed: torch.Tensor,
        command: torch.Tensor,
        target_point: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.encode(pixel_values, image_grid_thw)
        return self.actor(self.state(visual, speed, command, target_point))


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int = 233, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = _critic(state_dim + 3, hidden_dim)
        self.q2 = _critic(state_dim + 3, hidden_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat([state.float(), action.float()], dim=-1)
        return self.q1(value), self.q2(value)


def _critic(input_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def build_backbone(
    loaded: LoadedVisionBackbone,
    settings: Any,
    checkpoint: Path | None = None,
) -> OpticalDrivingBackbone:
    backbone = OpticalDrivingBackbone(loaded, settings)
    if checkpoint is not None:
        backbone.load_checkpoint(checkpoint)
    backbone.activate()
    return backbone


def pool_packed_tokens(packed: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    if sum(lengths) != packed.shape[0]:
        raise RuntimeError("Packed optical features and token lengths differ")
    return torch.stack([group.mean(dim=0) for group in packed.split(lengths)])


def decode_normalized_action(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] != 3:
        raise ValueError("Normalized driving action must have [steer,throttle,brake]")
    return torch.stack(
        [action[..., 0], (action[..., 1] + 1.0) / 2.0, (action[..., 2] + 1.0) / 2.0],
        dim=-1,
    )


def encode_control_target(controls: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            controls[..., 0].clamp(-1, 1),
            controls[..., 1].clamp(0, 1) * 2.0 - 1.0,
            controls[..., 2].clamp(0, 1) * 2.0 - 1.0,
        ],
        dim=-1,
    )


def optical_parameter_breakdown(backbone: OpticalDrivingBackbone) -> dict[str, int]:
    core = backbone.core
    return {
        "input_adapter": sum(p.numel() for p in core.input_adapter.parameters()),
        "input_norm": sum(p.numel() for p in core.input_norm.parameters()),
        "router": sum(p.numel() for p in core.router.parameters()),
        "expert_phase": sum(
            p.numel() for layer in core.expert_layers for p in layer.parameters()
        ),
        "global_phase": sum(p.numel() for p in core.global_phase.parameters()),
        "oeo": sum(p.numel() for p in core.interlayer_conversions.parameters()),
        "ccd_layernorm_linear": sum(
            p.numel() for p in backbone.recombiner.parameters()
        ),
        "total_trainable": sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        ),
    }


def trainable_parameter_report(module: nn.Module) -> dict[str, Any]:
    rows = [
        {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()}
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "trainable_tensors": len(rows),
        "trainable_parameters": sum(row["parameters"] for row in rows),
        "parameters": rows,
    }


def _grid_lengths(image_grid_thw: torch.Tensor) -> list[int]:
    grids = image_grid_thw.detach().cpu().long().tolist()
    if any(t != 1 for t, _, _ in grids):
        raise RuntimeError(f"Only one-frame RGB observations are supported: {grids}")
    return [int(t * height * width) for t, height, width in grids]


__all__ = [
    "BDDPretrainModel",
    "CCDLinearRecombiner",
    "DrivingActor",
    "LoadedVisionBackbone",
    "NativeVisionFeatureExtractor",
    "OpticalDrivingBackbone",
    "OpticalDrivingPolicy",
    "RoadStructureAuxiliaryHead",
    "TwinQCritic",
    "build_backbone",
    "decode_normalized_action",
    "encode_control_target",
    "load_vision_backbone",
    "preprocess_vision",
    "raw_ccd_readout",
]
