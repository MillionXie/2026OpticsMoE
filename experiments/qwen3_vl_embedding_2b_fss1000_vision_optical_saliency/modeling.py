from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    resolve_cached_model_source,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
    lengths_from_cu,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.replacement import (
    locate_visual,
)


@dataclass(frozen=True)
class LoadedVisionBackbone:
    model: nn.Module
    visual: nn.Module
    processor: Any
    device: torch.device
    load_time_sec: float


def load_vision_backbone(settings: Any, device: torch.device) -> LoadedVisionBackbone:
    """Load the official checkpoint but place only its frozen Vision module on GPU."""
    transformers = importlib.import_module("transformers")
    source = resolve_cached_model_source(settings.model_id, settings.cache_dir)
    using_snapshot = source != settings.model_id
    common = {
        "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
        "local_files_only": settings.local_files_only or using_snapshot,
        "trust_remote_code": True,
    }
    common = {key: value for key, value in common.items() if value is not None}
    processor = transformers.AutoProcessor.from_pretrained(
        source,
        min_pixels=settings.processor_min_pixels,
        max_pixels=settings.processor_max_pixels,
        **common,
    )
    model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        raise RuntimeError(
            "Installed transformers does not expose a Qwen3-VL model class."
        )
    started = time.perf_counter()
    model = model_class.from_pretrained(
        source,
        dtype=_dtype(settings.dtype),
        low_cpu_mem_usage=True,
        attn_implementation=settings.attn_implementation,
        **common,
    )
    model.requires_grad_(False).eval()
    visual = locate_visual(model)
    visual.to(device).requires_grad_(False).eval()
    settings.resolve_architecture(model)
    return LoadedVisionBackbone(
        model=model,
        visual=visual,
        processor=processor,
        device=device,
        load_time_sec=time.perf_counter() - started,
    )


def preprocess_vision(
    processor: Any,
    images: list[Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Use Qwen's native image processor without constructing a language prompt."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Qwen processor has no image_processor")
    values = image_processor(images=images, return_tensors="pt")
    required = ("pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(
            f"Qwen image processor omitted {missing}; returned {list(values.keys())}"
        )
    return {
        name: values[name].to(device, non_blocking=True)
        for name in required
    }


def restore_packed_spatial(
    packed: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Strictly restore packed Qwen tokens to [B,C,H,W] using runtime grid metadata."""
    if packed.ndim != 2:
        raise RuntimeError(f"Packed spatial features must be [sum(T),C], got {tuple(packed.shape)}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise RuntimeError(
            f"image_grid_thw must be [B,3], got {tuple(image_grid_thw.shape)}"
        )
    grids = image_grid_thw.detach().cpu().long().tolist()
    lengths = [int(t * h * w) for t, h, w in grids]
    if any(t != 1 for t, _, _ in grids):
        raise RuntimeError(
            f"Saliency expects one temporal frame per image, got grids={grids}; "
            "no temporal pooling or silent reshape is allowed."
        )
    if sum(lengths) != packed.shape[0]:
        raise RuntimeError(
            f"Packed token count {packed.shape[0]} does not match image_grid_thw "
            f"total {sum(lengths)} ({grids})"
        )
    spatial_shapes = {(int(h), int(w)) for _, h, w in grids}
    if len(spatial_shapes) != 1:
        raise RuntimeError(
            f"A batch contains different Qwen spatial grids {sorted(spatial_shapes)}. "
            "Use fixed 224x224 preprocessing; no padding/crop fallback is allowed."
        )
    rows = []
    offset = 0
    for (_, h, w), length in zip(grids, lengths):
        group = packed[offset:offset + length]
        if group.shape[0] != h * w:
            raise RuntimeError("Token/grid mismatch while restoring teacher feature map")
        rows.append(group.reshape(h, w, packed.shape[-1]).permute(2, 0, 1))
        offset += length
    return torch.stack(rows, dim=0)


def restore_detector_spatial(
    detector_readout: torch.Tensor,
    image_grid_thw: torch.Tensor,
    token_counts: list[int],
) -> torch.Tensor:
    """Map valid CCD token rows back to the processor-provided 2-D token grid."""
    if detector_readout.ndim != 3:
        raise RuntimeError(
            f"Vision CCD readout must be [B,rows,features], got {tuple(detector_readout.shape)}"
        )
    if detector_readout.shape[0] != len(token_counts):
        raise RuntimeError("CCD batch size does not match captured visual token counts")
    grids = image_grid_thw.detach().cpu().long().tolist()
    if len(grids) != len(token_counts):
        raise RuntimeError("image_grid_thw batch does not match CCD batch")
    packed = []
    for sample_index, ((temporal, height, width), count) in enumerate(zip(grids, token_counts)):
        expected = int(temporal * height * width)
        if temporal != 1:
            raise RuntimeError(
                f"Sample {sample_index} has temporal grid={temporal}; no temporal pooling is allowed"
            )
        if count != expected:
            raise RuntimeError(
                f"Sample {sample_index}: optical token count={count}, but "
                f"image_grid_thw={grids[sample_index]} requires {expected}"
            )
        if count > detector_readout.shape[1]:
            raise RuntimeError(
                f"Sample {sample_index} needs {count} CCD rows, only "
                f"{detector_readout.shape[1]} are available"
            )
        packed.append(detector_readout[sample_index, :count])
    return restore_packed_spatial(torch.cat(packed, dim=0), image_grid_thw)


class LightweightSegmentationHead(nn.Module):
    """Token projection plus a small convolutional decoder to 224x224 logits."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        decoder_channels: tuple[int, ...],
        groupnorm_groups: int,
        output_size: int = 224,
        refinement_enabled: bool = False,
        progressive_refinement_enabled: bool = False,
        detector_residual_enabled: bool = False,
        detector_identity_scale_init: float = 1.0,
        detector_input_scale_init: float = 0.1,
        detector_identity_scale_trainable: bool = False,
        detector_input_scale_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_size = int(output_size)
        self.token_norm = nn.LayerNorm(self.input_dim)
        self.token_projection = nn.Linear(self.input_dim, projection_dim)
        layers: list[nn.Module] = []
        current = int(projection_dim)
        for channels in decoder_channels:
            groups = min(int(groupnorm_groups), int(channels))
            while channels % groups:
                groups -= 1
            layers.extend(
                [
                    nn.Conv2d(current, int(channels), 3, padding=1, bias=False),
                    nn.GroupNorm(groups, int(channels)),
                    nn.GELU(),
                ]
            )
            current = int(channels)
        self.decoder = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(current, 1, 1)
        self.refinement_enabled = bool(refinement_enabled)
        self.progressive_refinement_enabled = bool(progressive_refinement_enabled)
        self.detector_residual_enabled = bool(detector_residual_enabled)
        if self.detector_residual_enabled:
            self._make_scale(
                "detector_identity_scale",
                detector_identity_scale_init,
                detector_identity_scale_trainable,
            )
            self._make_scale(
                "detector_input_scale",
                detector_input_scale_init,
                detector_input_scale_trainable,
            )
        else:
            self.detector_identity_scale = None
            self.detector_input_scale = None
        if self.refinement_enabled and self.progressive_refinement_enabled:
            raise ValueError("Local and progressive refinement cannot both be enabled")
        if self.refinement_enabled:
            groups = min(int(groupnorm_groups), int(current))
            while current % groups:
                groups -= 1
            self.refinement = nn.Sequential(
                nn.Conv2d(current, current, 3, padding=1, bias=False),
                nn.GroupNorm(groups, current),
                nn.GELU(),
                nn.Conv2d(current, 1, 1),
            )
            # Exact checkpoint-compatible initialization: before the first
            # update, enabling refinement changes no logits.
            nn.init.zeros_(self.refinement[-1].weight)
            nn.init.zeros_(self.refinement[-1].bias)
        else:
            self.refinement = None
        if self.progressive_refinement_enabled:
            progressive_channels = (64, 32, 16, 8)
            progressive_layers: list[nn.Module] = []
            progressive_current = int(projection_dim)
            for channels in progressive_channels:
                groups = min(int(groupnorm_groups), int(channels))
                while channels % groups:
                    groups -= 1
                progressive_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            progressive_current,
                            int(channels),
                            3,
                            padding=1,
                            bias=False,
                        ),
                        nn.GroupNorm(groups, int(channels)),
                        nn.GELU(),
                    )
                )
                progressive_current = int(channels)
            self.progressive_refinement = nn.ModuleList(progressive_layers)
            self.progressive_classifier = nn.Conv2d(progressive_current, 1, 1)
            # Preserve the source checkpoint's exact logits until training
            # begins while allowing the multi-resolution branch to learn.
            nn.init.zeros_(self.progressive_classifier.weight)
            nn.init.zeros_(self.progressive_classifier.bias)
        else:
            self.progressive_refinement = None
            self.progressive_classifier = None

    def _make_scale(self, name: str, value: float, trainable: bool) -> None:
        tensor = torch.tensor(float(value), dtype=torch.float32)
        if trainable:
            setattr(self, name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    def forward(
        self,
        spatial_features: torch.Tensor,
        input_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spatial_features.ndim != 4 or spatial_features.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Segmentation head expects [B,{self.input_dim},H,W], got "
                f"{tuple(spatial_features.shape)}"
            )
        if self.detector_residual_enabled:
            if input_residual is None or input_residual.shape != spatial_features.shape:
                raise RuntimeError(
                    "Detector residual requires a shape-compatible optical input "
                    f"feature; detector={tuple(spatial_features.shape)}, "
                    f"input={None if input_residual is None else tuple(input_residual.shape)}"
                )
            boundary_dtype = spatial_features.dtype
            spatial_features = (
                self.detector_identity_scale.float() * spatial_features.float()
                + self.detector_input_scale.float() * input_residual.float()
            ).to(boundary_dtype)
        tokens = spatial_features.permute(0, 2, 3, 1)
        projected = self.token_projection(self.token_norm(tokens.float()))
        decoded = self.decoder(projected.permute(0, 3, 1, 2))
        decoded = F.interpolate(
            decoded,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.classifier(decoded)
        if self.refinement is not None:
            logits = logits + self.refinement(decoded)
        if self.progressive_refinement is not None:
            progressive = projected.permute(0, 3, 1, 2)
            for block in self.progressive_refinement:
                progressive = F.interpolate(
                    progressive,
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=False,
                )
                progressive = block(progressive)
            if progressive.shape[-2:] != (self.output_size, self.output_size):
                progressive = F.interpolate(
                    progressive,
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            logits = logits + self.progressive_classifier(progressive)
        expected = (spatial_features.shape[0], 1, self.output_size, self.output_size)
        if logits.shape != expected:
            raise RuntimeError(f"Mask logits shape {tuple(logits.shape)} != {expected}")
        return logits

    def specification(self) -> dict[str, Any]:
        return {
            "type": "lightweight_spatial_segmentation_head",
            "input_dim": self.input_dim,
            "output_size": self.output_size,
            "refinement_enabled": self.refinement_enabled,
            "progressive_refinement_enabled": self.progressive_refinement_enabled,
            "detector_residual_enabled": self.detector_residual_enabled,
            "detector_identity_scale": (
                None
                if self.detector_identity_scale is None
                else float(self.detector_identity_scale.detach())
            ),
            "detector_input_scale": (
                None
                if self.detector_input_scale is None
                else float(self.detector_input_scale.detach())
            ),
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class FrozenQwenVisionTeacher(nn.Module):
    """Run all native frozen Vision blocks and expose the last pre-merger spatial hidden."""

    def __init__(self, loaded: LoadedVisionBackbone, head: LightweightSegmentationHead) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.head = head
        self._captured: torch.Tensor | None = None
        if not hasattr(self.visual, "blocks") or not len(self.visual.blocks):
            raise RuntimeError("Qwen visual module has no native blocks")
        self._hook = self.visual.blocks[-1].register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(value) or value.ndim != 2:
            raise RuntimeError("Final Qwen Vision block did not return packed [tokens,hidden]")
        self._captured = value

    def train(self, mode: bool = True) -> "FrozenQwenVisionTeacher":
        super().train(mode)
        self.visual.eval()
        self.head.train(mode)
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual.eval()
        with torch.no_grad():
            self._captured = None
            visual_dtype = next(self.visual.patch_embed.parameters()).dtype
            self.visual(pixel_values.to(visual_dtype), grid_thw=image_grid_thw)
            if self._captured is None:
                raise RuntimeError("Failed to capture the final native Qwen Vision hidden")
            packed = self._captured.detach()
        spatial = restore_packed_spatial(packed, image_grid_thw)
        return self.head(spatial), spatial

    def close(self) -> None:
        self._hook.remove()


class _VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class _OpticalCaptureBlock(nn.Module):
    """Consume patch/position hidden in MoE16 and keep native block output bypassed."""

    def __init__(self, core: HomogeneousMoEOpticalCore) -> None:
        super().__init__()
        self.core = core
        self.token_counts: list[int] = []
        self.current_input_fields: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        self.token_counts = lengths
        fields = self.core.encode_groups(list(hidden_states.split(lengths)))
        self.current_input_fields = fields
        field, routing = self.core.begin(fields)
        for index in range(len(self.core.expert_layers)):
            field = self.core.run_stage(index, field, routing)
        # This is the exact existing final optical path: global phase, 10 cm
        # propagation, square-law CCD crop/pool/LN/nonlinearity. The unused
        # hidden output adapter is deliberately not called.
        field = self.core.propagator(self.core.global_phase(field))
        readout, intensity = self.core.readout(field)
        if not torch.isfinite(readout).all() or torch.any(readout < 0):
            raise RuntimeError("Vision CCD readout must be finite and nonnegative")
        self.core.current_detector_readout = readout
        if self.core.capture_intermediate_fields:
            count = min(self.core.capture_sample_count, len(field))
            self.core.last_detector_intensity = intensity[:count].detach().cpu()
            self.core.last_detector_readout = readout[:count].detach().cpu()
        return hidden_states


class VisionOpticalSaliencyStudent(nn.Module):
    """Frozen Qwen patch/position stem + existing MoE16 CCD + segmentation head."""

    def __init__(
        self,
        loaded: LoadedVisionBackbone,
        settings: Any,
        head: LightweightSegmentationHead,
    ) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.original_blocks = list(self.visual.blocks)
        self.core = HomogeneousMoEOpticalCore(
            settings.vision_hidden_size,
            settings.max_visual_tokens,
            settings,
        ).to(loaded.device)
        self.core.output_adapter.requires_grad_(False)
        self.capture_block = _OpticalCaptureBlock(self.core)
        self.head = head
        self.student_blocks = nn.ModuleList(
            [self.capture_block]
            + [_VisionBypass() for _ in range(len(self.original_blocks) - 1)]
        )
        self._student_active = False

    def activate(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.visual.blocks[index] = block
        self._student_active = True

    def restore_native(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.visual.blocks[index] = block
        self._student_active = False

    def train(self, mode: bool = True) -> "VisionOpticalSaliencyStudent":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.head.train(mode)
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._student_active:
            self.activate()
        visual_dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(visual_dtype), grid_thw=image_grid_thw)
        detector = self.core.current_detector_readout
        if detector is None:
            raise RuntimeError("Optical Vision path did not produce a CCD readout")
        spatial = restore_detector_spatial(
            detector, image_grid_thw, self.capture_block.token_counts
        )
        input_fields = self.capture_block.current_input_fields
        input_spatial = (
            restore_detector_spatial(
                input_fields,
                image_grid_thw,
                self.capture_block.token_counts,
            )
            if self.head.detector_residual_enabled and input_fields is not None
            else None
        )
        logits = self.head(spatial, input_residual=input_spatial)
        return logits, spatial, detector

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()


def build_teacher(loaded: LoadedVisionBackbone, settings: Any) -> FrozenQwenVisionTeacher:
    head = LightweightSegmentationHead(
        input_dim=settings.vision_hidden_size,
        projection_dim=settings.segmentation_projection_dim,
        decoder_channels=settings.segmentation_channels,
        groupnorm_groups=settings.segmentation_groupnorm_groups,
        output_size=settings.image_size,
        refinement_enabled=False,
        progressive_refinement_enabled=False,
        detector_residual_enabled=False,
    ).to(loaded.device)
    return FrozenQwenVisionTeacher(loaded, head)


def build_student(loaded: LoadedVisionBackbone, settings: Any) -> VisionOpticalSaliencyStudent:
    head = LightweightSegmentationHead(
        input_dim=settings.detector_output_size,
        projection_dim=settings.segmentation_projection_dim,
        decoder_channels=settings.segmentation_channels,
        groupnorm_groups=settings.segmentation_groupnorm_groups,
        output_size=settings.image_size,
        refinement_enabled=settings.student_segmentation_refinement_enabled,
        progressive_refinement_enabled=(
            settings.student_segmentation_progressive_refinement_enabled
        ),
        detector_residual_enabled=settings.student_detector_residual_enabled,
        detector_identity_scale_init=settings.student_detector_identity_scale_init,
        detector_input_scale_init=settings.student_detector_input_scale_init,
        detector_identity_scale_trainable=(
            settings.student_detector_identity_scale_trainable
        ),
        detector_input_scale_trainable=(
            settings.student_detector_input_scale_trainable
        ),
    ).to(loaded.device)
    student = VisionOpticalSaliencyStudent(loaded, settings, head)
    # Native Qwen and the unused hidden restore adapter remain frozen.
    loaded.model.requires_grad_(False)
    student.core.requires_grad_(True)
    student.core.output_adapter.requires_grad_(False)
    student.head.requires_grad_(True)
    return student


def trainable_parameter_report(module: nn.Module, *, prefix: str) -> dict[str, Any]:
    values = []
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            values.append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": list(parameter.shape),
                    "parameters": parameter.numel(),
                }
            )
            seen.add(id(parameter))
    return {
        "trainable_parameters": sum(item["parameters"] for item in values),
        "trainable_tensors": len(values),
        "trainable_parameter_list": values,
    }


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(values)}") from exc
