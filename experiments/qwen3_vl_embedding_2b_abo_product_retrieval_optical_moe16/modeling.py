from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    OpticalRetrievalReadout,
    build_optical_student,
    resolve_cached_model_source,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
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
    """Load Qwen once, but move only the frozen Vision module onto the GPU."""
    transformers = importlib.import_module("transformers")
    source = resolve_cached_model_source(settings.model_id, settings.cache_dir)
    using_snapshot = source != settings.model_id
    if using_snapshot:
        print(f"[model] using local Hugging Face snapshot: {source}", flush=True)
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
        raise RuntimeError("Installed transformers has no Qwen3-VL model class")
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
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Qwen processor has no image_processor")
    values = image_processor(images=images, return_tensors="pt")
    missing = [
        key for key in ("pixel_values", "image_grid_thw") if key not in values
    ]
    if missing:
        raise RuntimeError(
            f"Qwen image processor omitted {missing}; returned {list(values.keys())}"
        )
    return {
        key: values[key].to(device, non_blocking=True)
        for key in ("pixel_values", "image_grid_thw")
    }


class _VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class OpticalVisionCapture(nn.Module):
    """Replace native Vision blocks with one MoE16 stage plus the global phase."""

    def __init__(self, core: HomogeneousMoEOpticalCore) -> None:
        super().__init__()
        self.core = core
        self.token_counts: list[int] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        if any(length > self.core.max_tokens for length in lengths):
            raise RuntimeError(
                f"visual token count {max(lengths)} exceeds optical rows="
                f"{self.core.max_tokens}; lower processor_max_pixels. No crop, "
                "pooling, reshape, or token truncation is allowed."
            )
        self.token_counts = lengths
        input_fields = self.core.encode_groups(list(hidden_states.split(lengths)))
        field, routing = self.core.begin(input_fields)
        if len(self.core.expert_layers) != 1:
            raise RuntimeError(
                "ABO retrieval requires exactly one expert phase stage"
            )
        field = self.core.run_stage(0, field, routing)
        field = self.core.propagator(self.core.global_phase(field))
        readout, intensity = self.core.readout(field)
        if readout.shape[1:] != (
            self.core.geometry.expert_size,
            self.core.geometry.expert_size,
        ):
            raise RuntimeError(
                f"CCD readout must be [B,224,224], got {tuple(readout.shape)}"
            )
        if not torch.isfinite(readout).all() or torch.any(readout < 0):
            raise RuntimeError("CCD detector features must be finite and nonnegative")
        self.core.current_detector_readout = readout
        if self.core.capture_intermediate_fields:
            count = min(self.core.capture_sample_count, len(field))
            self.core.last_detector_intensity = intensity[:count].detach().cpu()
            self.core.last_detector_readout = readout[:count].detach().cpu()
        # The native merger may finish its bookkeeping, but its output is not
        # consumed. The optical embedding is read directly from the CCD tensor.
        return hidden_states


class DetectorTokenProjection(nn.Module):
    """CCD token features -> signed 224-D token features -> pooled embedding."""

    def __init__(self, dimension: int = 224) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.norm = nn.LayerNorm(self.dimension)
        self.projection = nn.Linear(self.dimension, self.dimension)
        self.last_token_features: list[torch.Tensor] = []

    def forward(
        self, detector: torch.Tensor, token_counts: list[int]
    ) -> torch.Tensor:
        if detector.ndim != 3 or detector.shape[1:] != (
            self.dimension,
            self.dimension,
        ):
            raise RuntimeError(
                f"Detector projection expects [B,{self.dimension},{self.dimension}], "
                f"got {tuple(detector.shape)}"
            )
        if detector.shape[0] != len(token_counts):
            raise RuntimeError("CCD batch and visual token count list differ")
        pooled: list[torch.Tensor] = []
        token_features: list[torch.Tensor] = []
        for sample_index, count in enumerate(token_counts):
            if count <= 0 or count > self.dimension:
                raise RuntimeError(
                    f"Invalid visual token count {count} for 224-row CCD readout"
                )
            valid = detector[sample_index, :count].float()
            signed = self.projection(self.norm(valid))
            token_features.append(signed)
            pooled.append(signed.mean(dim=0))
        packed = torch.stack(pooled)
        norms = packed.norm(dim=-1)
        if not torch.isfinite(packed).all() or torch.any(norms <= 1e-12):
            raise RuntimeError(
                "Optical readout produced NaN/Inf or a zero-norm embedding"
            )
        self.last_token_features = token_features
        return F.normalize(packed, p=2, dim=-1)

    def specification(self) -> dict[str, Any]:
        return {
            "architecture": (
                "CCD [B,224,224] -> valid rows -> LayerNorm(224) -> "
                "Linear(224,224) -> mean token pooling -> L2 Normalize"
            ),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


class VisionOpticalRetrievalEncoder(nn.Module):
    """Frozen Qwen patch/position stem and trainable one-layer Optical MoE16."""

    def __init__(self, loaded: LoadedVisionBackbone, settings: Any) -> None:
        super().__init__()
        self.settings = settings
        self.visual = loaded.visual
        if not hasattr(self.visual, "blocks") or not len(self.visual.blocks):
            raise RuntimeError("Qwen visual module has no transformer blocks")
        self.original_blocks = list(self.visual.blocks)
        self.core = HomogeneousMoEOpticalCore(
            settings.vision_hidden_size,
            settings.max_visual_tokens,
            settings,
        ).to(loaded.device)
        # This experiment reads the CCD directly. The generic 224->Qwen hidden
        # restore adapter is neither called nor trainable.
        self.core.output_adapter.requires_grad_(False)
        self.capture = OpticalVisionCapture(self.core)
        self.readout = DetectorTokenProjection(settings.embedding_dim).to(
            loaded.device
        )
        self.student_blocks = nn.ModuleList(
            [self.capture]
            + [_VisionBypass() for _ in range(len(self.original_blocks) - 1)]
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

    def train(self, mode: bool = True) -> "VisionOpticalRetrievalEncoder":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.readout.train(mode)
        return self

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if not self._active:
            self.activate()
        visual_dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(visual_dtype), grid_thw=image_grid_thw)
        detector = self.core.current_detector_readout
        if detector is None:
            raise RuntimeError("Optical Vision path did not expose its CCD readout")
        return self.readout(detector, self.capture.token_counts)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def router_loss_components(self) -> dict[str, torch.Tensor]:
        balance, importance = self.core.router_losses()
        entropy = self.core.last_routing["normalized_entropy"]
        return {
            "vision_balance": balance,
            "vision_importance": importance,
            "vision_entropy_penalty": 1.0 - entropy,
            "language_balance": balance.new_zeros(()),
            "language_importance": balance.new_zeros(()),
            "language_entropy_penalty": balance.new_zeros(()),
        }

    def set_router_temperature(self, value: float) -> None:
        self.core.router.router.temperature = float(value)

    def router_diagnostics(self) -> dict[str, dict[str, torch.Tensor]]:
        return {"vision": self.core.last_routing}

    def deployment_state_dict(self) -> dict[str, Any]:
        return {
            "architecture": "vision_only",
            "optical_core": self.core.state_dict(),
            "detector_projection": self.readout.state_dict(),
        }

    def load_deployment_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("architecture") not in {None, "vision_only"}:
            raise RuntimeError("Checkpoint is not a vision-only optical encoder")
        self.core.load_state_dict(state["optical_core"])
        self.readout.load_state_dict(state["detector_projection"])


class MultimodalOpticalRetrievalEncoder(nn.Module):
    """One-layer Vision + one-layer Language Optical MoE retrieval student."""

    def __init__(self, loaded: LoadedBackbone, settings: Any) -> None:
        super().__init__()
        self.settings = settings
        self.model = loaded.model
        self.replacement, self.readout = build_optical_student(loaded, settings)
        # DeepStackMultimodalReplacement is an orchestration object rather than
        # nn.Module. Register every trainable optical module explicitly so
        # encoder.parameters(), optimizers and checkpoints cannot silently see
        # only the electronic retrieval readout while the surrogates sit outside
        # the module tree.
        self.vision_optical = self.replacement.vision_surrogate
        self.language_optical = self.replacement.language_surrogate
        if self.replacement.vision_pre_attention is not None:
            self.vision_pre_attention = self.replacement.vision_pre_attention
        if self.replacement.language_pre_attention is not None:
            self.language_pre_attention = self.replacement.language_pre_attention
        self.device = loaded.device

    def train(self, mode: bool = True) -> "MultimodalOpticalRetrievalEncoder":
        super().train(mode)
        self.model.eval()
        self.replacement.set_student_train_mode()
        self.readout.train(mode)
        return self

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embedding, _ = student_embeddings(
            self.model, self.replacement, self.readout, inputs
        )
        return embedding

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.replacement.router_losses()
        return (
            values["vision_balance"] + values["language_balance"],
            values["vision_importance"] + values["language_importance"],
        )

    def router_loss_components(self) -> dict[str, torch.Tensor]:
        values = self.replacement.router_losses()
        vision_entropy = (
            1.0
            - self.replacement.vision_surrogate.core.last_routing[
                "normalized_entropy"
            ]
        )
        language_entropy = (
            1.0
            - self.replacement.language_surrogate.core.last_routing[
                "normalized_entropy"
            ]
        )
        return {
            **values,
            "vision_entropy_penalty": vision_entropy,
            "language_entropy_penalty": language_entropy,
        }

    def set_router_temperature(self, value: float) -> None:
        self.replacement.vision_surrogate.core.router.router.temperature = float(
            value
        )
        self.replacement.language_surrogate.core.router.router.temperature = float(
            value
        )

    def router_diagnostics(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "vision": self.replacement.vision_surrogate.core.last_routing,
            "language": self.replacement.language_surrogate.core.last_routing,
        }

    def deployment_state_dict(self) -> dict[str, Any]:
        return {
            "architecture": "multimodal_optical",
            "vision_optical": self.replacement.vision_surrogate.state_dict(),
            "language_optical": self.replacement.language_surrogate.state_dict(),
            "retrieval_readout": self.readout.state_dict(),
            "prelude": self.replacement.prelude_state_dict(),
        }

    def load_deployment_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("architecture") != "multimodal_optical":
            raise RuntimeError("Checkpoint is not a multimodal optical encoder")
        self.replacement.vision_surrogate.load_state_dict(
            state["vision_optical"]
        )
        self.replacement.language_surrogate.load_state_dict(
            state["language_optical"]
        )
        self.readout.load_state_dict(state["retrieval_readout"])
        self.replacement.load_prelude_state_dict(state.get("prelude"))

    def restore_native(self) -> None:
        self.replacement.close()


class TrainingIdentityHead(nn.Module):
    """Stage-2-only product identity classifier; never used for retrieval."""

    def __init__(self, embedding_dim: int, item_count: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, item_count)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding.float())


def build_encoder(
    loaded: LoadedVisionBackbone | LoadedBackbone, settings: Any
) -> VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder:
    if settings.student_architecture == "multimodal_optical":
        if not isinstance(loaded, LoadedBackbone):
            raise TypeError("Multimodal optical Student needs the full Qwen backbone")
        return MultimodalOpticalRetrievalEncoder(loaded, settings)
    encoder = VisionOpticalRetrievalEncoder(loaded, settings).to(loaded.device)
    # Qwen is frozen. Optical core except the unused restore adapter and the
    # detector projection are trainable.
    loaded.model.requires_grad_(False)
    encoder.core.requires_grad_(True)
    encoder.core.output_adapter.requires_grad_(False)
    encoder.readout.requires_grad_(True)
    return encoder


def encode_product_images(
    loaded: LoadedVisionBackbone | LoadedBackbone,
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
    images: list[Any],
    settings: Any,
) -> torch.Tensor:
    if isinstance(encoder, MultimodalOpticalRetrievalEncoder):
        inputs = preprocess_images(
            loaded.processor, images, settings.instruction
        )
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        return encoder(inputs)
    inputs = preprocess_vision(loaded.processor, images, loaded.device)
    return encoder(inputs["pixel_values"], inputs["image_grid_thw"])


def unique_trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    values: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                values.append(parameter)
                seen.add(id(parameter))
    return values


def parameter_report(
    encoder: VisionOpticalRetrievalEncoder | MultimodalOpticalRetrievalEncoder,
    identity_head: nn.Module | None = None,
) -> dict[str, Any]:
    modules: list[nn.Module] = [encoder]
    if identity_head is not None:
        modules.append(identity_head)
    parameters = unique_trainable_parameters(*modules)
    rows = []
    for prefix, module in (
        ("optical_encoder", encoder),
        ("identity_head", identity_head),
    ):
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                rows.append(
                    {
                        "name": f"{prefix}.{name}",
                        "shape": list(parameter.shape),
                        "parameters": parameter.numel(),
                    }
                )
    multimodal = isinstance(encoder, MultimodalOpticalRetrievalEncoder)
    return {
        "architecture": (
            "Frozen Qwen patch/token stems -> one-layer Vision Optical MoE16 -> "
            "frozen merger/injection -> one-layer Language Optical MoE16 -> "
            "language CCD -> LN+Linear -> L2-normalized 224D embedding"
            if multimodal
            else
            "Frozen Qwen patch/position stem -> one-layer Optical MoE16 -> "
            "Global phase -> CCD 224x224 -> LN+Linear per valid row -> "
            "mean pooling -> L2-normalized 224D embedding"
        ),
        "optical": (
            {
                "vision": encoder.replacement.vision_surrogate.parameter_breakdown(),
                "language": encoder.replacement.language_surrogate.parameter_breakdown(),
            }
            if multimodal
            else encoder.core.parameter_breakdown()
        ),
        "detector_projection": encoder.readout.specification(),
        "identity_head": (
            None
            if identity_head is None
            else {
                "training_only": True,
                "parameters": sum(
                    parameter.numel() for parameter in identity_head.parameters()
                ),
            }
        ),
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
        "trainable_tensors": len(parameters),
        "trainable_parameter_list": rows,
    }


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}")
    return values[name]
