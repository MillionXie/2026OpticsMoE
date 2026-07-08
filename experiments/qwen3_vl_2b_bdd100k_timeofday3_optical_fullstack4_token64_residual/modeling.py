from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LoadedBackbone:
    model: nn.Module
    processor: Any
    device: torch.device
    load_time_sec: float


HEAD_TYPES = {"mlp", "linear", "bottleneck", "normalized_linear"}


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float,
                 head_type: str = "mlp", hidden_dim: int | None = None,
                 bottleneck_dim: int = 128, use_layernorm: bool = False) -> None:
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"Unsupported head_type={head_type!r}; expected one of {sorted(HEAD_TYPES)}")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when set")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.head_type = str(head_type)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else None
        self.bottleneck_dim = int(bottleneck_dim)
        self.use_layernorm = bool(use_layernorm or head_type == "normalized_linear")
        if head_type == "mlp":
            if self.hidden_dim is None:
                raise ValueError("mlp head requires hidden_dim")
            layers = [nn.Linear(feature_dim, self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
                      nn.Linear(self.hidden_dim, num_classes)]
        elif head_type == "linear":
            layers = [nn.Linear(feature_dim, num_classes)]
        elif head_type == "bottleneck":
            layers = []
            if self.use_layernorm:
                layers.append(nn.LayerNorm(feature_dim))
            layers.extend([nn.Linear(feature_dim, bottleneck_dim), nn.GELU(), nn.Dropout(dropout),
                           nn.Linear(bottleneck_dim, num_classes)])
        else:
            layers = [nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes)]
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def specification(self) -> dict[str, Any]:
        return {
            "type": self.head_type,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "use_layernorm": self.use_layernorm,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


class MLPHead(ClassificationHead):
    """Backward-compatible wrapper for the original two-layer MLP head."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout,
                         head_type="mlp", hidden_dim=hidden_dim)


def build_head(settings: Any, feature_dim: int, num_classes: int) -> ClassificationHead:
    head_type = str(getattr(settings, "head_type", "mlp"))
    configured_hidden = getattr(settings, "head_hidden_dim", None)
    hidden_dim = configured_hidden if configured_hidden is not None else getattr(settings, "hidden_dim", 1024)
    return ClassificationHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=float(getattr(settings, "dropout", 0.1)),
        head_type=head_type,
        hidden_dim=int(hidden_dim) if head_type == "mlp" else None,
        bottleneck_dim=int(getattr(settings, "head_bottleneck_dim", 128)),
        use_layernorm=bool(getattr(settings, "head_use_layernorm", False)),
    )


def optical_surrogate_parameter_breakdown(surrogate: nn.Module, prefix: str) -> dict[str, int]:
    def counts(module: nn.Module) -> tuple[int, int, int]:
        parameters=list(module.parameters())
        total=sum(parameter.numel() for parameter in parameters)
        trainable=sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return total,trainable,total-trainable

    result: dict[str,int]={}
    adapter_total=adapter_trainable=adapter_non_trainable=0
    for name in ("input_adapter","adapter_norm","output_adapter"):
        total,trainable,non_trainable=counts(getattr(surrogate,name))
        result[f"{prefix}_{name}_parameters"]=total
        result[f"{prefix}_{name}_trainable_parameters"]=trainable
        result[f"{prefix}_{name}_non_trainable_parameters"]=non_trainable
        adapter_total+=total;adapter_trainable+=trainable;adapter_non_trainable+=non_trainable
    result[f"{prefix}_adapter_total_parameters"]=adapter_total
    result[f"{prefix}_adapter_trainable_parameters"]=adapter_trainable
    result[f"{prefix}_adapter_non_trainable_parameters"]=adapter_non_trainable
    for component,attribute in (("phase_mask","phase_mask"),("amplitude_mask","amplitude_mask_logits"),("detector_bias","detector_bias")):
        tensors=[getattr(conversion,attribute) for conversion in surrogate.conversions if getattr(conversion,attribute,None) is not None]
        total=sum(tensor.numel() for tensor in tensors)
        trainable=sum(tensor.numel() for tensor in tensors if isinstance(tensor,nn.Parameter) and tensor.requires_grad)
        result[f"{prefix}_{component}_parameters"]=total
        result[f"{prefix}_{component}_trainable_parameters"]=trainable
        result[f"{prefix}_{component}_non_trainable_parameters"]=total-trainable
    scale_tensors=[surrogate.identity_scale,surrogate.modulated_scale]
    scale_total=sum(tensor.numel() for tensor in scale_tensors)
    scale_trainable=sum(tensor.numel() for tensor in scale_tensors if isinstance(tensor,nn.Parameter) and tensor.requires_grad)
    result[f"{prefix}_residual_scale_parameters"]=scale_total
    result[f"{prefix}_residual_scale_trainable_parameters"]=scale_trainable
    result[f"{prefix}_residual_scale_non_trainable_parameters"]=scale_total-scale_trainable
    physical_total=result[f"{prefix}_phase_mask_parameters"]+result[f"{prefix}_amplitude_mask_parameters"]
    physical_trainable=result[f"{prefix}_phase_mask_trainable_parameters"]+result[f"{prefix}_amplitude_mask_trainable_parameters"]
    result[f"{prefix}_optical_physical_parameters"]=physical_total
    result[f"{prefix}_optical_physical_trainable_parameters"]=physical_trainable
    result[f"{prefix}_optical_physical_non_trainable_parameters"]=physical_total-physical_trainable
    pytorch_total,trainable,_=counts(surrogate)
    component_total=adapter_total+physical_total+result[f"{prefix}_detector_bias_parameters"]+scale_total
    result[f"{prefix}_surrogate_total_parameters"]=component_total
    result[f"{prefix}_surrogate_trainable_parameters"]=trainable
    result[f"{prefix}_surrogate_non_trainable_parameters"]=component_total-trainable
    result[f"{prefix}_surrogate_pytorch_parameter_total"]=pytorch_total
    return result


def student_parameter_breakdown(vision: nn.Module, language: nn.Module, head: nn.Module) -> dict[str, Any]:
    vision_report=optical_surrogate_parameter_breakdown(vision,"vision")
    language_report=optical_surrogate_parameter_breakdown(language,"language")
    head_total=sum(parameter.numel() for parameter in head.parameters())
    head_trainable=sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    adapter_trainable=vision_report["vision_adapter_trainable_parameters"]+language_report["language_adapter_trainable_parameters"]
    phase_trainable=vision_report["vision_phase_mask_trainable_parameters"]+language_report["language_phase_mask_trainable_parameters"]
    amplitude_trainable=vision_report["vision_amplitude_mask_trainable_parameters"]+language_report["language_amplitude_mask_trainable_parameters"]
    detector_trainable=vision_report["vision_detector_bias_trainable_parameters"]+language_report["language_detector_bias_trainable_parameters"]
    residual_trainable=vision_report["vision_residual_scale_trainable_parameters"]+language_report["language_residual_scale_trainable_parameters"]
    student_total=vision_report["vision_surrogate_trainable_parameters"]+language_report["language_surrogate_trainable_parameters"]+head_trainable
    overall={
        "vision_adapter_trainable":vision_report["vision_adapter_trainable_parameters"],
        "language_adapter_trainable":language_report["language_adapter_trainable_parameters"],
        "adapter_trainable_total":adapter_trainable,
        "optical_phase_trainable":phase_trainable,
        "optical_amplitude_trainable":amplitude_trainable,
        "optical_physical_trainable_total":phase_trainable+amplitude_trainable,
        "detector_bias_trainable":detector_trainable,
        "residual_scale_trainable":residual_trainable,
        "classification_head_trainable":head_trainable,
        "student_total_trainable":student_total,
        "adapter_ratio_in_student_trainable":adapter_trainable/student_total if student_total else 0.0,
        "optical_physical_ratio_in_student_trainable":(phase_trainable+amplitude_trainable)/student_total if student_total else 0.0,
        "classification_head_ratio_in_student_trainable":head_trainable/student_total if student_total else 0.0,
    }
    return {**vision_report,**language_report,"classification_head_parameters":head_total,
            "classification_head_trainable_parameters":head_trainable,"parameter_breakdown":overall}


def load_backbone(
    model_id: str,
    cache_dir: Path | None,
    local_files_only: bool,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> LoadedBackbone:
    transformers = importlib.import_module("transformers")
    processor_cls = transformers.AutoProcessor
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError("Installed transformers does not support Qwen3-VL")

    common: dict[str, Any] = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": local_files_only,
    }
    processor_kwargs = {key: value for key, value in common.items() if value is not None}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    model_kwargs = {key: value for key, value in common.items() if value is not None}
    model_kwargs.update(
        {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
    )
    started = time.perf_counter()
    processor = processor_cls.from_pretrained(model_id, **processor_kwargs)
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return LoadedBackbone(model, processor, device, time.perf_counter() - started)


def parameter_report(model: nn.Module, head: nn.Module | None = None) -> dict[str, Any]:
    def describe(module: nn.Module | None) -> dict[str, Any]:
        if module is None:
            return {"parameters": 0, "trainable_parameters": 0, "bytes": 0, "gib": 0.0}
        parameters = list(module.parameters())
        count = sum(value.numel() for value in parameters)
        trainable = sum(value.numel() for value in parameters if value.requires_grad)
        size = sum(value.numel() * value.element_size() for value in parameters)
        return {
            "parameters": count,
            "trainable_parameters": trainable,
            "bytes": size,
            "gib": size / 1024**3,
        }

    core = getattr(model, "model", model)
    visual = getattr(core, "visual", None)
    language = getattr(core, "language_model", None)
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    text_config = getattr(config, "text_config", config)
    return {
        "backbone": describe(model),
        "vision": describe(visual),
        "language": describe(language),
        "mlp_head": describe(head),
        "architecture": {
            "vision_hidden_size": getattr(vision_config, "hidden_size", None),
            "vision_depth": getattr(vision_config, "depth", None),
            "vision_out_hidden_size": getattr(vision_config, "out_hidden_size", None),
            "spatial_merge_size": getattr(vision_config, "spatial_merge_size", None),
            "text_hidden_size": getattr(text_config, "hidden_size", None),
            "text_layers": getattr(text_config, "num_hidden_layers", None),
        },
    }
