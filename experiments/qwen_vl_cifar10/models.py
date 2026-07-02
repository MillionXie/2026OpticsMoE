from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn


SUPPORTED_MODEL_IDS = (
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
)
DEFAULT_MODEL_ID = SUPPORTED_MODEL_IDS[0]


@dataclass
class LoadedQwen:
    model: nn.Module
    processor: Any


class MLPHead(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_qwen(
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    device_map: str | None,
    trust_remote_code: bool,
) -> LoadedQwen:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "transformers with Qwen3-VL support is required. Install this experiment's "
            "requirements.txt."
        ) from exc

    processor_cls = getattr(transformers, "AutoProcessor", None)
    if processor_cls is None:
        raise RuntimeError(
            "The installed transformers package does not expose AutoProcessor."
        )

    model_cls = _resolve_model_class(transformers, model_id)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        try:
            importlib.import_module("accelerate")
        except ImportError as exc:
            raise RuntimeError(
                "--device-map auto requires accelerate. Install it with `pip install accelerate`."
            ) from exc
        load_kwargs["device_map"] = device_map

    try:
        processor = processor_cls.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        model = model_cls.from_pretrained(model_id, **load_kwargs)
    except (KeyError, ValueError, AttributeError) as exc:
        version = getattr(transformers, "__version__", "unknown")
        raise RuntimeError(
            f"Failed to load Qwen3-VL with transformers {version}. Install a release that includes "
            "Qwen3-VL dense and MoE mappings (see requirements.txt)."
        ) from exc

    if device_map is None:
        model.to(device)
    return LoadedQwen(model=model, processor=processor)


def _resolve_model_class(transformers: Any, model_id: str) -> Any:
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls

    is_moe = "30B-A3B" in model_id or "235B-A22B" in model_id
    class_name = (
        "Qwen3VLMoeForConditionalGeneration"
        if is_moe
        else "Qwen3VLForConditionalGeneration"
    )
    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise RuntimeError(
            "The installed transformers package lacks Qwen3-VL model classes. "
            "Install a current transformers release."
        )
    return cls


def freeze_backbone(model: nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


def apply_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    forbidden = ("mm_mlp", "projector", "multi_modal_projector")
    invalid = [
        name
        for name in target_modules
        if any(part in name.lower() for part in forbidden)
    ]
    if invalid:
        raise ValueError(
            f"LoRA target modules {invalid} are forbidden. This experiment never tunes "
            "mm_mlp/projector modules."
        )
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA mode requires peft. Install it with `pip install peft`."
        ) from exc

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(model, config)


def backbone_core(model: nn.Module) -> nn.Module:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    core = getattr(base, "model", None)
    if core is None:
        raise RuntimeError(
            "Unable to locate the Qwen3-VL multimodal backbone (`model.model`)."
        )
    return core


def model_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        embeddings = base.get_input_embeddings()
        return next(embeddings.parameters()).device
    except (AttributeError, StopIteration):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return fallback
