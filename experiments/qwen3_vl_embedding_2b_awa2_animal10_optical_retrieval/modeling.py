from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .optics.moe import LanguageDeepStackHomogeneousMoE, VisionDeepStackHomogeneousMoE
from .optics.replacement import DeepStackMultimodalReplacement


@dataclass(frozen=True)
class LoadedBackbone:
    model: nn.Module
    processor: Any
    device: torch.device
    load_time_sec: float


class OpticalRetrievalReadout(nn.Module):
    """Signed 64-D retrieval readout; deliberately contains no activation."""

    def __init__(self, detector_dim: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.detector_dim = int(detector_dim)
        self.embedding_dim = int(embedding_dim)
        self.norm = nn.LayerNorm(self.detector_dim)
        self.projection = nn.Linear(self.detector_dim, self.embedding_dim)

    def forward_unnormalized(self, detector_features: torch.Tensor) -> torch.Tensor:
        if detector_features.ndim != 2 or detector_features.shape[-1] != self.detector_dim:
            raise RuntimeError(
                f"Detector features must be [B,{self.detector_dim}], got "
                f"{tuple(detector_features.shape)}"
            )
        if not torch.isfinite(detector_features).all():
            raise RuntimeError("Detector features contain NaN or Inf")
        return self.projection(self.norm(detector_features.float()))

    def forward(self, detector_features: torch.Tensor) -> torch.Tensor:
        raw = self.forward_unnormalized(detector_features)
        norms = raw.norm(dim=-1)
        if not torch.isfinite(raw).all() or torch.any(norms <= 1e-12):
            raise RuntimeError("Retrieval readout produced NaN/Inf or a zero-norm embedding")
        output = F.normalize(raw, p=2, dim=-1)
        if not torch.isfinite(output).all():
            raise RuntimeError("Normalized student embedding contains NaN or Inf")
        return output

    def specification(self) -> dict[str, Any]:
        return {
            "type": "optical_retrieval_readout",
            "architecture": (
                f"LayerNorm({self.detector_dim}) -> Linear({self.detector_dim},"
                f"{self.embedding_dim}) -> L2Normalize"
            ),
            "post_linear_activation": None,
            "detector_dim": self.detector_dim,
            "embedding_dim": self.embedding_dim,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def official_mrl_embedding(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_dim: int = 64,
) -> torch.Tensor:
    """Official Qwen embedding pooling followed by Matryoshka truncation.

    Qwen3-VL-Embedding pools the last valid language token and L2 normalizes.
    Its low-dimensional interface retains the leading Matryoshka dimensions
    and normalizes again. No learned 2048->64 projection is introduced here.
    """
    if last_hidden_state.ndim != 3:
        raise RuntimeError(
            f"Expected Qwen hidden [B,S,D], got {tuple(last_hidden_state.shape)}"
        )
    if attention_mask.shape != last_hidden_state.shape[:2]:
        raise RuntimeError("attention_mask must match Qwen hidden [B,S]")
    if embedding_dim <= 0 or embedding_dim > last_hidden_state.shape[-1]:
        raise ValueError(
            f"embedding_dim={embedding_dim} is invalid for hidden size "
            f"{last_hidden_state.shape[-1]}"
        )
    positions = torch.arange(
        last_hidden_state.shape[1], device=last_hidden_state.device
    ).unsqueeze(0)
    positions = positions.expand_as(attention_mask).masked_fill(attention_mask.eq(0), -1)
    last_positions = positions.max(dim=1).values
    if torch.any(last_positions < 0):
        raise RuntimeError("Every Qwen embedding sample needs at least one valid token")
    batch = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    pooled = last_hidden_state[batch, last_positions].float()
    reduced = pooled[:, :embedding_dim]
    norms = reduced.norm(dim=-1)
    if not torch.isfinite(reduced).all() or torch.any(norms <= 1e-12):
        raise RuntimeError("Teacher produced NaN/Inf or zero-norm low-dimensional embedding")
    return F.normalize(reduced, p=2, dim=-1)


def load_backbone(settings: Any, device: torch.device) -> LoadedBackbone:
    transformers = importlib.import_module("transformers")
    source = resolve_cached_model_source(settings.model_id, settings.cache_dir)
    using_snapshot = source != settings.model_id
    common = {
        "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
        "local_files_only": settings.local_files_only or using_snapshot,
        "trust_remote_code": True,
    }
    processor = transformers.AutoProcessor.from_pretrained(
        source,
        min_pixels=settings.processor_min_pixels,
        max_pixels=settings.processor_max_pixels,
        **{key: value for key, value in common.items() if value is not None},
    )
    # The official embedding helper defines a thin Qwen3VLForEmbedding wrapper
    # around Qwen3VLModel. The public checkpoint config itself declares
    # Qwen3VLForConditionalGeneration, whose `.model` is the same base model.
    # Loading that supported class and consuming only `.model.last_hidden_state`
    # reproduces the official embedding forward without importing a second,
    # out-of-tree model implementation.
    model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        raise RuntimeError(
            "Installed transformers does not expose Qwen3VLForConditionalGeneration or "
            "AutoModelForImageTextToText. "
            "Install the Qwen3-VL-Embedding model-card requirements."
        )
    kwargs = {key: value for key, value in common.items() if value is not None}
    kwargs.update(
        {
            "dtype": _dtype(settings.dtype),
            "low_cpu_mem_usage": True,
            "attn_implementation": settings.attn_implementation,
        }
    )
    started = time.perf_counter()
    model = model_class.from_pretrained(source, **kwargs)
    model.to(device).requires_grad_(False).eval()
    return LoadedBackbone(model, processor, device, time.perf_counter() - started)


def build_optical_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[
    DeepStackMultimodalReplacement,
    OpticalRetrievalReadout,
]:
    settings.resolve_architecture(loaded.model)
    vision = VisionDeepStackHomogeneousMoE(settings.vision_hidden_size, settings)
    language = LanguageDeepStackHomogeneousMoE(settings.text_hidden_size, settings)
    vision.to(loaded.device)
    language.to(loaded.device)
    replacement = DeepStackMultimodalReplacement(
        loaded.model, vision, language, settings
    )
    readout = OpticalRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


def unique_trainable_parameters(
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
) -> list[nn.Parameter]:
    values: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in list(replacement.trainable_parameters()) + list(readout.parameters()):
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            values.append(parameter)
    return values


def trainable_parameter_report(
    model: nn.Module,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
) -> dict[str, Any]:
    trainable_ids = {
        id(parameter)
        for parameter in unique_trainable_parameters(replacement, readout)
    }
    names: list[dict[str, Any]] = []
    named_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in trainable_ids and id(parameter) not in named_ids:
            names.append(
                {
                    "name": f"qwen_replacement.{name}",
                    "shape": list(parameter.shape),
                    "parameters": parameter.numel(),
                }
            )
            named_ids.add(id(parameter))
    for name, parameter in readout.named_parameters():
        names.append(
            {
                "name": f"retrieval_readout.{name}",
                "shape": list(parameter.shape),
                "parameters": parameter.numel(),
            }
        )
    # In teacher mode the replacement modules are not registered under the
    # Qwen tree, so add any parameters not found through model.named_parameters.
    missing = trainable_ids - named_ids - {id(value) for value in readout.parameters()}
    for prefix, module in (
        ("vision_optical", replacement.vision_surrogate),
        ("language_optical", replacement.language_surrogate),
    ):
        for name, parameter in module.named_parameters():
            if id(parameter) in missing:
                names.append(
                    {
                        "name": f"{prefix}.{name}",
                        "shape": list(parameter.shape),
                        "parameters": parameter.numel(),
                    }
                )
                missing.remove(id(parameter))
    if missing:
        raise RuntimeError(f"Could not name {len(missing)} trainable parameters")
    vision = replacement.vision_surrogate.parameter_breakdown()
    language = replacement.language_surrogate.parameter_breakdown()
    count = sum(parameter.numel() for parameter in unique_trainable_parameters(replacement, readout))
    return {
        "teacher_model_id": getattr(model, "name_or_path", type(model).__name__),
        "teacher_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()
                                         if id(parameter) not in trainable_ids),
        "student_architecture": {
            "optical_structure": "one_expert_stage_plus_one_global_phase",
            "vision_expert_stages": len(
                replacement.vision_surrogate.core.expert_layers
            ),
            "language_expert_stages": len(
                replacement.language_surrogate.core.expert_layers
            ),
            "global_phase_planes_per_stack": 1,
            "vision_tap_stages": list(replacement.vision_surrogate.tap_stages),
            "student_deepstack_visual_indexes": list(
                replacement.deepstack_indexes
            ),
            "language_optical_layer_indexes": list(
                replacement.language_optical_layer_indexes
            ),
        },
        "vision_optical": vision,
        "language_optical": language,
        "retrieval_readout": readout.specification(),
        "trainable_parameters": count,
        "trainable_tensors": len(names),
        "trainable_parameter_list": names,
    }


def resolve_cached_model_source(model_id: str, cache_dir: Path | None) -> str:
    if Path(model_id).is_dir() or "/" not in model_id:
        return model_id
    roots: list[Path] = []
    if cache_dir is not None:
        roots.append(Path(cache_dir))
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    # Cluster accounts commonly keep large caches beside the repository rather
    # than below the login HOME (for example /DATA/.../user/.cache). Search the
    # working-directory ancestry so those snapshots remain reusable without
    # hard-coding a server-specific absolute path in experiment configs.
    working_directory = Path.cwd().resolve()
    for parent in (working_directory, *working_directory.parents):
        roots.append(parent / ".cache" / "huggingface" / "hub")
    repository = "models--" + model_id.replace("/", "--")
    for root in dict.fromkeys(path.resolve() for path in roots):
        directory = root / repository
        snapshots = directory / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates: list[Path] = []
        main_ref = directory / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots / revision)
        candidates.extend(
            sorted(
                (path for path in snapshots.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        for snapshot in candidates:
            if (snapshot / "config.json").is_file() and (
                snapshot / "preprocessor_config.json"
            ).is_file():
                return str(snapshot.resolve())
    return model_id


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name.lower() not in values:
        raise ValueError(f"Unsupported model dtype {name!r}")
    return values[name.lower()]
