from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from .datasets import make_indexed_loader
from .io_utils import write_json
from .modeling import LoadedBackbone
from .optics import VisionOpticalStackSurrogate, locate_visual
from .visualization import save_field, save_input_image


def build_source_encoder(loaded: LoadedBackbone, settings: Any) -> VisionOpticalStackSurrogate:
    hidden_size = int(loaded.model.config.vision_config.hidden_size)
    encoder = VisionOpticalStackSurrogate(
        hidden_size=hidden_size, optical_dim=settings.optical_dim,
        conversions=settings.optical_conversions_per_stack, field_size=settings.optical_field_size,
        padding_size=settings.optical_padding_size, wavelength_nm=settings.wavelength_nm,
        pixel_pitch_um=settings.pixel_pitch_um, distance_cm=settings.mask_distance_cm,
        amplitude_mask_enabled=settings.amplitude_mask_enabled, phase_init=settings.phase_init,
        phase_init_std=settings.phase_init_std, residual_enabled=True,
        identity_scale_init=1.0, modulated_scale_init=0.1,
        identity_scale_trainable=False, modulated_scale_trainable=True,
    ).to(loaded.device)
    _verify_source_config(settings, hidden_size)
    _load_adapter_weights(encoder, settings.source_vision_checkpoint)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _verify_source_config(settings: Any, hidden_size: int) -> None:
    path = settings.source_experiment_dir / "config_resolved.json"
    if not path.is_file():
        warnings.warn(f"Source config not found at {path}; pixel-budget compatibility cannot be verified")
        return
    source = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "optical_dim": settings.optical_dim,
        "optical_field_size": settings.optical_field_size,
        "vision_hidden_size": hidden_size,
    }
    mismatched = {key: (source.get(key), value) for key, value in expected.items() if source.get(key) != value}
    if mismatched:
        raise RuntimeError(
            f"Source experiment is incompatible with this probe: {mismatched}. "
            "Use the same processor pixel budget and token64 architecture as the source checkpoint."
        )


def _load_adapter_weights(encoder: VisionOpticalStackSurrogate, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Source vision checkpoint not found: {checkpoint}. Run the source student training first."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported source checkpoint payload: {type(state).__name__}")
    for module_name in ("input_adapter", "adapter_norm"):
        module = getattr(encoder, module_name)
        selected: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            marker = f"{module_name}."
            position = key.rfind(marker)
            if position >= 0:
                selected[key[position + len(marker):]] = value
        missing = set(module.state_dict()) - set(selected)
        if missing:
            raise RuntimeError(f"Source checkpoint lacks {module_name} tensors: {sorted(missing)}")
        module.load_state_dict(selected, strict=True)


@torch.no_grad()
def vision_block_input_groups(model: torch.nn.Module, pixel_values: torch.Tensor,
                              image_grid_thw: torch.Tensor) -> tuple[list[torch.Tensor], list[int]]:
    """Execute Qwen patch embedding and positional encoding, stopping before visual.blocks[0]."""
    visual = locate_visual(model)
    parameter = next(visual.parameters())
    hidden = visual.patch_embed(pixel_values.to(device=parameter.device, dtype=parameter.dtype))
    hidden = hidden + visual.fast_pos_embed_interpolate(image_grid_thw.to(parameter.device))
    counts = image_grid_thw.long().prod(dim=1).detach().cpu().tolist()
    if any(count <= 0 for count in counts) or sum(counts) != hidden.shape[0]:
        raise RuntimeError(
            f"image_grid_thw token counts {counts} do not match packed patch embedding {tuple(hidden.shape)}"
        )
    return list(hidden.split(counts, dim=0)), [int(value) for value in counts]


def _image_inputs(processor: Any, images: Sequence[Image.Image], device: torch.device) -> dict[str, torch.Tensor]:
    image_processor = getattr(processor, "image_processor", processor)
    values = image_processor(images=list(images), return_tensors="pt")
    required = ("pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Qwen image processor did not return: {', '.join(missing)}")
    return {name: values[name].to(device, non_blocking=True) for name in required}


@torch.no_grad()
def extract_split(loaded: LoadedBackbone, encoder: VisionOpticalStackSurrogate, dataset: Any,
                  split: str, settings: Any, class_names: Sequence[str]) -> dict[str, Any]:
    loader = make_indexed_loader(dataset, settings.feature_batch_size, settings.num_workers,
                                 False, settings.seed)
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    grid_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    visualized = 0
    started = time.perf_counter()
    for batch_index, (images, labels, indices) in enumerate(loader, 1):
        inputs = _image_inputs(loaded.processor, images, loaded.device)
        groups, counts = vision_block_input_groups(loaded.model, inputs["pixel_values"], inputs["image_grid_thw"])
        fields = encoder.encode_groups_to_input_fields(groups)
        if torch.any(fields < 0):
            warnings.warn("Vision optical input field contains negative values after Softplus")
        features = fields.flatten(1).detach().cpu()
        feature_chunks.append(features)
        label_chunks.append(labels.cpu()); index_chunks.append(indices.cpu())
        grid_chunks.append(inputs["image_grid_thw"].detach().cpu().long())
        count_chunks.append(torch.tensor(counts, dtype=torch.long))
        if settings.save_feature_visualizations and visualized < settings.feature_visualization_sample_count:
            for offset, image in enumerate(images):
                if visualized >= settings.feature_visualization_sample_count:
                    break
                _save_feature_example(settings, split, int(indices[offset]), image, fields[offset],
                                      int(labels[offset]), class_names, inputs["image_grid_thw"][offset], counts[offset])
                visualized += 1
        if settings.progress and (batch_index % 100 == 0 or batch_index == len(loader)):
            print(f"[extract_features] {split} batch={batch_index}/{len(loader)} samples={sum(len(x) for x in label_chunks)}/{len(dataset)}", flush=True)
    features = torch.cat(feature_chunks)
    labels = torch.cat(label_chunks).long(); sample_indices = torch.cat(index_chunks).long()
    grids = torch.cat(grid_chunks).long(); token_counts = torch.cat(count_chunks).long()
    stored_features = features.half() if settings.cache_dtype == "float16" else features.float()
    metadata = {
        "cache_schema_version": 1, "dataset": settings.dataset, "split": split,
        "feature_type": "vision_optical_input_field", "feature_dim": settings.optical_field_size ** 2,
        "field_shape": [settings.optical_field_size, settings.optical_field_size],
        "sample_count": len(labels), "class_names": list(class_names), "model_id": settings.model_id,
        "source_experiment_dir": str(settings.source_experiment_dir),
        "source_vision_checkpoint": str(settings.source_vision_checkpoint),
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "classification_prompt": settings.classification_prompt,
        "prompt_used_for_extraction": False,
        "vision_hidden_size": int(loaded.model.config.vision_config.hidden_size),
        "executed_modules": ["qwen_image_processor", "vision.patch_embed", "vision_position_embedding", "vision_input_adapter", "vision_adapter_norm", "softplus", "zero_padding"],
        "skipped_modules": ["optical_conversions", "vision_transformer_blocks", "vision_merger", "language_decoder", "language_optical_surrogate", "final_rmsnorm"],
    }
    payload = {
        "features": stored_features, "labels": labels, "sample_indices": sample_indices,
        "image_grid_thw": grids, "visual_token_counts": token_counts,
        "class_names": list(class_names), "metadata": metadata,
    }
    cache_path = settings.output_dir / "features" / f"{split}_vision_input_field.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(cache_path)
    elapsed = time.perf_counter() - started
    report = {
        **metadata, "cache_path": str(cache_path), "cache_dtype": settings.cache_dtype,
        "elapsed_sec": elapsed, "samples_per_sec": len(labels) / elapsed if elapsed else 0.0,
        "visual_token_count_min": int(token_counts.min()), "visual_token_count_max": int(token_counts.max()),
        "visual_token_count_mean": float(token_counts.float().mean()),
        "feature_min": float(features.min()), "feature_max": float(features.max()),
        "feature_mean": float(features.mean()), "feature_std": float(features.std(unbiased=False)),
        "feature_negative_count": int((features < 0).sum()),
    }
    write_json(settings.output_dir / "metrics" / f"feature_extraction_{split}.json", report)
    return report


def _save_feature_example(settings: Any, split: str, index: int, image: Image.Image,
                          field: torch.Tensor, label: int, class_names: Sequence[str],
                          grid: torch.Tensor, token_count: int) -> None:
    root = settings.output_dir / "figures" / "vision_input_fields"
    stem = f"{split}_sample_{index:06d}"
    save_input_image(image, root / f"{stem}_input.png")
    save_field(field, root / f"{stem}_vision_input_field.png", f"{split} sample {index}: vision optical input field")
    torch.save(field.detach().cpu(), root / f"{stem}_vision_input_field.pt")
    value = field.detach().float().cpu()
    write_json(root / f"{stem}_metadata.json", {
        "split": split, "sample_index": index, "true_label": label, "true_name": class_names[label],
        "image_grid_thw": grid.detach().cpu().long().tolist(), "visual_token_count": token_count,
        "feature_shape": list(value.shape), "feature_min": float(value.min()),
        "feature_max": float(value.max()), "feature_mean": float(value.mean()),
        "feature_std": float(value.std(unbiased=False)),
        "feature_sparsity_abs_lt_1e-6": float((value.abs() < 1e-6).float().mean()),
    })


def load_feature_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Vision-field feature cache not found: {path}; run --phase extract_features")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"features", "labels", "sample_indices", "image_grid_thw", "visual_token_counts", "class_names", "metadata"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Feature cache is missing fields: {sorted(missing)}")
    if payload["features"].ndim != 2 or payload["features"].shape[1] != 4096:
        raise RuntimeError(f"Expected cached [N,4096] features, got {tuple(payload['features'].shape)}")
    return payload


def source_parameter_report(encoder: VisionOpticalStackSurrogate) -> dict[str, int]:
    input_parameters = sum(p.numel() for p in encoder.input_adapter.parameters())
    norm_parameters = sum(p.numel() for p in encoder.adapter_norm.parameters())
    return {
        "source_vision_input_adapter_parameters": input_parameters,
        "source_vision_adapter_norm_parameters": norm_parameters,
        "source_frozen_encoder_parameters": input_parameters + norm_parameters,
        "source_trainable_parameters_during_probe": 0,
    }
