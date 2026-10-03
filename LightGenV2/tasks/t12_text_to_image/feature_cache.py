"""Cache frozen Qwen text features and frozen VAE image latents once."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageOps

from .dataset import read_manifest, sha256, validate_split_contract
from .settings import Settings


def _model_source(checkpoint: Path | None, model_id: str) -> tuple[str, bool]:
    return (str(checkpoint), True) if checkpoint is not None else (model_id, False)


def _load_image(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    value = torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float()
    return value.div(127.5).sub(1.0)


def _qwen_prompts(processor: Any, captions: Sequence[str]) -> dict[str, torch.Tensor]:
    prompts = []
    for caption in captions:
        messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    values = processor(text=prompts, padding=True, return_tensors="pt")
    return {
        key: value for key, value in values.items()
        if torch.is_tensor(value) and key not in {"token_type_ids", "mm_token_type_ids"}
    }


def _deduplicate_captions(captions: Sequence[str]) -> tuple[list[str], list[int]]:
    """Return first-seen unique captions and an index that restores row order."""

    unique: list[str] = []
    lookup: dict[str, int] = {}
    inverse: list[int] = []
    for caption in captions:
        index = lookup.get(caption)
        if index is None:
            index = len(unique)
            lookup[caption] = index
            unique.append(caption)
        inverse.append(index)
    return unique, inverse


@torch.inference_mode()
def _encode_qwen(model: Any, processor: Any, captions: Sequence[str], device: torch.device) -> torch.Tensor:
    inputs = {key: value.to(device) for key, value in _qwen_prompts(processor, captions).items()}
    core = getattr(model, "model", model)
    outputs = core(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
    hidden = outputs.last_hidden_state.float()
    mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
    return ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()


def _encode_caption_rows(
    model: Any,
    processor: Any,
    captions: Sequence[str],
    device: torch.device,
    batch_size: int,
    split: str,
) -> tuple[torch.Tensor, int]:
    """Encode repeated per-frame captions once, then restore sample order."""

    unique, inverse = _deduplicate_captions(captions)
    chunks = []
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        chunks.append(_encode_qwen(model, processor, batch, device))
        print(
            f"[T12 cache] {split} text: {min(start + len(batch), len(unique))}/{len(unique)} unique captions",
            flush=True,
        )
    features = torch.cat(chunks)
    return features[torch.tensor(inverse, dtype=torch.long)], len(unique)


@torch.inference_mode()
def _encode_vae(vae: Any, images: torch.Tensor) -> torch.Tensor:
    distribution = vae.encode(images).latent_dist
    latent = distribution.mode()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    return latent * scaling


def build_feature_cache(settings: Settings, device: torch.device, *, force: bool = False) -> dict[str, Any]:
    """Build immutable train/val/test caches shared by both comparison rows."""

    contract = validate_split_contract(settings.data_dir)
    qwen_source, qwen_local = _model_source(settings.qwen_checkpoint, settings.qwen_model)
    vae_source, vae_local = _model_source(settings.vae_checkpoint, settings.vae_model)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    expected = [settings.cache_dir / f"{split}.pt" for split in ("train", "val", "test")]
    existing = [path.is_file() for path in expected]
    if all(existing) and not force:
        for split, path in zip(("train", "val", "test"), expected):
            cached = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            meta = cached.get("meta", {})
            if (
                meta.get("manifest_sha256") != contract["manifest_sha256"][split]
                or meta.get("qwen") != qwen_source
                or meta.get("vae") != vae_source
                or meta.get("image_size") != settings.image_size
            ):
                raise ValueError(f"Stale feature cache for {split}; use a new data directory or --force")
        return contract
    if any(existing) and not force:
        raise FileExistsError("Partial T12 feature cache exists; inspect it, then use --force or a new data directory")

    from diffusers import AutoencoderKL
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_source, local_files_only=qwen_local)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_source, local_files_only=qwen_local,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    splits = ("train", "val", "test")
    rows_by_split = {
        split: read_manifest(settings.data_dir / f"{split}.jsonl") for split in splits
    }
    text_by_split: dict[str, torch.Tensor] = {}
    unique_captions: dict[str, int] = {}
    for split in splits:
        rows = rows_by_split[split]
        text_by_split[split], unique_captions[split] = _encode_caption_rows(
            qwen, processor, [row.caption for row in rows], device, settings.batch_size, split
        )
    del qwen, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        vae_source, local_files_only=vae_local,
        torch_dtype=torch.float32,
    ).to(device).eval().requires_grad_(False)

    for split, path in zip(splits, expected):
        rows = rows_by_split[split]
        latent_chunks = []
        for start in range(0, len(rows), settings.batch_size):
            batch = rows[start : start + settings.batch_size]
            images = torch.stack([_load_image(row.image_path, settings.image_size) for row in batch]).to(device)
            latent_chunks.append(_encode_vae(vae, images).cpu())
            print(f"[T12 cache] {split} images: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
        payload = {
            "meta": {
                "schema_version": 1,
                "split": split,
                "manifest_sha256": sha256(settings.data_dir / f"{split}.jsonl"),
                "qwen": qwen_source,
                "qwen_pooling": "masked mean of final native hidden state",
                "unique_captions": unique_captions[split],
                "vae": vae_source,
                "vae_scaling_factor": float(getattr(vae.config, "scaling_factor", 1.0)),
                "image_size": settings.image_size,
                "latent_shape": [settings.latent_channels, settings.latent_size, settings.latent_size],
            },
            "sample_ids": [row.sample_id for row in rows],
            "categories": [row.category for row in rows],
            "text": text_by_split[split].to(torch.bfloat16),
            "latent": torch.cat(latent_chunks).to(torch.float16),
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        path.with_suffix(".json").write_text(json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8")
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return contract


__all__ = ["build_feature_cache"]
