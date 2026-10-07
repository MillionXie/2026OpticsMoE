"""Fixed-test quality evaluation for one trained T12 comparison row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import CachedLatentDataset, read_manifest
from .feature_cache import _load_image, _model_source
from .modeling import build_model
from .settings import Settings


CLIP_EVALUATOR = "openai/clip-vit-base-patch32"


def _uint8(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().add(1).mul(127.5).clamp(0, 255).byte()


def _pil_batch(value: torch.Tensor) -> list[Image.Image]:
    return [Image.fromarray(item.permute(1, 2, 0).cpu().numpy(), mode="RGB") for item in value]


@torch.inference_mode()
def evaluate(
    settings: Settings,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    *,
    seed: int = 1234,
) -> dict[str, Any]:
    """Evaluate prior samples, not posterior reconstructions, on the sealed test prompts."""

    from diffusers import AutoencoderKL
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from transformers import AutoProcessor, CLIPModel

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != settings.variant:
        raise ValueError("Checkpoint/profile variant mismatch")
    model = build_model(settings, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)

    vae_source, vae_local = _model_source(settings.vae_checkpoint, settings.vae_model)
    vae = AutoencoderKL.from_pretrained(vae_source, local_files_only=vae_local).to(device).eval().requires_grad_(False)
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    clip_processor = AutoProcessor.from_pretrained(CLIP_EVALUATOR)
    clip = CLIPModel.from_pretrained(CLIP_EVALUATOR).to(device).eval().requires_grad_(False)

    dataset = CachedLatentDataset(settings.cache_dir / "test.pt")
    rows = read_manifest(settings.data_dir / "test.jsonl")
    if [row.sample_id for row in rows] != dataset.payload["sample_ids"]:
        raise ValueError("Test manifest/cache order mismatch")
    by_id = {row.sample_id: row for row in rows}
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False, num_workers=0)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, len(dataset)), normalize=False).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    categories = sorted({row.category for row in rows})
    category_inputs = clip_processor(
        text=[f"a studio photo of a {category}" for category in categories],
        padding=True, return_tensors="pt",
    )
    category_features = clip.get_text_features(**{
        key: value.to(device) for key, value in category_inputs.items() if torch.is_tensor(value)
    })
    category_features = torch.nn.functional.normalize(category_features.float(), dim=-1)

    clip_sum = diversity_sum = 0.0
    category_correct = count = 0
    saved = 0
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for batch_index, batch in enumerate(loader):
        ids = list(batch["sample_id"])
        text = batch["text"].to(device)
        first_latent = model.generate(text, seed=seed + batch_index)
        second_latent = model.generate(text, seed=seed + 100_000 + batch_index)
        first = vae.decode(first_latent / scale).sample.clamp(-1, 1)
        second = vae.decode(second_latent / scale).sample.clamp(-1, 1)
        real = torch.stack([_load_image(by_id[sample_id].image_path, settings.image_size) for sample_id in ids]).to(device)
        real_u8, fake_u8 = _uint8(real), _uint8(first)
        fid.update(real_u8, real=True)
        fid.update(fake_u8, real=False)
        kid.update(real_u8, real=True)
        kid.update(fake_u8, real=False)
        diversity_sum += float(lpips(first, second)) * len(ids)

        captions = [by_id[sample_id].caption for sample_id in ids]
        clip_inputs = clip_processor(text=captions, images=_pil_batch(fake_u8), padding=True, return_tensors="pt")
        clip_inputs = {key: value.to(device) for key, value in clip_inputs.items() if torch.is_tensor(value)}
        image_features = torch.nn.functional.normalize(
            clip.get_image_features(pixel_values=clip_inputs["pixel_values"]).float(), dim=-1
        )
        text_features = torch.nn.functional.normalize(
            clip.get_text_features(
                input_ids=clip_inputs["input_ids"], attention_mask=clip_inputs["attention_mask"]
            ).float(), dim=-1
        )
        clip_sum += float((image_features * text_features).sum(-1).sum())
        predicted = (image_features @ category_features.T).argmax(-1).cpu().tolist()
        category_correct += sum(categories[index] == by_id[sample_id].category for index, sample_id in zip(predicted, ids))
        count += len(ids)
        for sample_id, image in zip(ids, _pil_batch(fake_u8)):
            if saved >= 64:
                break
            image.save(sample_dir / f"{saved:03d}_{sample_id}.png")
            saved += 1

    kid_mean, kid_std = kid.compute()
    report = {
        "schema_version": 1,
        "variant": settings.variant,
        "checkpoint": str(checkpoint.resolve()),
        "test_samples": count,
        "sampling": "one prior N(0,I) style sample per fixed test caption",
        "seed": seed,
        "fid": float(fid.compute()),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "clip_cosine": clip_sum / count,
        "zero_shot_category_accuracy": category_correct / count,
        "lpips_seed_diversity": diversity_sum / count,
        "evaluator": CLIP_EVALUATOR,
        "vae": vae_source,
        "saved_samples": saved,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["CLIP_EVALUATOR", "evaluate"]
