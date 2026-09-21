"""One-pass pure generation and reference variation for the chair VAE-GAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from .feature_cache import _encode_qwen
from .cleanrender_gan import CleanRenderGenerator
from .cleanrender_vae import CleanRenderImageEncoder, load_cleanrender_vae_config
from .settings import TASK_DIR


def _load_reference(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as handle:
        image = ImageOps.fit(
            ImageOps.exif_transpose(handle).convert("RGB"), (size, size),
            method=Image.Resampling.LANCZOS,
        )
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)


def _to_pil(image: torch.Tensor) -> Image.Image:
    value = image.float().add(1).mul(127.5).clamp(0, 255).byte().cpu()
    return Image.fromarray(value.permute(1, 2, 0).numpy(), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate chairs from text or vary a reference chair")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--variation-strength", type=float, default=None)
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/qwen_cleanrender_chair_vae_gan.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if len(args.prompt) != 1:
        raise ValueError("Use exactly one chair prompt; vary it with multiple --seed values")
    seeds = args.seed or [11, 29, 47, 83]
    config = load_cleanrender_vae_config(args.config.expanduser().resolve())
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    decoder = CleanRenderGenerator(config.base, categories=1).to(device).eval()
    decoder.load_state_dict(payload["decoder_ema"])
    encoder = None
    if args.reference_image is not None:
        encoder = CleanRenderImageEncoder(config).to(device).eval()
        encoder.load_state_dict(payload["encoder_ema"])
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    qwen_checkpoint = args.qwen_checkpoint.expanduser().resolve()
    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    text = _encode_qwen(qwen, processor, args.prompt, device).to(device)
    del qwen, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    text = text.repeat(len(seeds), 1)
    with torch.inference_mode():
        if encoder is None:
            latent = torch.cat([
                decoder.sample_noise(1, device, seed) for seed in seeds
            ])
            mode = "pure_text_to_image"
            strength = None
        else:
            reference = _load_reference(args.reference_image.expanduser().resolve(), config.base.image_size, device)
            mean, _ = encoder(reference)
            strength = config.reference_variation_strength if args.variation_strength is None else args.variation_strength
            latent = torch.cat([encoder.seeded_variation(mean, strength, seed) for seed in seeds])
            mode = "reference_variation"
        images = decoder(text, latent)
    size, header = config.base.image_size, 32
    canvas = Image.new("RGB", (size * len(seeds), size + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (seed, image) in enumerate(zip(seeds, images)):
        draw.text((column * size + 4, 8), f"seed {seed}", fill="black")
        canvas.paste(_to_pil(image), (column * size, header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    metadata = {
        "mode": mode,
        "prompt": args.prompt[0],
        "seeds": seeds,
        "reference_image": None if args.reference_image is None else str(args.reference_image),
        "variation_strength": strength,
        "decoder_calls": 1,
        "iterative_sampling": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
