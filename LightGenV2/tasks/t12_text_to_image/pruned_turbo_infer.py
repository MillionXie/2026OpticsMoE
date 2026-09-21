"""One-call text-to-image inference with the structurally pruned Turbo UNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .compact_turbo import one_step_denoise
from .compact_turbo_infer import _to_pil
from .electronic_turbo_infer import _load_adapter
from .feature_cache import _encode_qwen
from .pruned_turbo import load_pruned_unet


@torch.inference_mode()
def generate(
    *,
    prompts: list[str],
    seeds: list[int],
    adapter_checkpoint: Path,
    pruned_unet: Path,
    qwen_checkpoint: Path,
    turbo_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    adapter, adapter_payload = _load_adapter(adapter_checkpoint, device)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    features = _encode_qwen(qwen, processor, prompts, device).to(device)
    condition = adapter.condition(features)
    del qwen, processor, features, adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from diffusers import AutoencoderKL, EulerDiscreteScheduler

    unet, manifest = load_pruned_unet(pruned_unet, device)
    unet.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(
        turbo_checkpoint, subfolder="vae", variant="fp16", torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1, device=device); sigma = scheduler.sigmas[0]
    condition = condition.to(dtype=unet.dtype)
    rows: list[list[Image.Image]] = []
    records = []
    for seed in seeds:
        noise = torch.cat([
            torch.randn(
                (1, 4, 64, 64),
                generator=torch.Generator(device=device).manual_seed(seed + index),
                device=device, dtype=unet.dtype,
            )
            for index in range(len(prompts))
        ])
        latent = one_step_denoise(unet, noise, condition, sigma)
        decoded = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        images = [_to_pil(image) for image in decoded]
        rows.append(images)
        for index, (prompt, image) in enumerate(zip(prompts, images)):
            filename = f"prompt_{index:02d}_seed_{seed}.png"
            image.save(output_dir / filename)
            records.append({"prompt": prompt, "seed": seed, "file": filename})
    cell, header, label = 512, 42, 110
    canvas = Image.new("RGB", (label + cell * len(prompts), header + cell * len(seeds)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, prompt in enumerate(prompts):
        draw.text((label + column * cell + 4, 4), prompt[:64], fill="black")
    for row, (seed, images) in enumerate(zip(seeds, rows)):
        top = header + row * cell
        draw.text((4, top + cell // 2), f"seed {seed}", fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * cell, top))
    canvas.save(output_dir / "grid.png")
    report = {
        "schema_version": 1,
        "variant": manifest["variant"],
        "prompts": prompts,
        "seeds": seeds,
        "random_seed_is_real": True,
        "inference_flow": "text -> frozen Qwen -> adapter -> Gaussian latent + one pruned UNet call -> one VAE decode",
        "unet_calls": 1,
        "vae_calls": 1,
        "iterative_loop": False,
        "unet_parameters": manifest["pruning"]["remaining_parameters"],
        "removed_parameters": manifest["pruning"]["removed_parameters"],
        "adapter_epoch": adapter_payload["epoch"],
        "images": records,
    }
    (output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    del unet, vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with the pruned one-step Turbo UNet")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--pruned-unet", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = generate(
        prompts=args.prompt, seeds=args.seed or [42],
        adapter_checkpoint=args.adapter_checkpoint.expanduser().resolve(),
        pruned_unet=args.pruned_unet.expanduser().resolve(),
        qwen_checkpoint=args.qwen_checkpoint.expanduser().resolve(),
        turbo_checkpoint=args.turbo_checkpoint.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(), device=torch.device(args.device),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate"]
