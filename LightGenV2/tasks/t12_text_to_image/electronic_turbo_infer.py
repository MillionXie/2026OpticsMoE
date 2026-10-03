"""Generate images from arbitrary text and real Gaussian seeds with the Qwen Turbo baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .electronic_turbo import QwenTurboConditionAdapter, TurboAdapterConfig
from .feature_cache import _encode_qwen


def _load_adapter(checkpoint: Path, device: torch.device) -> tuple[QwenTurboConditionAdapter, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = TurboAdapterConfig(**payload["config"])
    state = payload["adapter"]
    report = payload["architecture"]
    adapter = QwenTurboConditionAdapter(
        int(payload["settings"]["text_dim"]), config,
        state["teacher_mean"], state["basis"], state["coefficient_mean"],
        state["coefficient_std"], int(report["condition_tokens"]), int(report["condition_dim"]),
    )
    adapter.load_state_dict(state)
    return adapter.to(device).eval(), payload


@torch.inference_mode()
def generate(
    prompts: list[str], seeds: list[int], adapter_checkpoint: Path,
    qwen_checkpoint: Path, turbo_checkpoint: Path, output_dir: Path, device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter, payload = _load_adapter(adapter_checkpoint, device)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    qwen_features = _encode_qwen(qwen, processor, prompts, device).to(device)
    condition = adapter.condition(qwen_features)
    del qwen, processor, qwen_features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        turbo_checkpoint, torch_dtype=torch.float16, variant="fp16", local_files_only=True,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    condition = condition.to(dtype=pipe.unet.dtype)
    all_images: list[list[Image.Image]] = []
    image_records = []
    for seed in seeds:
        generators = [torch.Generator(device=device).manual_seed(seed) for _ in prompts]
        images = pipe(
            prompt_embeds=condition, num_inference_steps=1, guidance_scale=0.0,
            height=512, width=512, generator=generators,
        ).images
        all_images.append(images)
        for index, (prompt, image) in enumerate(zip(prompts, images)):
            filename = f"prompt_{index:02d}_seed_{seed}.png"
            image.save(output_dir / filename)
            image_records.append({"prompt": prompt, "seed": seed, "file": filename})
    cell, header, label = 512, 42, 110
    canvas = Image.new("RGB", (label + cell * len(prompts), header + cell * len(seeds)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, prompt in enumerate(prompts):
        draw.text((label + column * cell + 4, 4), prompt[:64], fill="black")
    for row, (seed, images) in enumerate(zip(seeds, all_images)):
        top = header + row * cell
        draw.text((4, top + cell // 2), f"seed {seed}", fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * cell, top))
    canvas.save(output_dir / "grid.png")
    report = {
        "schema_version": 1,
        "variant": payload["variant"],
        "prompts": prompts,
        "seeds": seeds,
        "random_seed_is_real": True,
        "randomness": "Each integer initializes an independent N(0,I) diffusion latent; no input image is used.",
        "inference_flow": "text -> frozen Qwen -> trained adapter -> seeded Gaussian latent + one frozen SD-Turbo UNet call -> one frozen VAE decode",
        "inference_iterations": 1,
        "images": image_records,
    }
    (output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen-conditioned one-step text-to-image inference")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = generate(
        args.prompt, args.seed or [42], args.adapter_checkpoint.expanduser().resolve(),
        args.qwen_checkpoint.expanduser().resolve(), args.turbo_checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(), torch.device(args.device),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
