"""Caption plus real Gaussian seed inference for the direct RGB generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .cleanrender_gan import CleanRenderGANConfig, CleanRenderGenerator
from .feature_cache import _encode_qwen


def _images(value: torch.Tensor) -> list[Image.Image]:
    array = value.float().add(1).mul(127.5).clamp(0, 255).byte().cpu()
    return [Image.fromarray(item.permute(1, 2, 0).numpy(), mode="RGB") for item in array]


@torch.inference_mode()
def generate(
    prompts: list[str],
    seeds: list[int],
    checkpoint: Path,
    qwen_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(payload["config"])
    config_values["generator_channels"] = tuple(config_values["generator_channels"])
    config = CleanRenderGANConfig(**config_values)
    model = CleanRenderGenerator(config, len(payload["categories"])).to(device).eval()
    model.load_state_dict(payload["generator_ema"])

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    text = _encode_qwen(qwen, processor, prompts, device).to(device)
    del qwen, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows: list[tuple[int, list[Image.Image]]] = []
    records = []
    for seed in seeds:
        shared_noise = model.sample_noise(1, device, seed).expand(len(prompts), -1)
        images = _images(model(text, shared_noise))
        rows.append((seed, images))
        for index, (prompt, image) in enumerate(zip(prompts, images)):
            filename = f"prompt_{index:02d}_seed_{seed}.png"
            image.save(output_dir / filename)
            records.append({"prompt": prompt, "seed": seed, "file": filename})
    size, label, header = config.image_size, 82, 24
    canvas = Image.new("RGB", (label + size * len(prompts), header + size * len(seeds)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, prompt in enumerate(prompts):
        draw.text((label + column * size + 3, 4), prompt[:20], fill="black")
    for row, (seed, images) in enumerate(rows):
        top = header + row * size
        draw.text((4, top + size // 2), f"seed {seed}", fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * size, top))
    canvas.save(output_dir / "grid.png")
    report = {
        "schema_version": 1,
        "variant": payload["variant"],
        "checkpoint_epoch": payload["epoch"],
        "prompts": prompts,
        "seeds": seeds,
        "random_seed_is_real": True,
        "same_seed_uses_same_gaussian_vector_across_prompts": True,
        "input_image_used": False,
        "inference_flow": "text -> frozen Qwen -> trained projection + seeded Gaussian -> one direct RGB decoder",
        "inference_iterations": 1,
        "images": records,
    }
    (output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer with the compact CleanRender generator")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = generate(
        args.prompt, args.seed or [11, 29, 47, 83],
        args.checkpoint.expanduser().resolve(), args.qwen_checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(), torch.device(args.device),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
