"""Text-to-image inference for the compact one-step latent student."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .distilled_decoder import DistilledDecoderConfig, QwenOneStepLatentStudent
from .feature_cache import _encode_qwen


def _load_student(
    checkpoint: Path, device: torch.device
) -> tuple[QwenOneStepLatentStudent, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = DistilledDecoderConfig(**payload["config"])
    model = QwenOneStepLatentStudent(int(payload["text_dim"]), config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    array = ((value.detach().float().clamp(-1, 1) + 1) * 127.5).byte()
    array = array.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


@torch.inference_mode()
def generate(
    prompts: list[str],
    seeds: list[int],
    student_checkpoint: Path,
    qwen_checkpoint: Path,
    turbo_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    student, payload = _load_student(student_checkpoint, device)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    qwen_features = _encode_qwen(qwen, processor, prompts, device).to(device)
    del qwen, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        turbo_checkpoint,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    rows: list[list[Image.Image]] = []
    records = []
    times = []
    for seed in seeds:
        noise = torch.cat([
            torch.randn(
                (1, 4, 64, 64),
                generator=torch.Generator(device=device).manual_seed(seed + index),
                device=device,
            )
            for index in range(len(prompts))
        ])
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        latent = student(noise, qwen_features)
        decoded = vae.decode(
            latent.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
        )[0]
        images = [_tensor_to_pil(image) for image in decoded]
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        rows.append(images)
        for prompt_index, (prompt, image) in enumerate(zip(prompts, images)):
            filename = f"prompt_{prompt_index:02d}_seed_{seed}.png"
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
        "variant": payload["variant"],
        "architecture": payload["architecture"],
        "prompts": prompts,
        "seeds": seeds,
        "seconds_per_batch": times,
        "inference_iterations": 1,
        "uses_pca": False,
        "uses_sd_turbo_unet": False,
        "inference_flow": "text -> frozen Qwen -> compact one-step latent student -> VAE decode",
        "images": records,
    }
    (output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del student, vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with the compact one-step latent student")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = generate(
        args.prompt,
        args.seed or [42],
        args.student_checkpoint.expanduser().resolve(),
        args.qwen_checkpoint.expanduser().resolve(),
        args.turbo_checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        torch.device(args.device),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_load_student", "generate"]
