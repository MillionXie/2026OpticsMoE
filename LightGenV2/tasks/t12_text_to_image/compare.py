"""Generate a matched visual comparison for LightGen and Qwen+VAE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw

from .feature_cache import _encode_qwen, _model_source
from .modeling import build_model
from .settings import Settings
from .settings import TASK_DIR, load_settings


DEFAULT_PROMPTS = (
    "a red ceramic mug with a curved handle on a light gray background",
    "a transparent bottle with a blue cap on a white background",
    "a black running shoe with a white sole on a pale blue background",
    "a green backpack with two front pockets on a beige background",
    "a yellow banana on a light gray background",
    "a small red toy car on a white background",
)


def _load_generator(settings: Settings, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != settings.variant:
        raise ValueError(f"Checkpoint variant {payload.get('variant')} != {settings.variant}")
    model = build_model(settings, device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)


def _to_images(value: torch.Tensor) -> list[Image.Image]:
    array = value.detach().float().add(1).mul(127.5).clamp(0, 255).byte().cpu()
    return [Image.fromarray(item.permute(1, 2, 0).numpy(), mode="RGB") for item in array]


def _grid(rows: list[tuple[str, Sequence[Image.Image]]], prompts: Sequence[str], path: Path) -> None:
    cell = rows[0][1][0].width
    label_width, header = 170, 54
    canvas = Image.new("RGB", (label_width + cell * len(prompts), header * len(rows) + cell * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, prompt in enumerate(prompts):
        draw.text((label_width + column * cell + 4, 4), prompt[:34], fill="black")
    for row_index, (name, images) in enumerate(rows):
        top = header + row_index * (cell + header)
        draw.text((8, top + cell // 2), name, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label_width + column * cell, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.inference_mode()
def compare(
    lightgen_settings: Settings,
    baseline_settings: Settings,
    lightgen_checkpoint: Path,
    baseline_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    seed: int = 42,
) -> dict[str, Any]:
    if lightgen_settings.qwen_model != baseline_settings.qwen_model or lightgen_settings.vae_model != baseline_settings.vae_model:
        raise ValueError("Comparison rows must share the exact Qwen and VAE identities")
    qwen_source, qwen_local = _model_source(lightgen_settings.qwen_checkpoint, lightgen_settings.qwen_model)
    vae_source, vae_local = _model_source(lightgen_settings.vae_checkpoint, lightgen_settings.vae_model)
    from diffusers import AutoencoderKL
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_source, local_files_only=qwen_local)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_source, local_files_only=qwen_local,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(vae_source, local_files_only=vae_local).to(device).eval().requires_grad_(False)
    text = _encode_qwen(qwen, processor, prompts, device).to(device)
    lightgen = _load_generator(lightgen_settings, lightgen_checkpoint, device)
    baseline = _load_generator(baseline_settings, baseline_checkpoint, device)
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    lightgen_rgb = vae.decode(lightgen.generate(text, seed=seed) / scale).sample
    baseline_rgb = vae.decode(baseline.generate(text, seed=seed) / scale).sample
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "matched_generation_grid.png"
    _grid(
        [("LightGen parallel", _to_images(lightgen_rgb)), ("Qwen + VAE baseline", _to_images(baseline_rgb))],
        prompts, grid_path,
    )
    report = {
        "schema_version": 1,
        "seed": seed,
        "prompts": list(prompts),
        "qwen": qwen_source,
        "vae": vae_source,
        "lightgen_checkpoint": str(lightgen_checkpoint.resolve()),
        "baseline_checkpoint": str(baseline_checkpoint.resolve()),
        "grid": str(grid_path.resolve()),
        "fairness": "same prompts, Qwen features, VAE decoder, style seeds and latent shape",
        "metrics_status": "visual grid only; FID/KID/CLIPScore require the fixed test run",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["DEFAULT_PROMPTS", "compare"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched T12 LightGen/baseline generation grid")
    parser.add_argument("--lightgen-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-json", type=Path, default=None)
    args = parser.parse_args()
    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        prompts = tuple(json.loads(args.prompts_json.read_text(encoding="utf-8")))
    report = compare(
        load_settings(TASK_DIR / "configs/lightgen_parallel.yaml"),
        load_settings(TASK_DIR / "configs/qwen_vae_baseline.yaml"),
        args.lightgen_checkpoint.resolve(), args.baseline_checkpoint.resolve(),
        args.output_dir.resolve(), torch.device(args.device), prompts, args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
