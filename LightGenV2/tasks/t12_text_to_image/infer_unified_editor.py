"""Single-pass full-frame inference for a packaged unified latent editor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from .electronic_turbo_infer import _load_adapter
from .feature_cache import _qwen_prompts
from .half_qwen import load_half_qwen_text_encoder
from .product_repair_model import RepairModelConfig, one_step_edit
from .progressive_student import build_narrow_optical_unet
from .qwen_mini_small import QwenMiniConfig, QwenMiniTextEncoder


@torch.inference_mode()
def infer(*, checkpoint: Path, qwen_checkpoint: Path, initial_unet: Path,
          vae_checkpoint: Path, adapter_checkpoint: Path, input_image: Path,
          prompt: str, output: Path, seed: int, device: torch.device) -> dict:
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = UNet2DConditionModel.load_config(initial_unet, subfolder="unet", local_files_only=True)
    unet, optical, _ = build_narrow_optical_unet(
        config, payload["student_widths"], RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet = unet.to(device).eval()
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    qwen, processor, _ = load_half_qwen_text_encoder(qwen_checkpoint, device, keep_layers=1)
    encoded = _qwen_prompts(processor, [prompt])
    ids = encoded["input_ids"][:, -64:].to(device)
    mask = encoded["attention_mask"][:, -64:].to(device).bool()
    embeddings = qwen.embed_tokens(ids)
    frontend = payload["text_frontend"]
    text = QwenMiniTextEncoder(QwenMiniConfig(**frontend["config"]))
    text.load_state_dict(frontend["text"])
    text = text.to(device).eval()
    bridge = nn.Linear(text.config.width, 2048)
    bridge.load_state_dict(frontend["bridge"])
    bridge = bridge.to(device).eval()
    with Image.open(input_image) as handle:
        image = ImageOps.fit(ImageOps.exif_transpose(handle).convert("RGB"), (256,256),
                              method=Image.Resampling.LANCZOS)
    reference = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().div(127.5).sub(1)
    reference = reference.unsqueeze(0).to(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_checkpoint, subfolder="vae", variant="fp16",
                                       torch_dtype=dtype, local_files_only=True).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(
        vae_checkpoint, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    sigma = scheduler.sigmas[0]
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pooled = bridge(text.hidden(embeddings.float(), mask).float())
        condition = adapter.condition(pooled)
        latent = vae.encode(reference.to(dtype)).latent_dist.mode() * vae.config.scaling_factor
        noise = torch.randn(latent.shape, generator=noise_generator, device=device, dtype=latent.dtype)
        training = payload["training_config"]
        edited = one_step_edit(unet, noise, latent.float(), condition, sigma,
                               noise_scale=float(training["noise_scale"]),
                               residual_scale=float(training["residual_scale"]))
        generated = vae.decode(edited.to(dtype)/vae.config.scaling_factor, return_dict=False)[0]
    array = generated[0].float().clamp(-1,1).add(1).mul(127.5).byte().permute(1,2,0).cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
    result = {"checkpoint": str(checkpoint), "input": str(input_image), "prompt": prompt,
              "seed": seed, "output": str(output), "counted_parameters": payload["counted_parameters"],
              "generator_calls": 1, "optical_alpha": float(optical.fusion.alpha),
              "hard_pixel_composite": False}
    output.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint","qwen-checkpoint","initial-unet","vae-checkpoint",
                 "adapter-checkpoint","input-image","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--prompt",required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    print(json.dumps(infer(checkpoint=args.checkpoint.resolve(),qwen_checkpoint=args.qwen_checkpoint.resolve(),
                           initial_unet=args.initial_unet.resolve(),vae_checkpoint=args.vae_checkpoint.resolve(),
                           adapter_checkpoint=args.adapter_checkpoint.resolve(),input_image=args.input_image.resolve(),
                           prompt=args.prompt,output=args.output.resolve(),seed=args.seed,
                           device=torch.device(args.device)),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
