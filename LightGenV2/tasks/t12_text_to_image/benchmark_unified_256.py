"""First language block to RGB timing, with optical FFTs bypassed for hardware estimate.

This is a latency proxy, not a measurement on physical optical equipment.
The same 256-pixel reference and prompt are used for every variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .benchmark_three_task_bundle import (_BypassOptics, _image, _large,
                                          _small, _timed, PHYSICAL_OPTICAL_MS)
from .electronic_turbo_infer import _load_adapter
from .product_repair_model import RepairModelConfig, one_step_edit
from .progressive_student import build_narrow_optical_unet
from .qwen_mini_small import PromptEmbeddingLookup, QwenMiniConfig, QwenMiniTextEncoder


@torch.inference_mode()
def _unified_large(args, device):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(args.large_checkpoint, map_location="cpu", weights_only=False)
    base = UNet2DConditionModel.load_config(args.initial_unet, subfolder="unet",
                                            local_files_only=True)
    unet, optical, _ = build_narrow_optical_unet(
        base, payload["student_widths"], RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet = unet.to(device).eval()
    optical.optical = _BypassOptics().to(device)
    adapter, _ = _load_adapter(args.adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    vae = AutoencoderKL.from_pretrained(args.turbo, subfolder="vae", variant="fp16",
                                        torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.turbo, subfolder="scheduler",
                                                        local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    frontend = payload["text_frontend"]
    text = QwenMiniTextEncoder(QwenMiniConfig(**frontend["config"]))
    text.load_state_dict(frontend["text"])
    text = text.to(device).eval()
    bridge = nn.Linear(text.config.width, 2048)
    bridge.load_state_dict(frontend["bridge"])
    bridge = bridge.to(device).eval()
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    embeddings, mask, _ = lookup.batch([args.prompt], device)
    projected = text.input_projection(embeddings.float()).detach()
    reference = _image(args.input_image, 256, device)
    noise = torch.zeros((1, 4, 32, 32), device=device)

    def run():
        with torch.autocast(device.type, dtype=torch.float16):
            value = projected
            for layer in text.layers:
                value = layer(value, mask)
            value = text.norm(value)
            last = mask.long().sum(1).clamp_min(1) - 1
            pooled = value[torch.arange(len(value), device=device), last]
            condition = bridge(pooled.float())
            encoded = vae.encode(reference.half()).latent_dist.mode() * vae.config.scaling_factor
            latent = one_step_edit(unet, noise, encoded.float(), adapter.condition(condition.float()),
                                   scheduler.sigmas[0],
                                   noise_scale=payload["training_config"]["noise_scale"],
                                   residual_scale=payload["training_config"]["residual_scale"])
            _ = vae.decode(latent.half() / vae.config.scaling_factor, return_dict=False)[0]

    result = _timed(run, device, args.warmup, args.repeats)
    return {"name": "unified_large_256", "measured_electronic_path_ms": result,
            "optical_hardware_assumption_ms": PHYSICAL_OPTICAL_MS,
            "estimated_total_mean_ms": result["mean_ms"] + PHYSICAL_OPTICAL_MS,
            "counted_parameters": payload["counted_parameters"],
            "boundary": "first retained Qwen-mini block to final RGB; projection and embedding excluded"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("small-checkpoint", "large-checkpoint", "baseline-checkpoint", "initial-unet",
                 "turbo", "adapter-checkpoint", "qwen-checkpoint", "embedding-cache",
                 "input-image", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = next(iter(PromptEmbeddingLookup(args.embedding_cache).lookup))
    device = torch.device(args.device)
    small = _small(name="unified_small_256", checkpoint=args.small_checkpoint,
                   qwen_checkpoint=args.qwen_checkpoint, input_image=args.input_image,
                   prompt=args.prompt, device=device, warmup=args.warmup, repeats=args.repeats)
    large = _unified_large(args, device)
    baseline = _large(name="qwen28_electronic_baseline_256", checkpoint=args.baseline_checkpoint,
                      qwen_layers=28, electronic_baseline=True,
                      initial_unet=args.initial_unet,
                      turbo=args.turbo, adapter_checkpoint=args.adapter_checkpoint,
                      qwen_checkpoint=args.qwen_checkpoint, input_image=args.input_image,
                      prompt=args.prompt, device=device, warmup=args.warmup, repeats=args.repeats)
    result = {"small": small, "large": large, "baseline": baseline,
              "note": "Optical 6.2682 ms is an assumed device delay, not measured. Baseline is an older background-only model; timing but not quality is compared."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
