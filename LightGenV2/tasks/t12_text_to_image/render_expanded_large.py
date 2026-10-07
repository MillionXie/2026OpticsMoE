"""Render a held-out target-design grid from the saved large optical editor."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .electronic_turbo_infer import _load_adapter
from .product_global_redesign_training import RedesignLatentDataset, _sample_grid
from .product_repair_model import RepairModelConfig
from .product_scene_training import SceneTrainingConfig
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from .progressive_student import build_narrow_optical_unet


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "initial-unet", "turbo-checkpoint", "adapter-checkpoint",
                 "latent-cache-dir", "data-dir", "instruction-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gallery", choices=("balanced", "four-designs"), default="balanced")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_config = UNet2DConditionModel.load_config(args.initial_unet, subfolder="unet",
                                                   local_files_only=True)
    unet, _, _ = build_narrow_optical_unet(
        base_config, tuple(payload["student_widths"]),
        RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet = unet.to(device).eval()
    adapter, _ = _load_adapter(args.adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(args.turbo_checkpoint, subfolder="vae",
                                        variant="fp16", torch_dtype=dtype,
                                        local_files_only=True).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.turbo_checkpoint,
                                                       subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    latents = RedesignLatentDataset(args.latent_cache_dir / "test.pt")
    raw = ExpandedUnifiedProductEditDataset(args.data_dir, "test", 256,
                                            args.instruction_cache)
    offsets = [4, 5, 6, 7] if args.gallery == "four-designs" else None
    _sample_grid(unet=unet, adapter=adapter, vae=vae, sigma=scheduler.sigmas[0],
                 latent_dataset=latents, raw_dataset=raw, output=args.output,
                 device=device, training=SceneTrainingConfig(**payload["training_config"]),
                 seed=2026, chosen_offsets=offsets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
