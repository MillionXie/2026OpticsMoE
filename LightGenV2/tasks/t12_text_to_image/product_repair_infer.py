"""Evaluate a trained one-step Qwen + optical decoder product editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .electronic_turbo_infer import _load_adapter
from .product_repair_data import ProductRepairDataset
from .product_repair_model import (
    RepairModelConfig,
    TextRegionRouter,
    architecture_report,
    attach_decoder_optics,
    expand_reference_conditioning,
)
from .product_repair_training import RepairLatentDataset, _evaluate, _sample_grid


@torch.inference_mode()
def evaluate_checkpoint(
    *,
    initial_unet: Path,
    turbo_checkpoint: Path,
    adapter_checkpoint: Path,
    trained_checkpoint: Path,
    latent_cache_dir: Path,
    data_dir: Path,
    instruction_cache: Path,
    output_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(trained_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_config = RepairModelConfig(**payload["model_config"])
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    router = TextRegionRouter(2048).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    expand_reference_conditioning(unet)
    optical = attach_decoder_optics(unet, model_config)
    unet.load_state_dict(payload["unet"])
    adapter.load_state_dict(payload["adapter"])
    router.load_state_dict(payload["region_router"])
    unet.eval(); adapter.eval(); router.eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1, device=device)
    sigma = scheduler.sigmas[0]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        turbo_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    test = RepairLatentDataset(latent_cache_dir / "test.pt")
    loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=4)
    metrics = _evaluate(unet, adapter, router, loader, sigma, device)
    raw_test = ProductRepairDataset(data_dir, "test", 256, instruction_cache, seed=seed)
    _sample_grid(
        unet=unet, adapter=adapter, region_router=router, vae=vae, sigma=sigma,
        latent_dataset=test, raw_dataset=raw_test, output=output_dir / "test_grid.jpg",
        device=device, seed=seed,
    )
    report = {
        "schema_version": 1,
        "split": "identity-disjoint test",
        "samples": len(test),
        "underlying_images": len(test) // 2,
        "metrics": metrics,
        "architecture": architecture_report(
            unet, vae, adapter, router, optical, model_config
        ),
        "checkpoint": str(trained_checkpoint),
        "random_seed_is_real": True,
        "same_seed_within_counterfactual_pair": True,
        "gan_used": False,
    }
    (output_dir / "test_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    del unet, vae, adapter, router
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ABO text-selected product restoration")
    parser.add_argument("--initial-unet", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-cache-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--instruction-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        initial_unet=args.initial_unet.resolve(),
        turbo_checkpoint=args.turbo_checkpoint.resolve(),
        adapter_checkpoint=args.adapter_checkpoint.resolve(),
        trained_checkpoint=args.trained_checkpoint.resolve(),
        latent_cache_dir=args.latent_cache_dir.resolve(), data_dir=args.data_dir.resolve(),
        instruction_cache=args.instruction_cache.resolve(), output_dir=args.output_dir.resolve(),
        device=torch.device(args.device), seed=args.seed,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

