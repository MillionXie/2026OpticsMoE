"""Evaluate a trained one-pass text-controlled ABO lamp scene generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .electronic_turbo_infer import _load_adapter
from .product_repair_model import (
    RepairModelConfig,
    architecture_report,
    attach_decoder_optics,
    expand_reference_conditioning,
)
from .product_scene_data import ProductSceneDataset, SCENES
from .product_scene_training import (
    SceneLatentDataset,
    TextSceneRouter,
    _sample_grid,
    evaluate_scene_model,
)


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
    training_config = payload["training_config"]
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    router = TextSceneRouter(2048).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    expand_reference_conditioning(unet)
    optical = attach_decoder_optics(unet, model_config)
    unet.load_state_dict(payload["unet"])
    adapter.load_state_dict(payload["adapter"])
    router.load_state_dict(payload["scene_router"])
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
    test = SceneLatentDataset(latent_cache_dir / "test.pt")
    loader = DataLoader(test, batch_size=6, shuffle=False, num_workers=4)
    metrics = evaluate_scene_model(
        unet, adapter, router, loader, sigma, device,
        residual_scale=float(training_config["residual_scale"]),
        noise_scale=float(training_config["noise_scale"]),
    )
    raw_test = ProductSceneDataset(data_dir, "test", 256, instruction_cache)
    _sample_grid(
        unet=unet, adapter=adapter, vae=vae, sigma=sigma,
        latent_dataset=test, raw_dataset=raw_test, output=output_dir / "test_grid.jpg",
        device=device, residual_scale=float(training_config["residual_scale"]),
        noise_scale=float(training_config["noise_scale"]), seed=seed,
    )
    architecture = architecture_report(unet, vae, adapter, router, optical, model_config)
    architecture["text_region_router_parameters"] = 0
    architecture["text_scene_router_parameters"] = sum(p.numel() for p in router.parameters())
    architecture["generation_tail_parameters"] = (
        architecture["unet_parameters"] + architecture["vae_decoder_parameters"]
        + architecture["qwen_condition_adapter_parameters"]
        + architecture["text_scene_router_parameters"]
    )
    report = {
        "schema_version": 1,
        "task": "white-background lamp + Qwen instruction -> styled product scene",
        "split": "identity-disjoint test",
        "samples": len(test),
        "underlying_input_images": len(test) // len(SCENES),
        "scenes_per_input": len(SCENES),
        "metrics": metrics,
        "architecture": architecture,
        "checkpoint": str(trained_checkpoint),
        "random_seed_is_real": True,
        "same_seed_for_compared_prompts": True,
        "gan_used": False,
        "inference_iterations": 1,
    }
    (output_dir / "test_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    del unet, vae, adapter, router
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ABO lamp scene generation")
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
        initial_unet=args.initial_unet.resolve(), turbo_checkpoint=args.turbo_checkpoint.resolve(),
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
