"""Evaluate the best compositional background-replacement checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .electronic_turbo_infer import _load_adapter
from .compact_product_model import ElectronicBaselineMarker, prepare_compact_optical_unet
from .product_repair_model import (
    RepairModelConfig,
    architecture_report,
    attach_decoder_optics,
    expand_reference_conditioning,
)
from .product_scene_replace_data import ProductBackgroundReplacementDataset
from .product_scene_replace_training import (
    ReplacementLatentDataset,
    TextAttributeRouter,
    _sample_grid,
    evaluate_replacement_model,
)


@torch.inference_mode()
def evaluate_checkpoint(
    *, initial_unet: Path, turbo_checkpoint: Path, adapter_checkpoint: Path,
    trained_checkpoint: Path, latent_cache_dir: Path, data_dir: Path,
    instruction_cache: Path, output_dir: Path, device: torch.device, seed: int = 42,
    compact_optical_mid: bool = False, electronic_baseline: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(trained_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_config = RepairModelConfig(**payload["model_config"])
    training_config = payload["training_config"]
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    router = TextAttributeRouter(2048).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    expand_reference_conditioning(unet)
    if compact_optical_mid and electronic_baseline:
        raise ValueError("Choose either compact optics or the electronic baseline")
    if electronic_baseline:
        optical = ElectronicBaselineMarker().to(device)
    elif compact_optical_mid:
        optical, _ = prepare_compact_optical_unet(unet, model_config)
    else:
        optical = attach_decoder_optics(unet, model_config)
    unet.load_state_dict(payload["unet"])
    adapter.load_state_dict(payload["adapter"])
    router.load_state_dict(payload["attribute_router"])
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
    test = ReplacementLatentDataset(latent_cache_dir / "test.pt")
    loader = DataLoader(test, batch_size=2, shuffle=False, num_workers=4)
    metrics = evaluate_replacement_model(
        unet, adapter, router, loader, sigma, device,
        residual_scale=float(training_config["residual_scale"]),
        noise_scale=float(training_config["noise_scale"]),
    )
    raw_test = ProductBackgroundReplacementDataset(data_dir, "test", 256, instruction_cache)
    _sample_grid(
        unet=unet, adapter=adapter, vae=vae, sigma=sigma,
        latent_dataset=test, raw_dataset=raw_test, output=output_dir / "test_grid.jpg",
        device=device, residual_scale=float(training_config["residual_scale"]),
        noise_scale=float(training_config["noise_scale"]), seed=seed,
    )
    qwen = torch.load(instruction_cache, map_location="cpu", weights_only=False)["meta"]["qwen_pruning"]
    architecture = architecture_report(unet, vae, adapter, router, optical, model_config)
    architecture.update({
        "text_attribute_router_parameters": sum(p.numel() for p in router.parameters()),
        "qwen_text_encoder_parameters": qwen["retained_text_encoder_parameters"],
        "qwen_language_layers": f"{qwen['language_layers_retained']}/{qwen['language_layers_original']}",
        "qwen_vision_tower_used": False, "qwen_lm_head_used": False,
        "vae_encoder_parameters": (
            sum(p.numel() for p in vae.encoder.parameters())
            + sum(p.numel() for p in vae.quant_conv.parameters())
        ),
        "token_embedding_parameters_excluded": qwen.get(
            "token_embedding_parameters_excluded_by_project_convention", 0
        ),
        "counted_qwen_parameters": qwen.get(
            "counted_text_encoder_parameters", qwen["retained_text_encoder_parameters"]
        ),
    })
    architecture["counted_end_to_end_parameters"] = (
        architecture["counted_qwen_parameters"]
        + architecture["vae_encoder_parameters"]
        + architecture["generation_tail_parameters"]
    )
    report = {
        "schema_version": 2,
        "task": "existing scene + compositional text -> replaced background",
        "split": "identity-disjoint test; exact target combinations excluded from training",
        "samples": len(test), "underlying_input_images": len(raw_test.rows),
        "target_combinations_per_input": raw_test.targets_per_source,
        "metrics": metrics, "architecture": architecture, "qwen_pruning": qwen,
        "checkpoint": str(trained_checkpoint), "random_seed_is_real": True,
        "same_seed_for_compared_prompts": True, "gan_used": False,
        "inference_iterations": 1,
    }
    (output_dir / "test_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del unet, vae, adapter, router
    if device.type == "cuda": torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate compositional background replacement")
    for name in (
        "initial-unet", "turbo-checkpoint", "adapter-checkpoint", "trained-checkpoint",
        "latent-cache-dir", "data-dir", "instruction-cache", "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compact-optical-mid", action="store_true")
    parser.add_argument("--electronic-baseline", action="store_true")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        initial_unet=args.initial_unet.resolve(), turbo_checkpoint=args.turbo_checkpoint.resolve(),
        adapter_checkpoint=args.adapter_checkpoint.resolve(), trained_checkpoint=args.trained_checkpoint.resolve(),
        latent_cache_dir=args.latent_cache_dir.resolve(), data_dir=args.data_dir.resolve(),
        instruction_cache=args.instruction_cache.resolve(), output_dir=args.output_dir.resolve(),
        device=torch.device(args.device), seed=args.seed,
        compact_optical_mid=args.compact_optical_mid,
        electronic_baseline=args.electronic_baseline,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
