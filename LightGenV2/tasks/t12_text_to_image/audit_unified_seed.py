"""Read-only audit of how much the deployed seed changes a fixed edit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .electronic_turbo_infer import _load_adapter
from .product_repair_model import RepairModelConfig, one_step_edit
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from .progressive_student import build_narrow_optical_unet
from .qwen_mini_small import PromptEmbeddingLookup, QwenMiniConfig, QwenMiniTextEncoder
from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor


def _indices(dataset):
    first = {}
    for source_index, source in enumerate(dataset.sources):
        first.setdefault(source["category"], source_index)
    return [first[category] * dataset.targets_per_source + offset
            for category in dataset.supported_categories for offset in (0, 4, 8)]


def _score(predictions, reference, target):
    stacked = torch.stack(predictions).float()
    return {
        "seed_pair_rgb_mae": float(F.l1_loss(stacked[0], stacked[1])),
        "mean_pixel_seed_std": float(stacked.std(0, unbiased=False).mean()),
        "reference_to_target_mae": float(F.l1_loss(reference.float(), target.float())),
        "mean_output_to_target_mae": float((stacked-target.float()).abs().mean()),
    }


@torch.inference_mode()
def _small(args, dataset, indices, device, seeds):
    payload = torch.load(args.small_checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload["editor_config"])
    values["widths"] = tuple(values["widths"])
    model = SmallFullFrameEditor(SmallEditorConfig(**values))
    model.text = QwenMiniTextEncoder(QwenMiniConfig(**payload["qwen_mini_config"]))
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    result = []
    for index in indices:
        sample = dataset[index]
        reference = sample["reference"][None].to(device)
        target = sample["target"][None].to(device)
        embeddings, mask, _ = lookup.batch([sample["prompt"]], device)
        predictions = []
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(reference.shape, generator=generator, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions.append(model(reference, (embeddings, mask), noise))
        result.append({"sample_id": sample["sample_id"], "mode": sample["mode"],
                       **_score(predictions, reference, target)})
    return result


@torch.inference_mode()
def _large(args, dataset, indices, device, seeds):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(args.large_checkpoint, map_location="cpu", weights_only=False)
    base = UNet2DConditionModel.load_config(args.initial_unet, subfolder="unet",
                                            local_files_only=True)
    unet, _, _ = build_narrow_optical_unet(
        base, payload["student_widths"], RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet = unet.to(device).eval()
    adapter, _ = _load_adapter(args.adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    vae = AutoencoderKL.from_pretrained(args.turbo, subfolder="vae", variant="fp16",
                                        torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.turbo, subfolder="scheduler",
                                                        local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    cache = torch.load(args.large_latents / "test.pt", map_location="cpu", weights_only=False)
    result = []
    for index in indices:
        sample = dataset[index]
        reference_latent = cache["reference"][index:index+1].float().to(device)
        condition = adapter.condition(cache["qwen_text"][index:index+1].float().to(device))
        predictions = []
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(reference_latent.shape, generator=generator, device=device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                latent = one_step_edit(unet, noise, reference_latent, condition,
                                       scheduler.sigmas[0],
                                       noise_scale=payload["training_config"]["noise_scale"],
                                       residual_scale=payload["training_config"]["residual_scale"])
                predictions.append(vae.decode(latent.half() / vae.config.scaling_factor,
                                              return_dict=False)[0])
        result.append({"sample_id": sample["sample_id"], "mode": sample["mode"],
                       **_score(predictions, sample["reference"][None].to(device),
                                sample["target"][None].to(device))})
    return result


def _aggregate(rows):
    keys = ("seed_pair_rgb_mae", "mean_pixel_seed_std", "reference_to_target_mae",
            "mean_output_to_target_mae")
    result = {}
    for mode in ("background", "object", "joint"):
        selected = [row for row in rows if row["mode"] == mode]
        result[mode] = {key: sum(row[key] for row in selected) / len(selected) for key in keys}
        result[mode]["seed_change_to_requested_edit_ratio"] = (
            result[mode]["seed_pair_rgb_mae"] /
            max(1e-8, result[mode]["reference_to_target_mae"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("small-checkpoint", "large-checkpoint", "data-dir", "instruction-cache",
                 "embedding-cache", "large-latents", "initial-unet", "turbo",
                 "adapter-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = ExpandedUnifiedProductEditDataset(args.data_dir, "test", 256,
                                                args.instruction_cache)
    indices = _indices(dataset)
    seeds = (11, 29, 47, 83)
    small = _small(args, dataset, indices, device, seeds)
    large = _large(args, dataset, indices, device, seeds)
    result = {"seeds": seeds, "test_pairs": len(indices), "small": _aggregate(small),
              "large": _aggregate(large), "sample_rows": {"small": small, "large": large},
              "pixel_range": "[-1, 1]",
              "interpretation": "Different seed has a real noise path, but effect magnitude must be compared with intended edit magnitude."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("seeds", "test_pairs", "small", "large")},
                     indent=2))


if __name__ == "__main__":
    main()
