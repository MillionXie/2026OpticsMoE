"""Paired full-versus-no-diffraction ablation on identical held-out edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .audit_unified_optics import _indices
from .electronic_turbo_infer import _load_adapter
from .product_repair_model import RepairModelConfig, one_step_edit
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from .progressive_student import build_narrow_optical_unet
from .qwen_mini_small import PromptEmbeddingLookup, QwenMiniConfig, QwenMiniTextEncoder
from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor


class _ZeroDiffraction(nn.Module):
    def expert(self, value):
        return torch.zeros_like(value)

    def global_block(self, value):
        return torch.zeros_like(value)


def _append(rows, mode, full, ablated, target):
    rows.append({"mode": mode,
                 "full_rgb_mse": float(F.mse_loss(full.float(), target.float())),
                 "no_diffraction_rgb_mse": float(F.mse_loss(ablated.float(), target.float())),
                 "full_vs_no_diffraction_rgb_mae": float(F.l1_loss(full.float(), ablated.float()))})


def _summary(rows):
    result = {}
    for mode in ("background", "object", "joint"):
        selected = [row for row in rows if row["mode"] == mode]
        result[mode] = {key: sum(row[key] for row in selected) / len(selected)
                        for key in ("full_rgb_mse", "no_diffraction_rgb_mse",
                                    "full_vs_no_diffraction_rgb_mae")}
        result[mode]["relative_mse_change_when_disabled"] = (
            result[mode]["no_diffraction_rgb_mse"] /
            result[mode]["full_rgb_mse"] - 1)
    return result


@torch.inference_mode()
def _small(args, dataset, indices, device):
    payload = torch.load(args.small_checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload["editor_config"])
    values["widths"] = tuple(values["widths"])
    model = SmallFullFrameEditor(SmallEditorConfig(**values))
    model.text = QwenMiniTextEncoder(QwenMiniConfig(**payload["qwen_mini_config"]))
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    original_optics = model.bottleneck.optical
    rows = []
    for index in indices:
        sample = dataset[index]
        reference = sample["reference"][None].to(device)
        target = sample["target"][None].to(device)
        embeddings, mask, _ = lookup.batch([sample["prompt"]], device)
        noise = torch.zeros_like(reference)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model.bottleneck.optical = original_optics
            full = model(reference, (embeddings, mask), noise)
            model.bottleneck.optical = _ZeroDiffraction()
            ablated = model(reference, (embeddings, mask), noise)
        _append(rows, sample["mode"], full, ablated, target)
    return _summary(rows)


@torch.inference_mode()
def _large(args, dataset, indices, device):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(args.large_checkpoint, map_location="cpu", weights_only=False)
    base = UNet2DConditionModel.load_config(args.initial_unet, subfolder="unet",
                                            local_files_only=True)
    unet, optical, _ = build_narrow_optical_unet(
        base, payload["student_widths"], RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet = unet.to(device).eval()
    original_optics = optical.optical
    adapter, _ = _load_adapter(args.adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    vae = AutoencoderKL.from_pretrained(args.turbo, subfolder="vae", variant="fp16",
                                        torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.turbo, subfolder="scheduler",
                                                        local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    cache = torch.load(args.large_latents / "test.pt", map_location="cpu", weights_only=False)
    rows = []
    for index in indices:
        sample = dataset[index]
        target = sample["target"][None].to(device)
        reference = cache["reference"][index:index+1].float().to(device)
        condition = adapter.condition(cache["qwen_text"][index:index+1].float().to(device))
        def generate():
            latent = one_step_edit(unet, torch.zeros_like(reference), reference, condition,
                                   scheduler.sigmas[0],
                                   noise_scale=payload["training_config"]["noise_scale"],
                                   residual_scale=payload["training_config"]["residual_scale"])
            return vae.decode(latent.half() / vae.config.scaling_factor,
                              return_dict=False)[0]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            optical.optical = original_optics
            full = generate()
            optical.optical = _ZeroDiffraction()
            ablated = generate()
        _append(rows, sample["mode"], full, ablated, target)
    return _summary(rows)


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
    result = {"pairs": len(indices), "small": _small(args, dataset, indices, device),
              "large": _large(args, dataset, indices, device),
              "ablation": "FFT expert and global outputs set to zero; electronic residual, identity/base path, learned gates and RMS fusion left intact"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
