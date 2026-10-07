"""Measure optical fusion alpha and actual Top-2 expert routing on held-out edits.

This runs the differentiable FFT simulator, not physical SLM hardware.  It
reports selection frequency and normalized mixture weight separately so an
apparently balanced softmax cannot hide a collapsed Top-2 assignment.
"""
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


def _indices(dataset: ExpandedUnifiedProductEditDataset) -> list[int]:
    # One held-out view of each of eight distinct products per category, with
    # all 12 edit offsets. This gives 288 balanced test pairs (96/mode).
    selected = {}
    for source_index, source in enumerate(dataset.sources):
        selected.setdefault(source["category"], {})
        selected[source["category"]].setdefault(source["sequence_id"], source_index)
    indices = []
    for category in dataset.supported_categories:
        for source_index in list(selected[category].values())[:8]:
            indices.extend(source_index * dataset.targets_per_source + offset
                           for offset in range(dataset.targets_per_source))
    return indices


def _record(accumulator: dict, mode: str, route: dict) -> None:
    probabilities = route["probabilities"].detach().float().cpu()
    selected = route["selected_mask"].detach().float().cpu()
    weights = route["weights"].detach().float().cpu()
    row = accumulator.setdefault(mode, {"count": 0,
                                       "probability": torch.zeros(probabilities.shape[-1]),
                                       "selection": torch.zeros(probabilities.shape[-1]),
                                       "weight": torch.zeros(probabilities.shape[-1])})
    row["count"] += len(probabilities)
    row["probability"] += probabilities.sum(0)
    row["selection"] += selected.sum(0)
    row["weight"] += weights.sum(0)


def _finish(accumulator: dict) -> dict:
    return {mode: {"n": row["count"],
                   "mean_softmax_probability": (row["probability"] / row["count"]).tolist(),
                   "top2_selection_frequency": (row["selection"] / row["count"]).tolist(),
                   "mean_mixture_weight": (row["weight"] / row["count"]).tolist()}
            for mode, row in accumulator.items()}


def _record_error(accumulator: dict, mode: str, output: torch.Tensor,
                  target: torch.Tensor) -> None:
    row = accumulator.setdefault(mode, {"sum": 0.0, "count": 0})
    row["sum"] += float(F.mse_loss(output.float(), target.float()))
    row["count"] += 1


def _finish_error(accumulator: dict) -> dict:
    return {mode: row["sum"] / row["count"] for mode, row in accumulator.items()}


@torch.inference_mode()
def audit_small(args, dataset, indices, device):
    payload = torch.load(args.small_checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload["editor_config"])
    values["widths"] = tuple(values["widths"])
    config = SmallEditorConfig(**values)
    model = SmallFullFrameEditor(config)
    model.text = QwenMiniTextEncoder(QwenMiniConfig(**payload["qwen_mini_config"]))
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    routes = {}
    errors = {}
    gate_statistics = {"changed_sum": 0.0, "changed_count": 0.0,
                       "unchanged_sum": 0.0, "unchanged_count": 0.0}
    for index in indices:
        sample = dataset[index]
        reference = sample["reference"][None].to(device)
        embeddings, mask, _ = lookup.batch([sample["prompt"]], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if model.source_gate is not None:
                output, source_logits = model(reference, (embeddings, mask),
                                              torch.zeros_like(reference), None,
                                              return_source_gate=True)
            else:
                output = model(reference, (embeddings, mask), torch.zeros_like(reference), None)
        _record(routes, sample["mode"], model.bottleneck.optical.last_routing)
        target = sample["target"][None].to(device)
        _record_error(errors, sample["mode"], output, target)
        if model.source_gate is not None:
            retention = source_logits.sigmoid()
            difference = (target-reference).abs().mean(1, keepdim=True)
            changed = difference > 0.10
            unchanged = difference < 0.02
            gate_statistics["changed_sum"] += float(retention[changed].sum())
            gate_statistics["changed_count"] += int(changed.sum())
            gate_statistics["unchanged_sum"] += float(retention[unchanged].sum())
            gate_statistics["unchanged_count"] += int(unchanged.sum())
    return {"resolution": config.image_size,
            "alpha": float(model.bottleneck.fusion.alpha),
            "expert_gate": float(model.bottleneck.expert_gate.sigmoid()),
            "global_gate": float(model.bottleneck.global_gate.sigmoid()),
            "by_mode": _finish(routes), "rgb_mse_by_mode": _finish_error(errors),
            "source_retention": {
                "changed_pixels": gate_statistics["changed_sum"] / max(1, gate_statistics["changed_count"]),
                "unchanged_pixels": gate_statistics["unchanged_sum"] / max(1, gate_statistics["unchanged_count"]),
            } if model.source_gate is not None else None}


@torch.inference_mode()
def audit_large(args, dataset, indices, device):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(args.large_checkpoint, map_location="cpu", weights_only=False)
    base = UNet2DConditionModel.load_config(args.initial_unet, subfolder="unet",
                                            local_files_only=True)
    unet, optical, _ = build_narrow_optical_unet(
        base, payload["student_widths"], RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet = unet.to(device).eval()
    adapter, _ = _load_adapter(args.adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.turbo, subfolder="scheduler",
                                                       local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    vae = AutoencoderKL.from_pretrained(args.turbo, subfolder="vae", variant="fp16",
                                       torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    cache = torch.load(args.large_latents / "test.pt", map_location="cpu", weights_only=False)
    routes = {}
    errors = {}
    for index in indices:
        sample = dataset[index]
        reference = cache["reference"][index:index+1].float().to(device)
        text = cache["qwen_text"][index:index+1].float().to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output_latent = one_step_edit(unet, torch.zeros_like(reference), reference,
                                          adapter.condition(text), scheduler.sigmas[0],
                                          noise_scale=payload["training_config"]["noise_scale"],
                                          residual_scale=payload["training_config"]["residual_scale"])
            output = vae.decode(output_latent.half() / vae.config.scaling_factor,
                                return_dict=False)[0]
        _record(routes, sample["mode"], optical.optical.last_routing)
        _record_error(errors, sample["mode"], output, sample["target"][None].to(device))
    return {"resolution": 256,
            "alpha": float(optical.fusion.alpha),
            "expert_gate": float(optical.expert_gate.sigmoid()),
            "global_gate": float(optical.global_gate.sigmoid()),
            "by_mode": _finish(routes), "rgb_mse_by_mode": _finish_error(errors)}


def main() -> int:
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
    result = {"schema_version": 1, "pairs": len(indices),
              "sample_rule": "one view of eight held-out product identities per category, all twelve edit offsets",
              "small": audit_small(args, dataset, indices, device),
              "large": audit_large(args, dataset, indices, device),
              "caveat": "Routing is from the learned FFT optical simulator; alpha is an RMS-normalized fusion coefficient, not a photon or latency fraction."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
