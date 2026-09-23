"""Matched timing for the 3-task large/small bundle and Qwen baseline.

The CUDA-event boundary begins at the input of the first retained language
Transformer block and ends after the RGB tensor is decoded. Tokenization,
token embedding lookup, model loading, and host-to-device input copies are
excluded. Optical FFT simulation is bypassed and replaced by 1.0447*6 ms.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from .electronic_turbo_infer import _load_adapter
from .feature_cache import _qwen_prompts
from .half_qwen import load_half_qwen_text_encoder
from .product_repair_model import RepairModelConfig, expand_reference_conditioning, one_step_edit
from .progressive_student import build_narrow_optical_unet
from .qwen_mini_small import QwenMiniConfig, QwenMiniTextEncoder
from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor


PHYSICAL_OPTICAL_MS = 1.0447 * 6
LARGE_RUNS = {
    "background": "abo_scene_replace_2layer_progressive_282m_v1",
    "redesign": "abo_global_redesign_2layer_282m_v1",
    "premium": "abo_premium_material_2layer_282m_v1",
}
SMALL_RUNS = {
    "background": "abo_background_qwenmini2_optical_under50m_v2",
    "redesign": "abo_redesign_qwenmini2_optical_under50m_v2",
    "premium": "abo_premium8_qwenmini2_optical_under50m_v2",
}


def _image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    value = ImageOps.fit(
        ImageOps.exif_transpose(Image.open(path)).convert("RGB"),
        (size, size), method=Image.Resampling.LANCZOS,
    )
    return torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)


def _timed(run: Callable[[], None], device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); run(); end.record(); torch.cuda.synchronize(device)
        values.append(float(start.elapsed_time(end)))
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": ordered[round((len(ordered) - 1) * 0.50)],
        "p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


@torch.inference_mode()
def _large(
    *, name: str, checkpoint: Path, qwen_layers: int, electronic_baseline: bool,
    initial_unet: Path, turbo: Path, adapter_checkpoint: Path, qwen_checkpoint: Path,
    input_image: Path, prompt: str, device: torch.device, warmup: int, repeats: int,
) -> dict[str, Any]:
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_config = RepairModelConfig(**payload["model_config"])
    training = payload["training_config"]
    qwen, processor, qwen_report = load_half_qwen_text_encoder(qwen_checkpoint, device, keep_layers=qwen_layers)
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.load_state_dict(payload["adapter"]); adapter.eval()
    if electronic_baseline:
        unet = UNet2DConditionModel.from_pretrained(
            initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
            local_files_only=True,
        )
        expand_reference_conditioning(unet)
        optical = None
    else:
        base_config = UNet2DConditionModel.load_config(initial_unet, subfolder="unet", local_files_only=True)
        unet, optical, _ = build_narrow_optical_unet(base_config, payload["student_widths"], model_config)
        optical.hardware_bypass = True
    unet.load_state_dict(payload["unet"]); unet = unet.to(device).eval()
    if optical is not None:
        optical.hardware_bypass = True
    vae = AutoencoderKL.from_pretrained(
        turbo, subfolder="vae", variant="fp16", torch_dtype=torch.float16, local_files_only=True,
    ).to(device).eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(turbo, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1, device=device); sigma = scheduler.sigmas[0]
    reference = _image(input_image, 256, device).half()
    token_inputs = {key: value.to(device) for key, value in _qwen_prompts(processor, [prompt]).items()}
    embeddings = qwen.embed_tokens(token_inputs.pop("input_ids")).detach()
    noise = torch.randn((1, 4, 32, 32), device=device, generator=torch.Generator(device=device).manual_seed(2026))

    def run() -> None:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            encoded = vae.encode(reference).latent_dist.mode() * vae.config.scaling_factor
            output = qwen(
                inputs_embeds=embeddings, **token_inputs, output_hidden_states=False,
                return_dict=True, use_cache=False,
            ).last_hidden_state.float()
            attention = token_inputs["attention_mask"].to(output.dtype).unsqueeze(-1)
            pooled = (output * attention).sum(1) / attention.sum(1).clamp_min(1)
            condition = adapter.condition(pooled)
            latent = one_step_edit(
                unet, noise, encoded.float(), condition, sigma,
                residual_scale=float(training["residual_scale"]),
                noise_scale=float(training["noise_scale"]),
            )
            _ = vae.decode(latent.half() / vae.config.scaling_factor, return_dict=False)[0]

    measured = _timed(run, device, warmup, repeats)
    hardware = measured["mean_ms"] + (0.0 if electronic_baseline else PHYSICAL_OPTICAL_MS)
    router_state = payload.get("attribute_router", payload.get("design_router", {}))
    counted_parameters = (
        int(qwen_report["counted_text_encoder_parameters"])
        + sum(parameter.numel() for parameter in vae.encoder.parameters())
        + sum(parameter.numel() for parameter in vae.quant_conv.parameters())
        + sum(parameter.numel() for parameter in unet.parameters())
        + sum(parameter.numel() for parameter in adapter.parameters())
        + sum(value.numel() for value in router_state.values())
        + sum(parameter.numel() for parameter in vae.decoder.parameters())
        + sum(parameter.numel() for parameter in vae.post_quant_conv.parameters())
    )
    result = {
        "name": name, "resolution": 256, "qwen_layers": qwen_layers,
        "counted_parameters": counted_parameters,
        "measured_optical_bypass": measured,
        "physical_optical_latency_ms": 0.0 if electronic_baseline else PHYSICAL_OPTICAL_MS,
        "estimated_hardware_mean_ms": hardware,
        "token_embedding_parameters_excluded": qwen_report["token_embedding_parameters_excluded_by_project_convention"],
    }
    del payload, qwen, processor, adapter, unet, vae, embeddings
    torch.cuda.empty_cache()
    return result


class _BypassOptics(nn.Module):
    def expert(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)

    def global_block(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)


def _small_generator(model: SmallFullFrameEditor, reference: torch.Tensor, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    s0 = model.stem(torch.cat((reference, 0.08 * noise), dim=1))
    s1 = model.down1(s0); s2 = model.down2(s1)
    value = model.bottleneck(model.down3(s2), condition)
    value = model.up3(value, s2, condition)
    value = model.up2(value, s1, condition)
    value = model.up1(value, s0, condition)
    delta = torch.tanh(model.to_delta(value)) * model.config.residual_limit
    return torch.tanh(torch.atanh(reference.float().clamp(-0.98, 0.98)) + delta.float())


@torch.inference_mode()
def _small(
    *, name: str, checkpoint: Path, qwen_checkpoint: Path, input_image: Path,
    prompt: str, device: torch.device, warmup: int, repeats: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    editor_values = dict(payload["editor_config"]); editor_values["widths"] = tuple(editor_values["widths"])
    editor_config = SmallEditorConfig(**editor_values)
    text_config = QwenMiniConfig(**payload["qwen_mini_config"])
    model = SmallFullFrameEditor(editor_config)
    model.text = QwenMiniTextEncoder(text_config)
    model.load_state_dict(payload["model"]); model = model.to(device).eval()
    # Remove software FFTs while preserving the electronic residual, fusion,
    # and all downstream decoder work.
    model.bottleneck.optical = _BypassOptics().to(device)
    language, processor, qwen_report = load_half_qwen_text_encoder(qwen_checkpoint, device, keep_layers=1)
    encoded = _qwen_prompts(processor, [prompt])
    ids = encoded["input_ids"][:, -text_config.max_length:].to(device)
    mask = encoded["attention_mask"][:, -text_config.max_length:].to(device).bool()
    embeddings = language.embed_tokens(ids)
    # Strict boundary: the 2048->768 projection precedes block 1.
    projected = model.text.input_projection(embeddings.float()).detach()
    reference = _image(input_image, editor_config.image_size, device)
    noise = torch.randn(reference.shape, device=device, generator=torch.Generator(device=device).manual_seed(2026))

    def run() -> None:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            value = projected
            for layer in model.text.layers:
                value = layer(value, mask)
            value = model.text.norm(value)
            last = mask.long().sum(1).clamp_min(1) - 1
            hidden = value[torch.arange(len(value), device=device), last]
            condition = model.text.condition_projection(hidden)
            _ = _small_generator(model, reference, noise, condition)

    measured = _timed(run, device, warmup, repeats)
    result = {
        "name": name, "resolution": editor_config.image_size, "qwen_layers": text_config.layers,
        "counted_parameters": int(payload["counted_parameters"]),
        "measured_optical_bypass": measured,
        "physical_optical_latency_ms": PHYSICAL_OPTICAL_MS,
        "estimated_hardware_mean_ms": measured["mean_ms"] + PHYSICAL_OPTICAL_MS,
        "token_embedding_parameters_excluded": qwen_report["token_embedding_parameters_excluded_by_project_convention"],
    }
    del payload, model, language, processor, embeddings, projected
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("runs-root", "initial-unet", "turbo", "adapter-checkpoint", "qwen-checkpoint", "input-image", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prompt", default="Regenerate this product as a slim premium design in a cool modern studio")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)
    baseline = _large(
        name="baseline_qwen28_electronic_decoder",
        checkpoint=(args.runs_root / "abo_scene_replace_28layer_electronic_baseline_v1" / "best_model.pt").resolve(),
        qwen_layers=28, electronic_baseline=True, initial_unet=args.initial_unet.resolve(),
        turbo=args.turbo.resolve(), adapter_checkpoint=args.adapter_checkpoint.resolve(),
        qwen_checkpoint=args.qwen_checkpoint.resolve(), input_image=args.input_image.resolve(),
        prompt=args.prompt, device=device, warmup=args.warmup, repeats=args.repeats,
    )
    large = {}
    for task, run_name in LARGE_RUNS.items():
        large[task] = _large(
            name=f"large_{task}_qwen2_optical", checkpoint=(args.runs_root / run_name / "best_model.pt").resolve(),
            qwen_layers=2, electronic_baseline=False, initial_unet=args.initial_unet.resolve(),
            turbo=args.turbo.resolve(), adapter_checkpoint=args.adapter_checkpoint.resolve(),
            qwen_checkpoint=args.qwen_checkpoint.resolve(), input_image=args.input_image.resolve(),
            prompt=args.prompt, device=device, warmup=args.warmup, repeats=args.repeats,
        )
    small = {}
    for task, run_name in SMALL_RUNS.items():
        small[task] = _small(
            name=f"small_{task}_qwenmini2_optical", checkpoint=(args.runs_root / run_name / "best_model.pt").resolve(),
            qwen_checkpoint=args.qwen_checkpoint.resolve(), input_image=args.input_image.resolve(),
            prompt=args.prompt, device=device, warmup=args.warmup, repeats=args.repeats,
        )
    for group in (large, small):
        for value in group.values():
            value["speedup_vs_baseline_x"] = baseline["estimated_hardware_mean_ms"] / value["estimated_hardware_mean_ms"]
            value["latency_reduction_fraction"] = 1.0 - value["estimated_hardware_mean_ms"] / baseline["estimated_hardware_mean_ms"]
    report = {
        "schema_version": 1, "device": torch.cuda.get_device_name(device), "batch_size": 1,
        "timing_boundary": "first retained language Transformer block input -> generated RGB tensor",
        "excluded": ["model loading", "tokenization", "token embedding lookup", "host-to-device input copy"],
        "optical_accounting": "software FFT bypassed; add 1.0447 * 6 ms; parallel residual is not added",
        "warmup": args.warmup, "repeats": args.repeats,
        "baseline": baseline, "large": large, "small": small,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
