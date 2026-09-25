"""Cache and fine-tune the 282M one-pass model for full-frame product redesign."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .compact_turbo import latent_gradient_loss
from .electronic_turbo_infer import _load_adapter
from .product_global_redesign_data import ProductGlobalRedesignDataset
from .product_repair_model import RepairModelConfig, one_step_edit
from .product_scene_training import SceneTrainingConfig, _paired_noise, _seed_everything, load_scene_config
from .progressive_student import build_narrow_optical_unet, copy_overlapping_state, counted_student_parameters


class RedesignLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def __len__(self) -> int:
        return len(self.payload["reference"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "reference": self.payload["reference"][index].float(),
            "target": self.payload["target"][index].float(),
            "qwen_text": self.payload["qwen_text"][index].float(),
            "catalogue_indices": self.payload["catalogue_indices"][index],
            "seed": self.payload.get("seeds", list(range(len(self))))[index],
        }
        if "modes" in self.payload:
            item["mode"] = self.payload["modes"][index]
        return item


@torch.inference_mode()
def cache_redesign_latents(
    *, data_dir: Path, instruction_cache: Path, vae_checkpoint: Path,
    output_dir: Path, image_size: int, device: torch.device,
    batch_size: int = 12, num_workers: int = 4,
    dataset_class=ProductGlobalRedesignDataset,
    task_name: str = "full-frame same-category product redesign",
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKL

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        vae_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    scale = float(vae.config.scaling_factor)
    summary = {
        "schema_version": 1, "task": task_name,
        "hard_pixel_composite_at_inference": False, "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = dataset_class(data_dir, split, image_size, instruction_cache)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
        values: dict[str, list[Any]] = {key: [] for key in (
            "reference", "target", "qwen_text", "catalogue_indices", "sample_ids", "prompts", "seeds",
        )}
        values["modes"] = []
        for batch in loader:
            reference = batch["reference"].to(device=device, dtype=dtype, non_blocking=True)
            target = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
            values["reference"].append((vae.encode(reference).latent_dist.mode() * scale).half().cpu())
            values["target"].append((vae.encode(target).latent_dist.mode() * scale).half().cpu())
            values["qwen_text"].append(batch["qwen_text"].bfloat16().cpu())
            values["catalogue_indices"].extend(batch["catalogue_index"].tolist())
            values["sample_ids"].extend(batch["sample_id"])
            values["prompts"].extend(batch["prompt"])
            if "mode" in batch:
                values["modes"].extend(batch["mode"])
            if "seed" in batch:
                values["seeds"].extend(torch.as_tensor(batch["seed"]).tolist())
            else:
                values["seeds"].extend(range(len(values["seeds"]), len(values["seeds"]) + len(reference)))
        packed = {
            key: torch.cat(value) if key in {"reference", "target", "qwen_text"} else value
            for key, value in values.items() if key != "modes" or value
        }
        torch.save(packed, output_dir / f"{split}.pt")
        summary["splits"][split] = len(dataset)
    (output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


class TextDesignRouter(nn.Module):
    def __init__(self, text_dim: int, catalogue_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, catalogue_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())


@torch.inference_mode()
def _seeded_noise(reference: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    values = []
    for seed in seeds.tolist():
        generator = torch.Generator(device=reference.device).manual_seed(int(seed))
        values.append(torch.randn(reference.shape[1:], generator=generator, device=reference.device, dtype=reference.dtype))
    return torch.stack(values)


@torch.inference_mode()
def _evaluate(unet, adapter, router, loader, sigma, device, training: SceneTrainingConfig, *, seeded_noise: bool = False) -> dict[str, float]:
    unet.eval(); adapter.eval(); router.eval()
    totals = {"latent_mse": 0.0, "latent_l1": 0.0, "reference_mse": 0.0, "catalogue_accuracy": 0.0}
    count = 0
    by_mode: dict[str, dict[str, float]] = {}
    generator = torch.Generator(device=device).manual_seed(6543)
    for batch in loader:
        reference = batch["reference"].to(device); target = batch["target"].to(device)
        text = batch["qwen_text"].to(device); labels = torch.as_tensor(batch["catalogue_indices"], device=device)
        noise = _seeded_noise(reference, batch["seed"]) if seeded_noise else _paired_noise(reference, generator)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = one_step_edit(
                unet, noise, reference, adapter.condition(text), sigma,
                residual_scale=training.residual_scale, noise_scale=training.noise_scale,
            ).float()
        size = len(reference)
        totals["latent_mse"] += float(F.mse_loss(output, target)) * size
        totals["latent_l1"] += float(F.l1_loss(output, target)) * size
        totals["reference_mse"] += float(F.mse_loss(reference, target)) * size
        totals["catalogue_accuracy"] += float((router(text).argmax(-1) == labels).float().mean()) * size
        if "mode" in batch:
            for index, mode in enumerate(batch["mode"]):
                row = by_mode.setdefault(mode, {"latent_mse": 0.0, "count": 0.0})
                row["latent_mse"] += float(F.mse_loss(output[index], target[index]))
                row["count"] += 1
        count += size
    result = {key: value / count for key, value in totals.items()}
    result["mse_improvement_over_copy"] = 1.0 - result["latent_mse"] / max(result["reference_mse"], 1e-8)
    for mode, row in by_mode.items():
        result[f"{mode}_latent_mse"] = row["latent_mse"] / row["count"]
    return result


@torch.inference_mode()
def _sample_grid(*, unet, adapter, vae, sigma, latent_dataset, raw_dataset, output: Path, device, training: SceneTrainingConfig, seed: int) -> None:
    # Show both supported categories instead of merely taking the first eight
    # manifest rows (ABO manifests are category-grouped).
    first_by_category: dict[str, int] = {}
    for source_index, source in enumerate(raw_dataset.sources):
        first_by_category.setdefault(source["category"], source_index)
    chosen = []
    display_per_category = min(8, raw_dataset.targets_per_source)
    for category in raw_dataset.supported_categories:
        start = first_by_category[category] * raw_dataset.targets_per_source
        chosen.extend(range(start, min(start + display_per_category, len(raw_dataset))))
    reference = latent_dataset.payload["reference"][chosen].float().to(device)
    text = latent_dataset.payload["qwen_text"][chosen].float().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    if "seeds" in latent_dataset.payload:
        noise = _seeded_noise(reference, torch.as_tensor([latent_dataset.payload["seeds"][index] for index in chosen]))
    else:
        noise = torch.randn(reference.shape, generator=generator, device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        latent = one_step_edit(
            unet, noise, reference, adapter.condition(text), sigma,
            residual_scale=training.residual_scale, noise_scale=training.noise_scale,
        )
        generated = vae.decode(latent.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    generated = generated.float().clamp(-1, 1).cpu()
    cell, labels = raw_dataset.image_size, 330
    canvas = Image.new("RGB", (labels + 3 * cell, len(chosen) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(chosen):
        raw = raw_dataset[index]
        for column, value in enumerate((raw["reference"], raw["target"], generated[row])):
            array = value.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (labels + column * cell, row * cell))
        draw.text((4, row * cell + 4), raw["target_id"], fill="black")
        draw.text((4, row * cell + 28), raw["prompt"][:52], fill="black")
        draw.text((4, row * cell + 52), "input | target | full generated", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94, subsampling=0)


def train_redesign_model(
    *, initial_unet: Path, turbo_checkpoint: Path, latent_cache_dir: Path,
    data_dir: Path, instruction_cache: Path, adapter_checkpoint: Path,
    warm_start_checkpoint: Path, output_dir: Path,
    model_config: RepairModelConfig, training_config: SceneTrainingConfig,
    device: torch.device, seed: int = 42,
    dataset_class=ProductGlobalRedesignDataset,
    task_name: str = "full-frame text-guided product redesign",
    student_widths: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True); _seed_everything(seed)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    warm = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    widths = tuple(int(value) for value in (student_widths or warm["student_widths"]))
    base_config = UNet2DConditionModel.load_config(initial_unet, subfolder="unet", local_files_only=True)
    unet, optical, pruning = build_narrow_optical_unet(base_config, widths, model_config)
    if widths == tuple(int(value) for value in warm["student_widths"]):
        unet.load_state_dict(warm["unet"])
        warm_copy = {"exact": True}
    else:
        warm_copy = copy_overlapping_state(unet, warm["unet"])
    unet = unet.to(device)
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.load_state_dict(warm["adapter"]); adapter.requires_grad_(True)
    datasets = {name: RedesignLatentDataset(latent_cache_dir / f"{name}.pt") for name in ("train", "val", "test")}
    loaders = {
        name: DataLoader(value, batch_size=training_config.batch_size, shuffle=name == "train", num_workers=training_config.num_workers, pin_memory=device.type == "cuda")
        for name, value in datasets.items()
    }
    text_dim = int(datasets["train"].payload["qwen_text"].shape[1])
    catalogue_size = max(datasets["train"].payload["catalogue_indices"]) + 1
    router = TextDesignRouter(text_dim, catalogue_size).to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(turbo_checkpoint, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1, device=device); sigma = scheduler.sigmas[0]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(turbo_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype, local_files_only=True).to(device).eval().requires_grad_(False)
    raw_val = dataset_class(data_dir, "val", training_config.image_size, instruction_cache)
    # Local fine-tuning: preserve the compressed encoder and adapt the optical
    # bottleneck plus decoder/output path to the new full-frame distribution.
    unet.requires_grad_(False); unet.enable_gradient_checkpointing()
    if "morphology" in task_name:
        # Shape editing changes global geometry, so the encoder must adapt as
        # well as the optical bottleneck and decoder. This changes no counted
        # parameters; Qwen and the VAE remain frozen.
        unet.requires_grad_(True)
    else:
        unet.conv_in.requires_grad_(True); unet.mid_block.requires_grad_(True)
        unet.up_blocks.requires_grad_(True); unet.conv_norm_out.requires_grad_(True); unet.conv_out.requires_grad_(True)
    trainable_unet = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optical_ids = {id(parameter) for parameter in optical.parameters()}
    optical_parameters = [parameter for parameter in trainable_unet if id(parameter) in optical_ids]
    electronic_parameters = [parameter for parameter in trainable_unet if id(parameter) not in optical_ids]
    optimizer = torch.optim.AdamW([
        {"params": electronic_parameters, "lr": training_config.learning_rate},
        {"params": optical_parameters, "lr": training_config.optical_learning_rate},
        {"params": adapter.parameters(), "lr": training_config.adapter_learning_rate},
        {"params": router.parameters(), "lr": training_config.adapter_learning_rate},
    ], weight_decay=training_config.weight_decay)
    morphology_task = "morphology" in task_name
    initial = _evaluate(unet, adapter, router, loaders["val"], sigma, device, training_config, seeded_noise=morphology_task)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = math.inf; best_epoch = 0; history = []; started = time.perf_counter()
    for epoch in range(1, training_config.epochs + 1):
        unet.train(); adapter.train(); router.train(); optimizer.zero_grad(set_to_none=True)
        total = samples = 0
        for step, batch in enumerate(loaders["train"], 1):
            reference = batch["reference"].to(device); target = batch["target"].to(device)
            text = batch["qwen_text"].to(device); labels = torch.as_tensor(batch["catalogue_indices"], device=device)
            noise = _seeded_noise(reference, batch["seed"]) if morphology_task else _paired_noise(reference)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = one_step_edit(
                    unet, noise, reference, adapter.condition(text), sigma,
                    residual_scale=training_config.residual_scale, noise_scale=training_config.noise_scale,
                ).float()
                loss = F.mse_loss(output, target) + 0.45 * F.l1_loss(output, target)
                loss = loss + training_config.detail_weight * latent_gradient_loss(output, target)
                loss = loss + training_config.scene_router_weight * F.cross_entropy(router(text), labels)
                decoded = vae.decode(output.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].float()
                with torch.no_grad():
                    decoded_target = vae.decode(target.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].float()
                loss = loss + 0.25 * F.l1_loss(decoded, decoded_target)
                scaled = loss / training_config.gradient_accumulation
            scaler.scale(scaled).backward()
            if step % training_config.gradient_accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([*trainable_unet, *adapter.parameters(), *router.parameters()], 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(reference); samples += len(reference)
        validation = _evaluate(unet, adapter, router, loaders["val"], sigma, device, training_config, seeded_noise=morphology_task)
        row = {"epoch": epoch, "train_loss": total / samples, "validation": validation, "optical_alpha": float(optical.fusion.alpha.detach())}
        history.append(row); print(json.dumps(row), flush=True)
        if validation["latent_mse"] < best:
            best = validation["latent_mse"]; best_epoch = epoch
            torch.save({
                "schema_version": 1, "student_widths": list(widths), "epoch": epoch,
                "model_config": asdict(model_config), "training_config": asdict(training_config),
                "unet": {key: value.detach().half().cpu() for key, value in unet.state_dict().items()},
                "adapter": {key: value.detach().half().cpu() for key, value in adapter.state_dict().items()},
                "design_router": {key: value.detach().half().cpu() for key, value in router.state_dict().items()},
                "validation": validation, "qwen_layers": 2,
            }, output_dir / "best_model.pt")
        _sample_grid(unet=unet, adapter=adapter, vae=vae, sigma=sigma, latent_dataset=datasets["val"], raw_dataset=raw_val, output=output_dir / "samples" / f"epoch_{epoch:03d}.jpg", device=device, training=training_config, seed=seed + epoch)
    best_payload = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False, mmap=True)
    unet.load_state_dict(best_payload["unet"]); adapter.load_state_dict(best_payload["adapter"]); router.load_state_dict(best_payload["design_router"])
    test = _evaluate(unet, adapter, router, loaders["test"], sigma, device, training_config, seeded_noise=morphology_task)
    qwen = torch.load(instruction_cache, map_location="cpu", weights_only=False)["meta"]["qwen_pruning"]
    parameters = counted_student_parameters(
        unet=unet, adapter=adapter, router=router,
        qwen_counted=qwen["counted_text_encoder_parameters"],
        vae_encoder=sum(p.numel() for p in vae.encoder.parameters()) + sum(p.numel() for p in vae.quant_conv.parameters()),
        vae_decoder=sum(p.numel() for p in vae.decoder.parameters()) + sum(p.numel() for p in vae.post_quant_conv.parameters()),
    )
    report = {
        "schema_version": 1, "task": task_name,
        "categories": list(raw_val.supported_categories), "chairs_used": False,
        "best_epoch": best_epoch, "initial_validation": initial, "test": test,
        "parameters": parameters, "attention_pruning": pruning, "history": history,
        "training_seconds": time.perf_counter() - started, "gan_used": False,
        "hard_pixel_composite_at_inference": False, "inference_iterations": 1,
        "warm_start_checkpoint": str(warm_start_checkpoint),
        "warm_start_copy": warm_copy,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del unet, adapter, router, vae, optimizer, warm, best_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


__all__ = [
    "RedesignLatentDataset", "TextDesignRouter", "cache_redesign_latents",
    "train_redesign_model", "load_scene_config",
]
