"""Paired training for geometry-preserving chair style transfer."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .chair_style_transfer import (
    ChairStyleConfig, ChairStyleDiscriminator, ChairStyleTransfer, STYLE_NAMES,
    architecture_report, config_payload,
)
from .cleanrender_training import CleanRenderDataset
from .dataset import validate_split_contract


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def foreground_mask(reference: torch.Tensor) -> torch.Tensor:
    return ChairStyleTransfer.foreground_mask(reference)


def apply_product_style(reference: torch.Tensor, style_index: torch.Tensor) -> torch.Tensor:
    """Create paired targets by recolouring only object pixels while preserving luminance."""
    rgb = reference.float().add(1).mul(0.5)
    luminance = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    dark = reference.new_tensor([
        [0.20, 0.07, 0.025], [0.035, 0.038, 0.045], [0.58, 0.56, 0.48], [0.025, 0.09, 0.28],
    ])[style_index][:, :, None, None]
    light = reference.new_tensor([
        [0.82, 0.52, 0.24], [0.38, 0.39, 0.42], [0.98, 0.97, 0.89], [0.22, 0.58, 0.94],
    ])[style_index][:, :, None, None]
    styled = dark + (light - dark) * luminance.pow(0.82)
    styled = 0.92 * styled + 0.08 * rgb
    mask = foreground_mask(reference)
    output = rgb * (1 - mask) + styled * mask
    return output.mul(2).sub(1).clamp(-1, 1)


class ChairStyleDataset(Dataset[dict[str, Any]]):
    def __init__(self, data_dir: Path, split: str, image_size: int, style_cache: Path) -> None:
        self.base = CleanRenderDataset(data_dir, split, image_size)
        cache = torch.load(style_cache, map_location="cpu", weights_only=False)
        if cache["style_names"] != list(STYLE_NAMES):
            raise ValueError("Style cache ordering is incompatible")
        self.style_text = cache["text"].float()

    def __len__(self) -> int:
        return len(self.base) * len(STYLE_NAMES)

    def __getitem__(self, index: int) -> dict[str, Any]:
        style = index % len(STYLE_NAMES)
        item = self.base[index // len(STYLE_NAMES)]
        item = {**item, "style_index": style, "style_name": STYLE_NAMES[style], "style_text": self.style_text[style]}
        return item


def _edge(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(1, keepdim=True)
    kx = value.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    return torch.sqrt(F.conv2d(gray, kx, padding=1).square() + F.conv2d(gray, ky, padding=1).square() + 1e-6)


def _loader(data_dir: Path, split: str, config: ChairStyleConfig, style_cache: Path, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ChairStyleDataset(data_dir, split, config.image_size, style_cache),
        batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=config.num_workers > 0,
        drop_last=shuffle, generator=torch.Generator().manual_seed(seed),
    )


def _set_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def _update_ema(ema: torch.nn.Module, model: torch.nn.Module, beta: float = 0.995) -> None:
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - beta)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def _load_electronic_warmstart(model: ChairStyleTransfer, discriminator: ChairStyleDiscriminator, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["generator_ema"]
    report = model.load_state_dict(state, strict=False)
    allowed = ("bottleneck.optical", "bottleneck.fusion", "bottleneck.expert_gate", "bottleneck.global_gate")
    unexpected = list(report.unexpected_keys)
    invalid_missing = [name for name in report.missing_keys if not name.startswith(allowed)]
    if unexpected or invalid_missing:
        raise RuntimeError(f"Unsafe electronic warmstart: missing={invalid_missing}, unexpected={unexpected}")
    if "discriminator" in payload:
        discriminator.load_state_dict(payload["discriminator"])
    return {"source": str(path), "epoch": int(payload["epoch"]), "missing_optical_keys": list(report.missing_keys)}


def _configure_optical_only(model: ChairStyleTransfer) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    prefixes = ("bottleneck.optical", "bottleneck.fusion", "bottleneck.expert_gate", "bottleneck.global_gate")
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True); selected.append(parameter)
    if not selected:
        raise RuntimeError("No optical parameters selected")
    return selected


def _losses(output: torch.Tensor, target: torch.Tensor, reference: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    weight = 1 + 6 * mask
    reconstruction = ((output.float() - target.float()).abs() * weight).sum() / (weight.sum() * 3)
    target_edge, output_edge, reference_edge = _edge(target), _edge(output), _edge(reference)
    edge = (output_edge - target_edge).abs().mean()
    edge_support = reference_edge.div(0.25).clamp(0, 1)
    identity_edge = ((output_edge - reference_edge).abs() * edge_support).sum() / edge_support.sum().clamp_min(1)
    background = ((output.float() - reference.float()).abs() * (1 - mask)).mean()
    residual = (delta.float().abs() * mask).mean()
    return dict(reconstruction=reconstruction, edge=edge, identity_edge=identity_edge, background=background, residual=residual)


@torch.no_grad()
def _evaluate(model: ChairStyleTransfer, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); totals = {"l1": 0.0, "edge_drift": 0.0, "background_drift": 0.0}; samples = 0
    for batch in loader:
        reference = batch["image"].to(device); style = batch["style_text"].to(device)
        indices = batch["style_index"].to(device)
        target = apply_product_style(reference, indices)
        output, _, mask = model.forward_with_aux(reference, style)
        count = len(reference); samples += count
        totals["l1"] += float(F.l1_loss(output.float(), target.float())) * count
        totals["edge_drift"] += float((_edge(output) - _edge(reference)).abs().mean()) * count
        totals["background_drift"] += float(((output-reference).abs() * (1-mask)).mean()) * count
    return {name: value / samples for name, value in totals.items()}


def _to_pil(value: torch.Tensor) -> Image.Image:
    array = value.detach().float().add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


@torch.no_grad()
def save_sample(model: ChairStyleTransfer, loader: DataLoader, output: Path, device: torch.device, rows: int = 4) -> None:
    model.eval(); batch = next(iter(loader))
    reference = batch["image"][:rows].to(device); style = batch["style_text"][:rows].to(device)
    indices = batch["style_index"][:rows].to(device)
    target = apply_product_style(reference, indices); prediction = model(reference, style)
    tile, header = 128, 24
    canvas = Image.new("RGB", (3 * tile, rows * (tile + header)), "white"); draw = ImageDraw.Draw(canvas)
    for row in range(rows):
        for column, (name, tensor) in enumerate((
            ("reference", reference[row]), ("paired target", target[row]), ("prediction", prediction[row]),
        )):
            x, y = column * tile, row * (tile + header)
            draw.text((x + 4, y + 4), name, fill="black"); canvas.paste(_to_pil(tensor), (x, y + header))
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output)


def train_chair_style_transfer(
    data_dir: Path, style_cache: Path, run_dir: Path, config: ChairStyleConfig,
    device: torch.device, *, seed: int = 91, initialize_checkpoint: Path | None = None,
) -> dict[str, Any]:
    _seed_everything(seed); validate_split_contract(data_dir); run_dir.mkdir(parents=True, exist_ok=False)
    train_loader = _loader(data_dir, "train", config, style_cache, seed, True)
    val_loader = _loader(data_dir, "val", config, style_cache, seed + 1, False)
    model = ChairStyleTransfer(config).to(device)
    discriminator = ChairStyleDiscriminator(config.text_dim).to(device)
    warmstart = None
    if initialize_checkpoint is not None:
        warmstart = _load_electronic_warmstart(model, discriminator, initialize_checkpoint)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    generator_parameters = _configure_optical_only(model) if config.optical and warmstart else list(model.parameters())
    regular = [p for name, p in model.named_parameters() if p.requires_grad and "optical" not in name]
    phases = [p for name, p in model.named_parameters() if p.requires_grad and "optical" in name]
    groups = []
    if regular: groups.append({"params": regular, "lr": config.learning_rate})
    if phases: groups.append({"params": phases, "lr": config.phase_learning_rate})
    generator_optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay, betas=(0.5, 0.99))
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.99))
    architecture = architecture_report(model, discriminator)
    architecture["optimized_generator_parameters"] = sum(p.numel() for p in generator_parameters)
    (run_dir / "architecture.json").write_text(json.dumps(architecture, indent=2) + "\n", encoding="utf-8")
    history = []; best = float("inf"); best_epoch = 0
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda")
    for epoch in range(1, config.epochs + 1):
        model.train(); discriminator.train(); totals: dict[str, float] = {}; samples = 0
        adversarial_scale = config.adversarial_weight * min(1.0, epoch / max(1, config.adversarial_warmup_epochs))
        for batch in train_loader:
            reference = batch["image"].to(device, non_blocking=True)
            style = batch["style_text"].to(device, non_blocking=True)
            indices = batch["style_index"].to(device, non_blocking=True)
            target = apply_product_style(reference, indices)
            if torch.rand(()) < 0.5:
                reference, target = reference.flip(-1), target.flip(-1)
            with torch.autocast(**autocast):
                output, delta, mask = model.forward_with_aux(reference, style)
            _set_grad(discriminator, True); discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**autocast):
                real_score = discriminator(target, style); fake_score = discriminator(output.detach(), style)
                discriminator_loss = F.relu(1-real_score.float()).mean() + F.relu(1+fake_score.float()).mean()
            discriminator_loss.backward(); discriminator_optimizer.step()
            _set_grad(discriminator, False); generator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**autocast):
                fake_score = discriminator(output, style); parts = _losses(output, target, reference, delta, mask)
                generator_loss = (
                    config.reconstruction_weight * parts["reconstruction"] + config.edge_weight * parts["edge"]
                    + config.identity_edge_weight * parts["identity_edge"] + config.background_weight * parts["background"]
                    + config.residual_weight * parts["residual"] - adversarial_scale * fake_score.float().mean()
                )
            generator_loss.backward(); torch.nn.utils.clip_grad_norm_(generator_parameters, 5.0); generator_optimizer.step()
            _update_ema(ema, model); _set_grad(discriminator, True)
            count = len(reference); samples += count
            values = {"generator": float(generator_loss.detach()), "discriminator": float(discriminator_loss.detach()), **{k: float(v.detach()) for k,v in parts.items()}}
            for name, value in values.items(): totals[name] = totals.get(name, 0.0) + value * count
        train_metrics = {name: value / samples for name, value in totals.items()}
        validation = _evaluate(ema, val_loader, device); score = validation["l1"] + 0.25 * validation["edge_drift"]
        record = {"epoch": epoch, "train": train_metrics, "validation": validation, "selection_score": score}
        history.append(record); print(json.dumps(record), flush=True)
        payload = {
            "epoch": epoch, "generator": model.state_dict(), "generator_ema": ema.state_dict(),
            "discriminator": discriminator.state_dict(), "config": config_payload(config),
            "architecture": architecture, "validation": validation, "warmstart": warmstart,
            "style_names": list(STYLE_NAMES),
        }
        torch.save(payload, run_dir / "latest_checkpoint.pt")
        if score < best:
            best, best_epoch = score, epoch; torch.save(payload, run_dir / "best_checkpoint.pt")
        if epoch == 1 or epoch % config.sample_every_epochs == 0 or epoch == config.epochs:
            save_sample(ema, val_loader, run_dir / f"sample_epoch_{epoch:04d}.png", device)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {"best_epoch": best_epoch, "best_selection_score": best, "architecture": architecture, "warmstart": warmstart}
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = ["ChairStyleDataset", "apply_product_style", "save_sample", "train_chair_style_transfer"]
