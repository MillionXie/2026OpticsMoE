"""Training loop for the paired LightGen and Qwen+VAE baseline profiles."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import CachedLatentDataset
from .feature_cache import _model_source
from .losses import conditional_vae_loss
from .modeling import PatchDiscriminator, TextConditionedVAE, build_model
from .settings import Settings


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(settings: Settings, split: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        CachedLatentDataset(settings.cache_dir / f"{split}.pt"),
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        generator=torch.Generator().manual_seed(settings.seed),
    )


def _beta(settings: Settings, epoch: int) -> float:
    if settings.kl_warmup_epochs <= 0:
        return settings.kl_weight
    return settings.kl_weight * min(1.0, float(epoch + 1) / settings.kl_warmup_epochs)


def _epoch(
    model: TextConditionedVAE,
    loader: DataLoader,
    device: torch.device,
    settings: Settings,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            text = batch["text"].to(device, non_blocking=True)
            latent = batch["latent"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.amp and device.type == "cuda"):
                output = model.forward_train(text, latent)
                loss, metrics = conditional_vae_loss(
                    output, latent, kl_weight=_beta(settings, epoch), free_bits=settings.free_bits
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = len(text)
            samples += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / samples for key, value in totals.items()}


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _load_warmstart(
    model: TextConditionedVAE,
    settings: Settings,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Load model weights while allowing decoder residual blocks to be inserted.

    A depth-zero latent head stores its two suffix convolutions at indices 4 and
    6. Deeper heads insert residual blocks before that suffix, so these two
    tensors need deterministic index translation. All other parameters must
    either match exactly or belong to newly inserted decoder blocks.
    """

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != settings.variant:
        raise ValueError(
            f"Warm-start variant {payload.get('variant')!r} does not match {settings.variant!r}"
        )
    source = payload.get("model")
    if not isinstance(source, dict):
        raise ValueError(f"Warm-start checkpoint has no model state: {path}")
    source_settings = payload.get("settings", {})
    source_architecture = payload.get("architecture", {})
    source_depth = int(
        source_settings.get(
            "decoder_depth", source_architecture.get("decoder_residual_depth", 0)
        )
    )
    target_depth = int(settings.decoder_depth)
    target = model.state_dict()
    translated: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    remapped: dict[str, str] = {}
    suffix_indices = {
        4 + source_depth: 4 + target_depth,
        6 + source_depth: 6 + target_depth,
    }
    for key, value in source.items():
        candidate = key
        for source_index, target_index in suffix_indices.items():
            prefix = f"generator.head.net.{source_index}."
            if source_depth != target_depth and key.startswith(prefix):
                candidate = f"generator.head.net.{target_index}." + key[len(prefix):]
                remapped[key] = candidate
                break
        if candidate in target and target[candidate].shape == value.shape:
            translated[candidate] = value
        else:
            skipped.append(key)
    incompatible = model.load_state_dict(translated, strict=False)
    allowed_missing_prefixes = tuple(
        f"generator.head.net.{index}." for index in range(4, 4 + target_depth)
    )
    illegal_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if skipped or incompatible.unexpected_keys or illegal_missing:
        raise ValueError(
            "Incompatible warm-start checkpoint: "
            f"skipped={skipped}, unexpected={incompatible.unexpected_keys}, "
            f"illegal_missing={illegal_missing}"
        )
    return {
        "path": str(path),
        "source_epoch": payload.get("epoch"),
        "source_decoder_depth": source_depth,
        "target_decoder_depth": target_depth,
        "copied_tensors": len(translated),
        "remapped_tensors": remapped,
        "new_tensors": list(incompatible.missing_keys),
    }


def _discriminator_loss(
    real: torch.Tensor,
    posterior: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    return (
        F.relu(1.0 - real.float()).mean()
        + 0.5 * F.relu(1.0 + posterior.float()).mean()
        + 0.5 * F.relu(1.0 + prior.float()).mean()
    )


def _adversarial_epoch(
    model: TextConditionedVAE,
    discriminator: PatchDiscriminator,
    vae: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    settings: Settings,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train posterior reconstruction and random-prior realism without an inference loop."""

    model.train()
    discriminator.train()
    totals: dict[str, float] = {}
    samples = 0
    adversarial_active = epoch >= settings.adversarial_start_epoch
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    for batch in loader:
        text = batch["text"].to(device, non_blocking=True)
        target = batch["latent"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.amp):
            output = model.forward_train(text, target)
            base_loss, metrics = conditional_vae_loss(
                output, target, kl_weight=_beta(settings, epoch), free_bits=settings.free_bits
            )
            with torch.no_grad():
                real_image = vae.decode(target / scaling).sample
            posterior_image = vae.decode(output.predicted_latent / scaling).sample
            pixel_loss = F.l1_loss(posterior_image.float(), real_image.float())

        discriminator_loss = torch.zeros((), device=device)
        generator_adversarial = torch.zeros((), device=device)
        feature_matching = torch.zeros((), device=device)
        prior_adversarial = torch.zeros((), device=device)
        if adversarial_active:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.amp):
                prior_style = torch.randn_like(output.style)
                prior_latent = model.generator(text, prior_style)
                prior_image = vae.decode(prior_latent / scaling).sample

            _set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.amp):
                real_logits, _ = discriminator(real_image.detach())
                posterior_logits, _ = discriminator(posterior_image.detach())
                prior_logits, _ = discriminator(prior_image.detach())
                discriminator_loss = _discriminator_loss(real_logits, posterior_logits, prior_logits)
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            discriminator_optimizer.step()

            _set_requires_grad(discriminator, False)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.amp):
                posterior_logits, posterior_features = discriminator(posterior_image)
                prior_logits, _ = discriminator(prior_image)
                with torch.no_grad():
                    _, real_features = discriminator(real_image)
                generator_adversarial = -posterior_logits.float().mean()
                prior_adversarial = -prior_logits.float().mean()
                feature_matching = sum(
                    F.l1_loss(fake.float(), real.float())
                    for fake, real in zip(posterior_features, real_features)
                ) / len(real_features)

        total = (
            base_loss
            + settings.pixel_reconstruction_weight * pixel_loss
            + settings.adversarial_weight * generator_adversarial
            + settings.prior_adversarial_weight * prior_adversarial
            + settings.feature_matching_weight * feature_matching
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _set_requires_grad(discriminator, True)

        count = len(text)
        samples += count
        metrics.update({
            "loss": float(total.detach()),
            "base_loss": float(base_loss.detach()),
            "pixel_l1": float(pixel_loss.detach()),
            "generator_adversarial": float(generator_adversarial.detach()),
            "prior_adversarial": float(prior_adversarial.detach()),
            "feature_matching": float(feature_matching.detach()),
            "discriminator_loss": float(discriminator_loss.detach()),
            "adversarial_active": float(adversarial_active),
        })
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / samples for key, value in totals.items()}


def train(
    settings: Settings,
    device: torch.device,
    *,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [settings.output_dir / name for name in ("best_checkpoint.pt", "last_checkpoint.pt")]
    if any(path.exists() for path in existing):
        raise FileExistsError("Training checkpoints already exist; choose a new --run-dir to preserve the run")
    model = build_model(settings, device)
    warmstart = _load_warmstart(model, settings, init_checkpoint) if init_checkpoint else None
    if warmstart:
        print(json.dumps({"warmstart": warmstart}), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    discriminator = None
    discriminator_optimizer = None
    vae = None
    if settings.adversarial_enabled:
        from diffusers import AutoencoderKL

        vae_source, vae_local = _model_source(settings.vae_checkpoint, settings.vae_model)
        vae = AutoencoderKL.from_pretrained(
            vae_source, local_files_only=vae_local, torch_dtype=torch.float32
        ).to(device).eval().requires_grad_(False)
        discriminator = PatchDiscriminator(settings.discriminator_width).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=settings.discriminator_learning_rate, betas=(0.0, 0.99)
        )
    train_loader = _loader(settings, "train", True)
    val_loader = _loader(settings, "val", False)
    best = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(settings.epochs):
        if settings.adversarial_enabled:
            assert discriminator is not None and discriminator_optimizer is not None and vae is not None
            train_metrics = _adversarial_epoch(
                model, discriminator, vae, train_loader, device, settings, epoch,
                optimizer, discriminator_optimizer,
            )
        else:
            train_metrics = _epoch(model, train_loader, device, settings, epoch, optimizer)
        val_metrics = _epoch(model, val_loader, device, settings, epoch, None)
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch + 1,
            "variant": settings.variant,
            "settings": settings.to_dict(),
            "architecture": model.architecture_report(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val": val_metrics,
            "warmstart": warmstart,
        }
        if discriminator is not None and discriminator_optimizer is not None:
            payload["discriminator"] = discriminator.state_dict()
            payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        torch.save(payload, settings.output_dir / "last_checkpoint.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(payload, settings.output_dir / "best_checkpoint.pt")
    (settings.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    report = {
        "best_val_loss": best,
        "epochs": settings.epochs,
        "adversarial_enabled": settings.adversarial_enabled,
        "architecture": model.architecture_report(),
        "warmstart": warmstart,
    }
    (settings.output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["_load_warmstart", "seed_everything", "train"]
