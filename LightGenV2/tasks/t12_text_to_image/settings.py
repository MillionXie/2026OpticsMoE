"""Configuration loading and validation for T12."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


TASK_DIR = Path(__file__).resolve().parent
VARIANTS = {"lightgen_parallel", "qwen_vae_baseline"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        elif key != "base_config":
            result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.get("base_config")
    if parent:
        base_path = (path.parent / str(parent)).resolve()
        return _merge(_read(base_path), raw)
    return raw


def _at(raw: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = raw
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass
class Settings:
    config_path: Path
    variant: str
    seed: int
    data_dir: Path
    output_dir: Path
    image_size: int
    latent_channels: int
    latent_size: int
    text_dim: int
    width: int
    token_grid: int
    style_dim: int
    electronic_depth: int
    decoder_depth: int
    optical_backend: str
    fusion_alpha_initial: float
    fusion_alpha_minimum: float
    fusion_alpha_maximum: float
    fusion_rms_epsilon: float
    qwen_model: str
    qwen_checkpoint: Path | None
    vae_model: str
    vae_checkpoint: Path | None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    kl_weight: float
    kl_warmup_epochs: int
    free_bits: float
    num_workers: int
    amp: bool
    freeze_warmstarted_base: bool
    adversarial_enabled: bool
    discriminator_width: int
    discriminator_learning_rate: float
    adversarial_start_epoch: int
    adversarial_weight: float
    prior_adversarial_weight: float
    feature_matching_weight: float
    pixel_reconstruction_weight: float
    prior_latent_delta_weight: float

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown T12 variant: {self.variant}")
        if self.image_size != self.latent_size * 8:
            raise ValueError("T12 assumes a VAE downsampling factor of exactly 8")
        if self.latent_size != self.token_grid * 2:
            raise ValueError("The latent head performs exactly one 2x upsampling")
        if self.width % 8:
            raise ValueError("model.width must be divisible by 8")
        if self.decoder_depth < 0:
            raise ValueError("model.decoder_depth must be non-negative")
        if not 0 <= self.fusion_alpha_minimum < self.fusion_alpha_initial < self.fusion_alpha_maximum <= 1:
            raise ValueError("Fusion alpha must start strictly inside its configured range")
        if self.variant == "qwen_vae_baseline" and self.optical_backend != "none":
            raise ValueError("The Qwen+VAE baseline must not instantiate an optical path")
        if self.variant == "lightgen_parallel" and self.optical_backend == "none":
            raise ValueError("The LightGen variant requires an optical path")
        if self.optical_backend not in {"none", "compact_fft", "audited_dc20"}:
            raise ValueError(f"Unknown optical backend: {self.optical_backend}")
        if self.adversarial_enabled:
            if self.discriminator_width <= 0 or self.adversarial_start_epoch < 0:
                raise ValueError("Invalid adversarial discriminator settings")
            if min(
                self.adversarial_weight,
                self.prior_adversarial_weight,
                self.feature_matching_weight,
                self.pixel_reconstruction_weight,
                self.prior_latent_delta_weight,
            ) < 0:
                raise ValueError("Adversarial loss weights must be non-negative")

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "feature_cache"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def load_settings(path: str | Path) -> Settings:
    config = Path(path).expanduser().resolve()
    raw = _read(config)
    base = config.parent
    qwen_checkpoint = _at(raw, "qwen.checkpoint")
    vae_checkpoint = _at(raw, "vae.checkpoint")
    settings = Settings(
        config_path=config,
        variant=str(_at(raw, "model.variant", "lightgen_parallel")),
        seed=int(_at(raw, "seed", 42)),
        data_dir=_path(_at(raw, "dataset.data_dir"), base),
        output_dir=_path(_at(raw, "output_dir"), base),
        image_size=int(_at(raw, "dataset.image_size", 256)),
        latent_channels=int(_at(raw, "vae.latent_channels", 4)),
        latent_size=int(_at(raw, "vae.latent_size", 32)),
        text_dim=int(_at(raw, "qwen.hidden_size", 2048)),
        width=int(_at(raw, "model.width", 192)),
        token_grid=int(_at(raw, "model.token_grid", 16)),
        style_dim=int(_at(raw, "model.style_dim", 256)),
        electronic_depth=int(_at(raw, "model.electronic_depth", 2)),
        decoder_depth=int(_at(raw, "model.decoder_depth", 0)),
        optical_backend=str(_at(raw, "model.optical_backend", "compact_fft")),
        fusion_alpha_initial=float(_at(raw, "fusion.alpha_initial", 0.40)),
        fusion_alpha_minimum=float(_at(raw, "fusion.alpha_minimum", 0.05)),
        fusion_alpha_maximum=float(_at(raw, "fusion.alpha_maximum", 0.95)),
        fusion_rms_epsilon=float(_at(raw, "fusion.rms_epsilon", 1e-6)),
        qwen_model=str(_at(raw, "qwen.model", "Qwen/Qwen3-VL-2B-Instruct")),
        qwen_checkpoint=None if qwen_checkpoint in (None, "") else _path(qwen_checkpoint, base),
        vae_model=str(_at(raw, "vae.model", "stabilityai/sd-vae-ft-mse")),
        vae_checkpoint=None if vae_checkpoint in (None, "") else _path(vae_checkpoint, base),
        batch_size=int(_at(raw, "training.batch_size", 32)),
        epochs=int(_at(raw, "training.epochs", 80)),
        learning_rate=float(_at(raw, "training.learning_rate", 2e-4)),
        weight_decay=float(_at(raw, "training.weight_decay", 0.01)),
        kl_weight=float(_at(raw, "loss.kl_weight", 2e-3)),
        kl_warmup_epochs=int(_at(raw, "loss.kl_warmup_epochs", 15)),
        free_bits=float(_at(raw, "loss.free_bits", 0.02)),
        num_workers=int(_at(raw, "training.num_workers", 4)),
        amp=bool(_at(raw, "training.amp", True)),
        freeze_warmstarted_base=bool(_at(raw, "training.freeze_warmstarted_base", False)),
        adversarial_enabled=bool(_at(raw, "adversarial.enabled", False)),
        discriminator_width=int(_at(raw, "adversarial.discriminator_width", 48)),
        discriminator_learning_rate=float(_at(raw, "adversarial.learning_rate", 1e-4)),
        adversarial_start_epoch=int(_at(raw, "adversarial.start_epoch", 5)),
        adversarial_weight=float(_at(raw, "adversarial.posterior_weight", 0.05)),
        prior_adversarial_weight=float(_at(raw, "adversarial.prior_weight", 0.05)),
        feature_matching_weight=float(_at(raw, "adversarial.feature_matching_weight", 0.10)),
        pixel_reconstruction_weight=float(_at(raw, "adversarial.pixel_reconstruction_weight", 0.10)),
        prior_latent_delta_weight=float(_at(raw, "adversarial.prior_latent_delta_weight", 0.0)),
    )
    settings.validate()
    return settings


__all__ = ["Settings", "TASK_DIR", "VARIANTS", "load_settings"]
