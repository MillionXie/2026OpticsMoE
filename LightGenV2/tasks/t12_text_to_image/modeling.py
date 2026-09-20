"""Single-pass conditional VAE generators for the LightGen and baseline rows.

The central contract is explicit: at each hybrid stage the electronic and
optical branches receive the same input tensor.  Only their outputs are fused.
There is no E->O cascade and no recurrent/denoising loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .settings import Settings


def _range_logit(initial: float, minimum: float, maximum: float) -> torch.Tensor:
    position = (float(initial) - float(minimum)) / (float(maximum) - float(minimum))
    return torch.logit(torch.tensor(position))


class ScaleMatchedFusion(nn.Module):
    """Detached-RMS convex fusion used by the existing LightGenV2 tasks."""

    def __init__(self, initial: float, minimum: float, maximum: float, epsilon: float) -> None:
        super().__init__()
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.epsilon = float(epsilon)
        self.raw_alpha = nn.Parameter(_range_logit(initial, minimum, maximum))
        self.last_diagnostics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(self.raw_alpha)

    def _rms(self, value: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, value.ndim))
        return value.float().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(self.epsilon)

    def forward(self, electronic: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        if electronic.shape != optical.shape:
            raise ValueError("Parallel branches must have identical output shapes")
        electronic32, optical32 = electronic.float(), optical.float()
        re, ro = self._rms(electronic32).detach(), self._rms(optical32).detach()
        mixture = (1.0 - self.alpha) * electronic32 / re + self.alpha * optical32 / ro
        fused = re * mixture / self._rms(mixture).detach()
        with torch.no_grad():
            self.last_diagnostics = {
                "alpha": float(self.alpha),
                "electronic_rms": float(re.mean()),
                "optical_rms": float(ro.mean()),
                "fused_to_electronic_rms": float((self._rms(fused) / re).mean()),
            }
        return fused.to(electronic.dtype)


class ConditionedElectronicResidual(nn.Module):
    """Spatial residual mixer with AdaLN-style text/style conditioning."""

    def __init__(self, width: int, condition_dim: int, grid: int) -> None:
        super().__init__()
        self.width = int(width)
        self.grid = int(grid)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition = nn.Linear(condition_dim, 2 * width)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, width, 1)
        )
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        if count != self.grid**2 or width != self.width:
            raise ValueError(f"Expected [B,{self.grid**2},{self.width}], got {tuple(tokens.shape)}")
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        value = self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        value = value.transpose(1, 2).reshape(batch, width, self.grid, self.grid)
        value = self.pointwise(self.depthwise(value))
        value = value.flatten(2).transpose(1, 2)
        return tokens + torch.sigmoid(self.gate) * value


class CompactFourierOptics(nn.Module):
    """Small differentiable phase-only simulator for development and CI.

    It preserves the Router -> Top-2 experts -> global phase semantics.  Formal
    hardware runs will replace this backend with the audited 224/478/518 DC20
    path; results from this compact backend must be labelled simulation only.
    """

    def __init__(self, width: int, grid: int, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.width, self.grid = int(width), int(grid)
        self.experts, self.top_k = int(experts), int(top_k)
        self.norm = nn.LayerNorm(width)
        self.amplitude = nn.Linear(width, 1)
        self.readout = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.router_phase = nn.Parameter(torch.randn(experts, grid, grid) * 0.05)
        self.expert_phase = nn.Parameter(torch.randn(experts, grid, grid) * 0.05)
        self.global_phase = nn.Parameter(torch.randn(grid, grid) * 0.05)
        self.last_routing: dict[str, torch.Tensor] = {}

    @staticmethod
    def _propagate(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        field = torch.complex(amplitude, torch.zeros_like(amplitude)) * torch.exp(
            torch.complex(torch.zeros_like(phase), phase)
        )
        detector = torch.fft.fftshift(torch.fft.fft2(field, norm="ortho"), dim=(-2, -1))
        intensity = detector.abs().square()
        return intensity / intensity.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    def _amplitude(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        value = F.softplus(self.amplitude(self.norm(tokens))).squeeze(-1)
        return value.reshape(batch, self.grid, self.grid)

    def expert(self, tokens: torch.Tensor) -> torch.Tensor:
        amplitude = self._amplitude(tokens)
        router_intensity = self._propagate(
            amplitude[:, None], self.router_phase[None]
        )
        centre = max(1, self.grid // 4)
        start = (self.grid - centre) // 2
        scores = router_intensity[..., start : start + centre, start : start + centre].mean((-2, -1))
        probabilities = scores.softmax(dim=-1)
        indices = probabilities.topk(self.top_k, dim=-1).indices
        mask = torch.zeros_like(probabilities).scatter(1, indices, 1.0)
        weights = probabilities * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_intensity = self._propagate(amplitude[:, None], self.expert_phase[None])
        mixed = (expert_intensity * weights[:, :, None, None]).sum(dim=1)
        output = self.readout(mixed.flatten(1).unsqueeze(-1))
        self.last_routing = {
            "probabilities": probabilities,
            "selected_mask": mask.bool(),
            "weights": weights,
        }
        return output

    def global_block(self, tokens: torch.Tensor) -> torch.Tensor:
        amplitude = self._amplitude(tokens)
        intensity = self._propagate(amplitude, self.global_phase)
        return self.readout(intensity.flatten(1).unsqueeze(-1))


class ParallelHybridBackbone(nn.Module):
    """Two hybrid stages; E and O are parallel within every stage."""

    def __init__(self, settings: Settings, condition_dim: int) -> None:
        super().__init__()
        if settings.optical_backend != "compact_fft":
            raise ValueError(
                "This initial T12 implementation supports compact_fft for simulation; "
                "the audited DC20 adapter is a separate formal-run milestone"
            )
        self.electronic1 = ConditionedElectronicResidual(settings.width, condition_dim, settings.token_grid)
        self.electronic2 = ConditionedElectronicResidual(settings.width, condition_dim, settings.token_grid)
        self.optical = CompactFourierOptics(settings.width, settings.token_grid)
        fusion_args = (
            settings.fusion_alpha_initial,
            settings.fusion_alpha_minimum,
            settings.fusion_alpha_maximum,
            settings.fusion_rms_epsilon,
        )
        self.fusion1 = ScaleMatchedFusion(*fusion_args)
        self.fusion2 = ScaleMatchedFusion(*fusion_args)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        stage1_input = tokens
        electronic1 = self.electronic1(stage1_input, condition)
        optical1 = self.optical.expert(stage1_input)
        stage2_input = self.fusion1(electronic1, optical1)
        electronic2 = self.electronic2(stage2_input, condition)
        optical2 = self.optical.global_block(stage2_input)
        return self.fusion2(electronic2, optical2)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "fusion1": self.fusion1.last_diagnostics,
            "fusion2": self.fusion2.last_diagnostics,
            "route_mean": self.optical.last_routing.get("weights", torch.empty(0)).detach().mean(0).cpu().tolist(),
        }


class AuditedDC20Backbone(nn.Module):
    """Adapter around LightGenV2's existing 224/478/518 DC20 hybrid core.

    ``BalancedVisionCore`` already implements the required graph twice:
    electronic residual and optical expert/global propagation consume the same
    stage input, then detached-RMS scale-matched fusion produces the next stage.
    T12 only supplies conditioned spatial tokens and reshapes the packed output.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        from LightGenV2.tasks.t01_object_retrieval.settings import load_settings as load_optical_settings
        from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import (
            BalancedVisionCore,
        )

        config = (
            Path(__file__).resolve().parents[1]
            / "t01_object_retrieval/configs/moe_optical_router_scale_matched_dc20.yaml"
        )
        optical = load_optical_settings(config)
        optical.fusion_mode = "scale_matched_convex"
        optical.fusion_alpha_initial = settings.fusion_alpha_initial
        optical.fusion_alpha_min = settings.fusion_alpha_minimum
        optical.fusion_alpha_max = settings.fusion_alpha_maximum
        optical.fusion_rms_epsilon = settings.fusion_rms_epsilon
        self.grid = settings.token_grid
        self.core = BalancedVisionCore(settings.width, self.grid**2, optical)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        del condition  # Text/style are already injected into every input token.
        batch, count, width = tokens.shape
        packed, _ = self.core.forward_groups(
            list(tokens.unbind(0)),
            causal=False,
            spatial_shapes=[(1, self.grid, self.grid)] * batch,
        )
        return packed.reshape(batch, count, width)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "fusion": self.core.fusion_diagnostics(),
            "route_mean": self.core.last_routing.get("weights", torch.empty(0)).detach().mean(0).cpu().tolist(),
        }


class ElectronicBackbone(nn.Module):
    """Qwen+VAE baseline: same token contract and electronic residual family."""

    def __init__(self, settings: Settings, condition_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedElectronicResidual(settings.width, condition_dim, settings.token_grid)
            for _ in range(settings.electronic_depth)
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, condition)
        return tokens


class DecoderResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
        )
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.sigmoid(self.gate) * self.net(value)


class LatentHead(nn.Module):
    def __init__(self, width: int, channels: int, depth: int = 0) -> None:
        super().__init__()
        prefix: list[nn.Module] = [
            nn.Conv2d(width, width * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        ]
        residuals = [DecoderResidualBlock(width) for _ in range(int(depth))]
        suffix: list[nn.Module] = [
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, channels, 1),
        ]
        self.net = nn.Sequential(*prefix, *residuals, *suffix)

    def forward(self, tokens: torch.Tensor, grid: int) -> torch.Tensor:
        value = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)
        return self.net(value)


class ConditionalLatentGenerator(nn.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        width, grid = settings.width, settings.token_grid
        self.text_projection = nn.Sequential(nn.LayerNorm(settings.text_dim), nn.Linear(settings.text_dim, width))
        self.style_projection = nn.Linear(settings.style_dim, width)
        self.condition_projection = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.learned_tokens = nn.Parameter(torch.randn(1, grid**2, width) / math.sqrt(width))
        self.position = nn.Parameter(torch.randn(1, grid**2, width) * 0.01)
        self.backbone: nn.Module
        if settings.variant == "lightgen_parallel":
            if settings.optical_backend == "compact_fft":
                self.backbone = ParallelHybridBackbone(settings, width)
            elif settings.optical_backend == "audited_dc20":
                self.backbone = AuditedDC20Backbone(settings)
            else:
                raise ValueError(f"Unknown optical backend {settings.optical_backend!r}")
        else:
            self.backbone = ElectronicBackbone(settings, width)
        self.head = LatentHead(width, settings.latent_channels, settings.decoder_depth)

    def forward(self, text: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        text_value = self.text_projection(text.float())
        style_value = self.style_projection(style.float())
        condition = self.condition_projection(torch.cat((text_value, style_value), dim=-1))
        tokens = self.learned_tokens + self.position + text_value[:, None] + style_value[:, None]
        tokens = self.backbone(tokens, condition)
        return self.head(tokens, self.settings.token_grid)


class StylePosterior(nn.Module):
    """Training-only posterior; inference always samples N(0,I)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        width = max(32, settings.width // 2)
        self.image = nn.Sequential(
            nn.Conv2d(settings.latent_channels, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.text = nn.Sequential(nn.LayerNorm(settings.text_dim), nn.Linear(settings.text_dim, width))
        self.statistics = nn.Linear(2 * width, 2 * settings.style_dim)

    def forward(self, target_latent: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.statistics(torch.cat((self.image(target_latent.float()), self.text(text.float())), dim=-1)).chunk(2, dim=-1)


@dataclass
class TrainOutput:
    predicted_latent: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_logvar: torch.Tensor
    style: torch.Tensor


class TextConditionedVAE(nn.Module):
    """Conditional VAE around either the LightGen or electronic generator."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.generator = ConditionalLatentGenerator(settings)
        self.posterior = StylePosterior(settings)

    def forward_train(self, text: torch.Tensor, target_latent: torch.Tensor) -> TrainOutput:
        mean, logvar = self.posterior(target_latent, text)
        logvar = logvar.clamp(-12.0, 8.0)
        style = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return TrainOutput(self.generator(text, style), mean, logvar, style)

    def generate(self, text: torch.Tensor, *, seed: int | None = None) -> torch.Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=text.device).manual_seed(int(seed))
        style = torch.randn(
            text.shape[0], self.settings.style_dim,
            device=text.device, dtype=torch.float32, generator=generator,
        )
        return self.generator(text, style)

    def architecture_report(self) -> dict[str, Any]:
        lightgen = self.settings.variant == "lightgen_parallel"
        return {
            "task": "single-object text-to-image generation",
            "variant": self.settings.variant,
            "text_encoder": "frozen Qwen3-VL-2B-Instruct cached text features",
            "training_only_posterior": True,
            "inference_iterations": 1,
            "vae_decoder_calls": 1,
            "decoder_residual_depth": self.settings.decoder_depth,
            "branch_graph": (
                "stage input -> {electronic residual || optical block} -> detached-RMS fusion"
                if lightgen else "two conditioned electronic residual blocks"
            ),
            "optical_backend": self.settings.optical_backend,
            "optical_and_electronic_are_parallel": lightgen,
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


def build_model(settings: Settings, device: torch.device | str = "cpu") -> TextConditionedVAE:
    return TextConditionedVAE(settings).to(device)


class PatchDiscriminator(nn.Module):
    """Small spectral-normalized RGB PatchGAN used only by adversarial profiles."""

    def __init__(self, width: int = 48) -> None:
        super().__init__()
        channels = (3, width, width * 2, width * 4, width * 8)
        self.blocks = nn.ModuleList()
        for index, (input_channels, output_channels) in enumerate(zip(channels, channels[1:])):
            convolution = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1)
            )
            self.blocks.append(nn.Sequential(convolution, nn.LeakyReLU(0.2, inplace=False)))
        self.output = nn.utils.spectral_norm(nn.Conv2d(channels[-1], 1, 3, padding=1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        value = image
        for block in self.blocks:
            value = block(value)
            features.append(value)
        return self.output(value), features


__all__ = [
    "AuditedDC20Backbone", "CompactFourierOptics", "ConditionalLatentGenerator", "ParallelHybridBackbone",
    "PatchDiscriminator", "ScaleMatchedFusion", "TextConditionedVAE", "TrainOutput", "build_model",
]
