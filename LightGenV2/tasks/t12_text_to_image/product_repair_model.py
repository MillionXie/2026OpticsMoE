"""One-step image-and-text product editor with decoder-side parallel optics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .modeling import CompactFourierOptics, ScaleMatchedFusion


@dataclass(frozen=True)
class RepairModelConfig:
    optical_width: int = 224
    optical_grid: int = 8
    optical_experts: int = 4
    optical_top_k: int = 2
    alpha_initial: float = 0.50
    alpha_minimum: float = 0.40
    alpha_maximum: float = 0.75
    rms_epsilon: float = 1e-6

    def validate(self) -> None:
        if min(self.optical_width, self.optical_grid, self.optical_experts, self.optical_top_k) <= 0:
            raise ValueError("Optical dimensions must be positive")
        if self.optical_top_k > self.optical_experts:
            raise ValueError("optical_top_k cannot exceed optical_experts")
        if not 0.4 <= self.alpha_minimum < self.alpha_initial < self.alpha_maximum <= 1.0:
            raise ValueError("Decoder optical alpha must stay at or above 0.4")


class ParallelOpticalDecoderBlock(nn.Module):
    """Wrap the deepest UNet up block with true E/O parallel branches.

    The original electronic up block and the optical branch consume the same
    deepest decoder activation.  Only their outputs are fused; the optical
    branch never consumes the electronic result.
    """

    has_cross_attention = True

    def __init__(
        self,
        electronic: nn.Module,
        *,
        input_channels: int,
        output_channels: int,
        timestep_dim: int,
        condition_dim: int,
        config: RepairModelConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.electronic = electronic
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.grid = int(config.optical_grid)
        width = int(config.optical_width)
        self.input_norm = nn.LayerNorm(input_channels)
        self.input_projection = nn.Linear(input_channels, width)
        self.timestep_projection = nn.Linear(timestep_dim, width)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.optical = CompactFourierOptics(
            width, self.grid, experts=config.optical_experts, top_k=config.optical_top_k
        )
        self.expert_gate = nn.Parameter(torch.tensor(-1.0))
        self.global_gate = nn.Parameter(torch.tensor(-1.0))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, output_channels)
        self.base_projection = nn.Conv2d(input_channels, output_channels, 1)
        self.output_gate = nn.Parameter(torch.tensor(-1.5))
        self.fusion = ScaleMatchedFusion(
            config.alpha_initial,
            config.alpha_minimum,
            config.alpha_maximum,
            config.rms_epsilon,
        )
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)
        if input_channels == output_channels:
            nn.init.dirac_(self.base_projection.weight)
            nn.init.zeros_(self.base_projection.bias)

    @property
    def resnets(self):
        # UNet2DConditionModel reads this before dispatching the residual tuple.
        return self.electronic.resnets

    @property
    def upsamplers(self):
        return self.electronic.upsamplers

    def optical_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("electronic."):
                yield name, parameter

    def _optical_branch(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, channels, height, width = hidden_states.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} decoder channels, got {channels}")
        pooled = F.adaptive_avg_pool2d(hidden_states.float(), (self.grid, self.grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.input_projection(self.input_norm(tokens))
        tokens = tokens + self.timestep_projection(temb.float())[:, None]
        text = encoder_hidden_states.float().mean(dim=1)
        tokens = F.silu(tokens + self.condition_projection(text)[:, None])
        expert = self.optical.expert(tokens)
        tokens = tokens + torch.sigmoid(self.expert_gate) * expert
        global_value = self.optical.global_block(tokens)
        tokens = tokens + torch.sigmoid(self.global_gate) * global_value
        delta = self.output_projection(self.output_norm(tokens))
        delta = delta.transpose(1, 2).reshape(batch, self.output_channels, self.grid, self.grid)
        base = self.base_projection(hidden_states.float())
        base = F.interpolate(base, output_size, mode="bilinear", align_corners=False)
        delta = F.interpolate(delta, output_size, mode="bilinear", align_corners=False)
        return base + torch.sigmoid(self.output_gate) * delta

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        upsample_size: int | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temb is None or encoder_hidden_states is None:
            raise ValueError("Decoder optical block requires timestep and text conditions")
        shared_input = hidden_states
        electronic = self.electronic(
            hidden_states=shared_input,
            res_hidden_states_tuple=res_hidden_states_tuple,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            upsample_size=upsample_size,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        optical = self._optical_branch(
            shared_input, temb, encoder_hidden_states, electronic.shape[-2:]
        )
        return self.fusion(electronic, optical.to(electronic.dtype))

    def architecture_report(self) -> dict[str, Any]:
        return {
            "location": "first/deepest UNet decoder up block",
            "parallel_contract": "electronic(shared_input) || optical(shared_input), then RMS fusion",
            "electronic_and_optical_are_parallel": True,
            "optical_backend": "differentiable phase-only FFT simulation",
            "optical_grid": [self.grid, self.grid],
            "optical_parameters": sum(p.numel() for _, p in self.optical_parameters()),
            "alpha": float(self.fusion.alpha.detach()),
            "alpha_minimum": self.fusion.minimum,
            "alpha_maximum": self.fusion.maximum,
        }


def expand_reference_conditioning(unet: nn.Module) -> nn.Conv2d:
    """Expand the pretrained latent input from noise-only 4ch to noise+reference 8ch."""

    old = unet.conv_in
    if old.in_channels == 8:
        return old
    if old.in_channels != 4:
        raise ValueError(f"Expected a four-channel latent UNet, got {old.in_channels}")
    new = nn.Conv2d(
        8,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight[:, :4].copy_(old.weight)
        new.weight[:, 4:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.config.in_channels = 8
    return new


def attach_decoder_optics(
    unet: nn.Module, config: RepairModelConfig
) -> ParallelOpticalDecoderBlock:
    if isinstance(unet.up_blocks[0], ParallelOpticalDecoderBlock):
        return unet.up_blocks[0]
    electronic = unet.up_blocks[0]
    input_channels = int(unet.config.block_out_channels[-1])
    output_channels = int(electronic.resnets[-1].out_channels)
    timestep_dim = int(unet.time_embedding.linear_2.out_features)
    condition_dim = int(unet.config.cross_attention_dim)
    wrapper = ParallelOpticalDecoderBlock(
        electronic,
        input_channels=input_channels,
        output_channels=output_channels,
        timestep_dim=timestep_dim,
        condition_dim=condition_dim,
        config=config,
    )
    unet.up_blocks[0] = wrapper
    return wrapper


def one_step_edit(
    unet: nn.Module,
    noise: torch.Tensor,
    reference_latent: torch.Tensor,
    condition: torch.Tensor,
    sigma: torch.Tensor,
    *,
    timestep: int = 999,
) -> torch.Tensor:
    latent = noise * sigma
    scaled_noise = latent / torch.sqrt(sigma.square() + 1)
    model_input = torch.cat((scaled_noise, reference_latent), dim=1)
    timesteps = torch.full((len(noise),), timestep, device=noise.device, dtype=torch.long)
    prediction = unet(
        model_input, timesteps, encoder_hidden_states=condition, return_dict=False
    )[0]
    return latent - sigma * prediction


def architecture_report(
    unet: nn.Module,
    vae: nn.Module,
    adapter: nn.Module,
    optical: ParallelOpticalDecoderBlock,
    config: RepairModelConfig,
) -> dict[str, Any]:
    unet_parameters = sum(p.numel() for p in unet.parameters())
    decoder_parameters = sum(p.numel() for p in vae.decoder.parameters())
    adapter_parameters = sum(p.numel() for p in adapter.parameters())
    return {
        "variant": "qwen_bksdm_v2_tiny_reference_decoder_optical",
        "flow": "image->frozen VAE encoder; text->frozen Qwen->adapter; one UNet call->VAE decoder",
        "unet_parameters": unet_parameters,
        "vae_decoder_parameters": decoder_parameters,
        "qwen_condition_adapter_parameters": adapter_parameters,
        "generation_tail_parameters": unet_parameters + decoder_parameters + adapter_parameters,
        "qwen_parameters_excluded_from_tail": True,
        "inference_iterations": 1,
        "unet_calls": 1,
        "vae_decoder_calls": 1,
        "reference_conditioning": "8ch concatenation of noisy latent and input-image latent",
        "optical_decoder": optical.architecture_report(),
        "config": asdict(config),
    }


__all__ = [
    "ParallelOpticalDecoderBlock",
    "RepairModelConfig",
    "architecture_report",
    "attach_decoder_optics",
    "expand_reference_conditioning",
    "one_step_edit",
]
