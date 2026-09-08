"""No Transformer/attention frontend; symmetric router/expert/global paths."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .modeling import LightGenOpenMojiEditor, _compact, DenseTwoStageOpticalPath, OpticalDetectorTopKRouter
from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import (
    BalancedLanguageCore, BalancedVisionCore,
)


class ScaleOnlyCCD(nn.Module):
    """Only a per-frame scalar gain: no pixel clipping, log, gamma or threshold."""
    def forward(self, intensity):
        value = intensity.float()
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError('CCD intensity must be finite and nonnegative')
        return value / value.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class PositionReadout(nn.Module):
    """Input-independent trainable positional coefficients, NOT attention.

    One linear map over token position, shared by channels. Padding is zero.
    No max, no mean/max concatenation and no content-conditioned weights.
    Like any fixed-size readout this compresses the sequence; it is not lossless.
    """
    def __init__(self, max_tokens):
        super().__init__()
        self.max_tokens = max_tokens
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, max_tokens) / max_tokens)

    def forward(self, groups):
        if not groups or any(g.ndim != 2 or not 0 < len(g) <= self.max_tokens for g in groups):
            raise ValueError('Expected nonempty sequences within max_tokens')
        padded = torch.stack([F.pad(g, (0, 0, 0, self.max_tokens - len(g))) for g in groups])
        return torch.einsum('blc,l->bc', padded, self.weight.to(padded.dtype))


class PoolOnlyReadout(nn.Module):
    """Linear spatial binning of CCD intensity, without learned pixel transforms."""
    def __init__(self, previous):
        super().__init__()
        self.geometry = previous.geometry
        self.output_size = previous.output_size
        self.pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward_intensity(self, intensity):
        aperture = self.geometry.detector_aperture
        if intensity.ndim != 3 or intensity.shape[-2:] != (aperture.height, aperture.width):
            raise ValueError('CCD shape does not match the detector aperture')
        return self.pool(intensity.float().unsqueeze(1)).squeeze(1), intensity

    def forward(self, field):
        aperture = self.geometry.detector_aperture
        intensity = field.to(torch.complex64).abs().square().float()
        return self.forward_intensity(intensity[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1])


def position_encoding(length, width, device):
    p = torch.arange(length, device=device).float()[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width))
    code = torch.zeros(length, width, device=device)
    code[:, 0::2] = torch.sin(p * frequency)
    code[:, 1::2] = torch.cos(p * frequency[:code[:, 1::2].shape[1]])
    return code


def latent_forward(core, groups, causal, spatial_shapes=None):
    padded, mask, lengths = core._pad_groups(groups)
    latent = core.input_norm(core.input_adapter(padded.float()))
    electronic1 = core.blocks[0](latent, padding_mask=mask, causal=causal, **(
        {'spatial_shapes': spatial_shapes} if spatial_shapes is not None else {}))
    disabled = core.fusion_mode == 'electronic_only' or core.fusion_ablation_mode == 'remove_optical'
    if disabled:
        optical1, routing = None, None
    else:
        optical1, routing, optical_lengths = core.optical_branch.run_expert_block(latent, mask)
    fused1 = core._fuse(electronic1, optical1, core.block1_optical_fusion, mask, 'block1')
    electronic2 = core.blocks[1](fused1, padding_mask=mask, causal=causal, **(
        {'spatial_shapes': spatial_shapes} if spatial_shapes is not None else {}))
    optical2 = None
    if not disabled:
        amplitude = core.optical_branch.encode_global_input(fused1, mask, routing)
        optical2 = core.optical_branch.run_global_block(amplitude, optical_lengths, mask, fused1.dtype)
    final = core.output_norm(core._fuse(electronic2, optical2, core.block2_optical_fusion, mask, 'block2'))
    final = final.masked_fill(mask.unsqueeze(-1), 0)
    core.last_latent_groups = [final[i, :n] for i, n in enumerate(lengths)]
    core.last_routing = routing
    return torch.cat(core.last_latent_groups), final


class TokenLanguageCore(BalancedLanguageCore):
    def forward_groups(self, groups, *, causal=True, **kwargs):
        if not causal:
            raise ValueError('Language uses causal electronic convolution')
        return latent_forward(self, groups, True)


class PatchVisionCore(BalancedVisionCore):
    def forward_groups(self, groups, *, causal, spatial_shapes=None):
        if causal or spatial_shapes is None:
            raise ValueError('Vision requires spatial layout')
        return latent_forward(self, groups, False, spatial_shapes)


class EmbeddingOnlyEditor(LightGenOpenMojiEditor):
    def __init__(self, settings):
        super().__init__(settings)
        compact = _compact(settings)
        self.language_core = TokenLanguageCore(2048, settings.max_language_tokens, compact)
        self.vision_core = PatchVisionCore(1024, 196, compact)
        for core, max_tokens in [(self.language_core, settings.max_language_tokens), (self.vision_core, 196)]:
            del core.output_adapter
            del core.residual_logit
            if self.router_backend == 'none':
                core.optical_branch = DenseTwoStageOpticalPath(core.width, compact, max_tokens=max_tokens)
            else:
                core.optical_branch.core.router = OpticalDetectorTopKRouter(core.optical_branch.core.geometry, compact)
            core.optical_branch.ccd_normalizer = ScaleOnlyCCD()
            branch = core.optical_branch
            branch.core.readout = PoolOnlyReadout(branch.core.readout)
            branch.expert_readout = PoolOnlyReadout(branch.expert_readout)
        self.position_readout = PositionReadout(settings.max_language_tokens)
        width = settings.electronic_width
        self.language_pool = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.editor = nn.ModuleList(list(self.editor)[:settings.editor_depth])
        self.checkpoint_architecture = (
            f't04_embedding_only_{self.router_backend}_alpha{settings.fusion_alpha_minimum:.4f}_'
            f'{settings.fusion_alpha_maximum:.4f}_e{settings.editor_depth}_scaleccd_positionlinear_v1')
        self.assert_contract()

    def _language_condition(self, groups):
        positioned = []
        for group in groups:
            if group.ndim != 2 or group.shape[1] != 2048 or not 0 < len(group) <= self.settings.max_language_tokens:
                raise ValueError('Expected nonempty raw embedding token sequence [L<=64,2048]')
            rms = group.float().square().mean().sqrt().detach().clamp_min(1e-6)
            positioned.append(group + self.settings.position_scale * rms * position_encoding(len(group), 2048, group.device))
        self.language_core.forward_groups(positioned, causal=True)
        return self.language_pool(self.position_readout(self.language_core.last_latent_groups))

    def assert_contract(self):
        for module in self.modules():
            name = type(module).__name__.lower()
            if 'attention' in name or 'transformer' in name:
                raise RuntimeError(f'Forbidden inference module: {name}')
        for core in (self.language_core, self.vision_core):
            if not core.fusion_alpha_min > 0.4:
                raise RuntimeError('Every feature-stage alpha must remain strictly above 0.4')
            if not isinstance(core.optical_branch.ccd_normalizer, ScaleOnlyCCD):
                raise RuntimeError('CCD normalization must be scalar-only')

    def architecture_report(self):
        result = super().architecture_report()
        result.update(text='frozen Qwen embed_tokens lookup ONLY + sinusoidal positions; zero language TF',
                      language_summary='input-independent learned positional linear readout after optical global',
                      contextual_cache_allowed=False, native_transformer_blocks=0, attention_modules=0,
                      ccd_preprocessing='I / max(mean(I), epsilon); no pixelwise clip/log/gamma',
                      legacy_output_adapters=False, editor_depth=len(self.editor),
                      initialization='random task network; no old teacher/decoder warmstart',
                      data_contract='deduplicated grid+instruction v2; source-grid preservation remains explicit')
        return result
