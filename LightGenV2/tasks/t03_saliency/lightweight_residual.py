"""Task-local GRN adaptation; not a ConvNeXt backbone or attention network.

Equation reference: Woo et al., ConvNeXt V2 (CVPR 2023), arXiv:2301.00808.
Independently expressed for [B,196,C] token grids with FP32 response statistics.
The 196 unpadded SALICON tokens correspond to the 14x14 spatial grid; L2
aggregation is permutation-invariant and must never cross the batch dimension.
"""
import torch
from torch import nn


class PackedSpatialDepthwise(nn.Module):
    """Spatial mixing inside the expanded MLP; preserve Qwen 2x2-block packing.

    Inspired by expanded-space depthwise convolution (MobileNetV2/Mix-FFN),
    not their backbones. Fixed SALICON 14x14 grid, no attention or batch mixing.
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        if dilation not in (1, 2):
            raise ValueError("Only dilation 1/2 is audited")
        # Constructor's discarded random weights must not shift paired-run RNG.
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv2d(channels, channels, 3, padding=dilation,
                                  dilation=dilation, groups=channels, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 1, 1] = 1

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (196, self.conv.in_channels):
            raise ValueError("Spatial FFN requires unpadded [B,196,C] Qwen tokens")
        b, _, c = tokens.shape
        grid = tokens.reshape(b, 7, 7, 2, 2, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, 14, 14)
        grid = self.conv(grid)
        return grid.reshape(b, c, 7, 2, 7, 2).permute(0, 2, 4, 3, 5, 1).reshape(b, 196, c)


def configure_spatial_ffn(hybrid: nn.Module, dilation: int) -> None:
    if len(hybrid.blocks) != 2:
        raise ValueError("T03 spatial FFN expects two electronic residuals")
    for block in hybrid.blocks:
        if not isinstance(block.mlp[1], nn.GELU):
            raise ValueError("Expected original GELU at mlp.1")
        linear = block.mlp[0]
        spatial = PackedSpatialDepthwise(linear.out_features, dilation).to(
            device=linear.weight.device, dtype=linear.weight.dtype)
        # Learned legacy names mlp.0 and mlp.3 remain unchanged.
        block.mlp[1] = nn.Sequential(spatial, block.mlp[1])


def initialize_identity_spatial_ffn(source: dict, target: dict) -> tuple[dict, bool]:
    missing = target.keys() - source.keys()
    if not missing:
        return source, False
    expected = {f"hybrid.blocks.{i}.mlp.1.0.conv.weight" for i in range(2)}
    if missing != expected or source.keys() - target.keys():
        raise RuntimeError("Spatial FFN transfer only allows two new depthwise weights")
    for key in expected:
        w = target[key]
        identity = torch.zeros_like(w)
        identity[:, 0, 1, 1] = 1
        if w.shape != (384, 1, 3, 3) or not torch.equal(w, identity):
            raise RuntimeError("New expanded-space convolution must be a 384-channel identity")
    return {**source, **{k: target[k].detach().clone() for k in expected}}, True


class TokenGlobalResponseNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != 196 or tokens.shape[2] != self.gamma.numel():
            raise ValueError("T03 GRN requires an unpadded [B,196,C] SALICON grid")
        values = tokens.float()
        channel_response = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        relative_response = channel_response / (channel_response.mean(dim=-1, keepdim=True) + 1e-6)
        correction = values * relative_response * self.gamma + self.beta
        return (values + correction).to(tokens.dtype)


def configure_grn(hybrid: nn.Module) -> None:
    for block in hybrid.blocks:
        if not isinstance(block.mlp[2], nn.Dropout):
            raise ValueError("Expected legacy post-GELU dropout before GRN insertion")
        width = block.mlp[0].out_features
        grn = TokenGlobalResponseNorm(width).to(block.mlp[0].weight.device)
        # Preserve all old learned parameter names, particularly mlp.3.weight.
        block.mlp[2] = nn.Sequential(grn, block.mlp[2])


def initialize_identity_grn(source: dict, target: dict) -> tuple[dict, bool]:
    expected = {f"hybrid.blocks.{i}.mlp.2.0.{name}" for i in range(2) for name in ("gamma", "beta")}
    missing = target.keys() - source.keys()
    if not missing:
        return source, False
    if missing != expected or source.keys() - target.keys():
        raise RuntimeError("GRN transfer only permits four new gamma/beta tensors")
    if any(torch.count_nonzero(target[k]).item() for k in expected):
        raise RuntimeError("New GRN must initialize as the identity")
    return {**source, **{k: target[k].detach().clone() for k in expected}}, True
