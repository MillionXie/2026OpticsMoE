"""Task-local GRN adaptation; not a ConvNeXt backbone or attention network.

Equation reference: Woo et al., ConvNeXt V2 (CVPR 2023), arXiv:2301.00808.
Independently expressed for [B,196,C] token grids with FP32 response statistics.
The 196 unpadded SALICON tokens correspond to the 14x14 spatial grid; L2
aggregation is permutation-invariant and must never cross the batch dimension.
"""
import torch
from torch import nn


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
