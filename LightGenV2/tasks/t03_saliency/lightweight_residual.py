"""Task-local GRN adaptation; not a ConvNeXt backbone or attention network.

Equation reference: Woo et al., ConvNeXt V2 (CVPR 2023), arXiv:2301.00808.
Independently expressed for [B,196,C] token grids with FP32 response statistics.
The 196 unpadded SALICON tokens correspond to the 14x14 spatial grid; L2
aggregation is permutation-invariant and must never cross the batch dimension.
"""
import torch
from torch import nn


def widen_ffn_checkpoint(source: dict, target: dict) -> tuple[dict, bool]:
    """Net2Wider-style 384->576 expansion, restricted to the two existing FFNs.

    Duplicate the first 192 hidden channels, splitting their outgoing weights
    0.4/0.6. Unequal splits allow different gradients without changing the
    deterministic initial function. Dropout equivalence is NOT claimed.
    Reference: Chen et al., Net2Net, ICLR 2016, arXiv:1511.05641.
    """
    if source.keys() != target.keys():
        raise RuntimeError("FFN widening cannot add/remove checkpoint keys")
    shapes = {}
    for i in range(2):
        prefix = f"hybrid.blocks.{i}.mlp."
        shapes.update({prefix+'0.weight': ((384,192),(576,192)),
                       prefix+'0.bias': ((384,),(576,)),
                       prefix+'1.0.conv.weight': ((384,1,3,3),(576,1,3,3)),
                       prefix+'3.weight': ((192,384),(192,576))})
    changed = {k for k in source if source[k].shape != target[k].shape}
    if not changed:
        if any(tuple(source[k].shape) != pair[1] for k,pair in shapes.items()):
            raise RuntimeError("Expected already widened 576-channel FFN")
        return source, False
    if changed != shapes.keys():
        raise RuntimeError("Only eight audited FFN tensors may change shape")
    result = dict(source)
    for key,(old_shape,new_shape) in shapes.items():
        value = source[key]
        if tuple(value.shape) != old_shape or tuple(target[key].shape) != new_shape:
            raise RuntimeError(f"Unexpected FFN widening shape: {key}")
        if key.endswith('3.weight'):
            result[key] = torch.cat((value[:,:192]*.4, value[:,192:], value[:,:192]*.6), dim=1)
        else:
            result[key] = torch.cat((value, value[:192]), dim=0)
    return result, True


def configure_wide_ffn(hybrid: nn.Module) -> None:
    """Enlarge only existing CFFN hidden space; preserve other initialization RNG."""
    if len(hybrid.blocks) != 2:
        raise ValueError("FFN widening requires two existing residuals")
    old = {f'hybrid.{k}':v.detach().clone() for k,v in hybrid.state_dict().items()}
    for block in hybrid.blocks:
        first, last = block.mlp[0], block.mlp[3]
        spatial = block.mlp[1][0]
        if (first.in_features, first.out_features, last.in_features, last.out_features) != (192,384,384,192):
            raise ValueError("Only 192->384->192 CFFN widening is audited")
        if not isinstance(spatial, PackedSpatialDepthwise) or spatial.conv.dilation != (1,1):
            raise ValueError("Widening requires existing dilation1 CFFN")
        device, dtype = first.weight.device, first.weight.dtype
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            block.mlp[0] = nn.Linear(192,576,device=device,dtype=dtype)
            block.mlp[3] = nn.Linear(576,192,device=device,dtype=dtype)
            block.mlp[1][0] = PackedSpatialDepthwise(576,1).to(device=device,dtype=dtype)
    target = {f'hybrid.{k}':v for k,v in hybrid.state_dict().items()}
    expanded, _ = widen_ffn_checkpoint(old, target)
    hybrid.load_state_dict({k.removeprefix('hybrid.'):v for k,v in expanded.items()}, strict=True)


class SpatialTokenLinear(nn.Linear):
    """Static low-rank spatial MLP inside the existing electronic pointwise op.

    Borrows token mixing from MLP-Mixer (arXiv:2105.01601), not its backbone.
    Shared across channels; never mixes samples or computes attention weights.
    The zero up-projection preserves the exact initial pointwise function.
    """
    def __init__(self, original: nn.Linear, rank: int = 16):
        if rank != 16 or type(original) is not nn.Linear:
            raise ValueError("Expected original Linear and audited spatial rank16")
        device, dtype = original.weight.device, original.weight.dtype
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            super().__init__(original.in_features, original.out_features,
                             bias=original.bias is not None, device=device, dtype=dtype)
        self.weight, self.bias = original.weight, original.bias
        # Deterministic 4x4 low-frequency 2D DCT basis on the true 14x14 grid.
        pos = torch.arange(14, device=device, dtype=torch.float32) + .5
        freq = torch.arange(4, device=device, dtype=torch.float32)[:, None]
        basis = torch.cos(torch.pi * freq * pos / 14)
        basis = basis / basis.square().sum(-1, keepdim=True).sqrt()
        down = torch.einsum('ay,bx->abyx', basis, basis).reshape(16, 196)
        self.spatial_down = nn.Parameter(down.to(dtype))
        self.spatial_up = nn.Parameter(torch.zeros(196, 16, device=device, dtype=dtype))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (196, self.in_features):
            raise ValueError("Spatial token mixing requires unpadded [B,196,C]")
        b, _, c = tokens.shape
        grid = tokens.reshape(b,7,7,2,2,c).permute(0,5,1,3,2,4).reshape(b,c,196)
        update = torch.nn.functional.linear(
            torch.nn.functional.gelu(torch.nn.functional.linear(grid, self.spatial_down)),
            self.spatial_up)
        update = update.reshape(b,c,7,2,7,2).permute(0,2,4,3,5,1).reshape(b,196,c)
        return torch.nn.functional.linear(tokens + update, self.weight, self.bias)


def configure_global_mixing(hybrid: nn.Module, rank: int) -> None:
    if len(hybrid.blocks) != 2:
        raise ValueError("T03 global mixing expects exactly two existing residuals")
    for block in hybrid.blocks:
        block.token_pointwise = SpatialTokenLinear(block.token_pointwise, rank)


def initialize_identity_global_mixing(source: dict, target: dict) -> tuple[dict, bool]:
    expected = {f"hybrid.blocks.{i}.token_pointwise.spatial_{name}"
                for i in range(2) for name in ('down','up')}
    missing = target.keys() - source.keys()
    if not missing:
        return source, False
    if missing != expected or source.keys() - target.keys():
        raise RuntimeError("Global transfer only allows four spatial projection tensors")
    for key in expected:
        shape = (196,16) if key.endswith('up') else (16,196)
        if target[key].shape != shape or not torch.isfinite(target[key]).all():
            raise RuntimeError("Invalid global projection initialization")
        if key.endswith('up') and torch.count_nonzero(target[key]).item():
            raise RuntimeError("Global up-projection must be zero for identity transfer")
    return {**source, **{k:target[k].detach().clone() for k in expected}}, True


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
