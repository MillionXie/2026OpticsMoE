"""Mask Generator — Transformer + MLP that maps CLIP features to D2NN phase masks.

Architecture (mirrors TF embedding_d2nn_transformer):
    CLIP feature [B, 1024]
      → Linear(1024 → 768)
      → Transformer Encoder ×4 (8 heads, d=768)
      → LayerNorm(768)
      → Flatten
      → Dropout(0.2)
      → Linear(768 → 3072) + GELU
      → Dropout(0.2)
      → Linear(3072 → num_layers × mask_size²)
      → Reshape → [B, num_layers, mask_size, mask_size]
      → Mean over batch → [num_layers, mask_size, mask_size]
      → sigmoid × 2π → phase ∈ [0, 2π]
      → (Optional) 4-bit quantisation via STE

Supports optional phase quantisation via num_bits parameter:
    - None / 0: continuous phase (no quantisation)
    - 4: 4-bit, 16 levels in [0, 2π]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_phase_ste(phase: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Quantize phase to `num_bits` with straight-through estimator.

    Forward: quantized phase (2^num_bits levels in [0, 2π]).
    Backward: gradient flows as if phase is continuous.

    Args:
        phase: tensor in [0, 2π].
        num_bits: number of bits (e.g. 4 → 16 levels).

    Returns:
        quantized phase in [0, 2π] with STE gradient.
    """
    levels = 2 ** num_bits
    # Normalise to [0, 1]
    phase_norm = phase / (2.0 * math.pi)
    # Quantize
    phase_q_norm = torch.round(phase_norm * (levels - 1)) / (levels - 1)
    phase_q = phase_q_norm * (2.0 * math.pi)
    # STE: forward uses quantised, backward flows through phase
    return phase + (phase_q - phase).detach()


class TransformerBlockPT(nn.Module):
    """Standard Pre-LN Transformer Encoder block (PyTorch style).

    Architecture:
        LN → MHA (+res) → LN → FFN(768→3072→768) (+res)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlockTF(nn.Module):
    """Post-LN Transformer Encoder block (exact TF mirror).

    Architecture (matches TF embedding_d2nn_transformer):
        LN → MHA (+res) → LN → Dense(768→1536)+GELU → LN → Dense(1536→768)+GELU → Dropout (+res)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.dense1 = nn.Linear(dim, dim * 2)       # 768 → 1536
        self.norm3 = nn.LayerNorm(dim * 2, eps=1e-6)
        self.dense2 = nn.Linear(dim * 2, dim)        # 1536 → 768
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x1 = self.norm1(x)
        attn_out = self.attn(x1, x1, x1)[0]
        x2 = attn_out + x                          # residual after MHA

        x3 = self.norm2(x2)
        x3 = F.gelu(self.dense1(x3))              # 768 → 1536 + GELU
        x3 = self.norm3(x3)
        x3 = F.gelu(self.dense2(x3))              # 1536 → 768 + GELU
        x3 = self.drop(x3)
        encoded = x3 + x2                          # residual after FFN
        return encoded


# Default alias for backward compatibility
TransformerBlock = TransformerBlockPT


class MaskGenerator(nn.Module):
    """Generate phase masks from CLIP image features.

    Input:  CLIP features [B, clip_dim]  (default 1024 for RN50)
    Output: phase masks [num_layers, mask_size, mask_size] in [0, 2π]

    Masks are averaged over the batch dimension (like TF embedding_d2nn_transformer).
    """

    def __init__(
        self,
        clip_dim: int = 1024,
        transformer_dim: int = 768,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        mlp_hidden_dim: int = 3072,
        dropout: float = 0.1,
        final_dropout: float = 0.2,
        num_layers: int = 5,
        mask_size: int = 256,
        num_bits: int = 0,
        transformer_style: str = "pytorch",
        use_skip: bool = False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.mask_size = int(mask_size)
        self.num_bits = int(num_bits) if num_bits else 0
        self.use_skip = use_skip
        mask_pixels = num_layers * mask_size * mask_size

        # Select Transformer block style
        if transformer_style == "tf":
            block_cls = TransformerBlockTF
        else:
            block_cls = TransformerBlockPT

        # Project CLIP features into transformer dimension
        self.input_proj = nn.Linear(clip_dim, transformer_dim)

        # Skip projection: CLIP → transformer_dim (bypass Transformer)
        if use_skip:
            self.skip_proj = nn.Linear(clip_dim, transformer_dim)

        # Transformer encoder stack
        self.transformer_blocks = nn.ModuleList([
            block_cls(transformer_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # Output head
        self.final_norm = nn.LayerNorm(transformer_dim, eps=1e-6)
        self.final_dropout = nn.Dropout(final_dropout)
        self.mlp_hidden = nn.Sequential(
            nn.Linear(transformer_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(final_dropout),
        )
        self.output_proj = nn.Linear(mlp_hidden_dim, mask_pixels)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """Forward pass — batch-averaged mask generation.

        Args:
            clip_features: [B, clip_dim]

        Returns:
            phase_masks: [num_layers, mask_size, mask_size] in [0, 2π]
        """
        B = clip_features.shape[0]

        # Project & add singleton sequence dim
        x = self.input_proj(clip_features)          # [B, 768]
        x = x.unsqueeze(1)                           # [B, 1, 768]

        # Transformer layers
        for blk in self.transformer_blocks:
            x = blk(x)

        # Final norm & flatten
        x = self.final_norm(x)                       # [B, 1, 768]
        x = x.squeeze(1)                             # [B, 768]

        # Skip connection: CLIP projected directly, bypassing Transformer
        if self.use_skip:
            skip = self.skip_proj(clip_features)      # [B, 768]
            x = x + skip

        # MLP head
        x = self.final_dropout(x)
        x = self.mlp_hidden(x)                       # [B, 3072]
        x = self.output_proj(x)                      # [B, num_layers * mask_size²]

        # Reshape to per-sample masks
        masks = x.view(B, self.num_layers, self.mask_size, self.mask_size)

        # Average over batch → one shared mask
        masks = masks.mean(dim=0)                    # [num_layers, mask_size, mask_size]

        # Unconstrained phase (same as traditional D2NN)
        phase = masks

        # Optional quantisation
        if self.num_bits > 0:
            phase = quantize_phase_ste(phase, self.num_bits)

        return phase  # [num_layers, mask_size, mask_size]


# Re-export
__all__ = ["MaskGenerator", "TransformerBlock"]
