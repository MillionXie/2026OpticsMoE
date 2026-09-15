from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _bounded_fusion(raw_logit: torch.Tensor, minimum: float) -> torch.Tensor:
    return float(minimum) + (1.0 - float(minimum)) * torch.sigmoid(raw_logit)


class OfflineElectronicResidualMLPBlock(nn.Module):
    """Exact causal Language mixer, copied here to keep the lab tail self-contained."""

    def __init__(
        self,
        width: int,
        expansion: float,
        dropout: float,
        initial_residual_weight: float,
        token_mixer_enabled: bool,
        token_mixer_kernel_size: int,
    ) -> None:
        super().__init__()
        hidden_width = int(round(width * expansion))
        self.token_mixer_enabled = bool(token_mixer_enabled)
        self.token_mixer_kernel_size = int(token_mixer_kernel_size)
        self.token_mixer_type = "depthwise_conv1d"
        if self.token_mixer_enabled:
            self.token_norm = nn.LayerNorm(width)
            self.token_depthwise = nn.Conv1d(
                width,
                width,
                kernel_size=self.token_mixer_kernel_size,
                groups=width,
                bias=False,
            )
            self.token_pointwise = nn.Linear(width, width)
            self.token_dropout = nn.Dropout(dropout)
            self.token_residual_logit = nn.Parameter(
                torch.logit(torch.tensor(float(initial_residual_weight)))
            )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
            nn.Dropout(dropout),
        )
        self.residual_logit = nn.Parameter(
            torch.logit(torch.tensor(float(initial_residual_weight)))
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        if not causal:
            raise RuntimeError("Offline Language Block 2 must use causal mixing")
        if self.token_mixer_enabled:
            token_input = self.token_norm(hidden).masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
            token_input = F.pad(
                token_input.transpose(1, 2),
                (self.token_mixer_kernel_size - 1, 0),
            )
            token_update = self.token_depthwise(token_input).transpose(1, 2)
            token_update = self.token_pointwise(F.gelu(token_update))
            token_update = self.token_dropout(token_update)
            hidden = hidden + torch.sigmoid(self.token_residual_logit) * token_update
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        update = self.mlp(self.norm(hidden))
        hidden = hidden + torch.sigmoid(self.residual_logit) * update
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class OfflineFullPlaneReadout(nn.Module):
    """Parameter-compatible 478-to-token readout without an optical simulator."""

    def __init__(
        self,
        *,
        detector_size: int,
        output_size: int,
        layernorm_eps: float,
        layernorm_affine: bool,
        layernorm_scope: str,
        nonlinearity: str,
    ) -> None:
        super().__init__()
        self.detector_size = int(detector_size)
        self.output_size = int(output_size)
        if layernorm_scope not in {"per_token", "full_plane"}:
            raise ValueError(f"Unsupported detector LayerNorm scope {layernorm_scope!r}")
        if nonlinearity not in {"relu", "softplus"}:
            raise ValueError(f"Unsupported detector nonlinearity {nonlinearity!r}")
        self.layernorm_scope = str(layernorm_scope)
        self.nonlinearity = str(nonlinearity)
        self.pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        normalized_shape: int | tuple[int, int] = (
            self.output_size
            if self.layernorm_scope == "per_token"
            else (self.output_size, self.output_size)
        )
        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=float(layernorm_eps),
            elementwise_affine=bool(layernorm_affine),
        )

    def forward(self, detector_intensity: torch.Tensor) -> torch.Tensor:
        expected = (self.detector_size, self.detector_size)
        if detector_intensity.ndim != 3 or tuple(detector_intensity.shape[-2:]) != expected:
            raise ValueError(
                f"Offline measured CCD must be [B,{expected[0]},{expected[1]}], "
                f"got {tuple(detector_intensity.shape)}"
            )
        if not torch.isfinite(detector_intensity).all():
            raise RuntimeError("Offline measured CCD contains NaN or Inf")
        if torch.any(detector_intensity < -1.0e-7):
            raise RuntimeError("Offline measured CCD must be nonnegative")
        pooled = self.pool(
            detector_intensity.float().clamp_min(0.0).unsqueeze(1)
        ).squeeze(1)
        normalized = self.norm(pooled)
        return (
            F.relu(normalized)
            if self.nonlinearity == "relu"
            else F.softplus(normalized)
        )


class LanguageGlobalOfflineTail(nn.Module):
    """Full-parity Language-global tail with no Qwen or optical simulator."""

    def __init__(
        self,
        *,
        width: int,
        max_tokens: int,
        expansion: float,
        dropout: float,
        initial_residual_weight: float,
        token_mixer_enabled: bool,
        token_mixer_type: str,
        token_mixer_kernel_size: int,
        detector_size: int,
        detector_output_size: int,
        detector_layernorm_eps: float,
        detector_layernorm_affine: bool,
        detector_layernorm_scope: str,
        detector_nonlinearity: str,
        ccd_relative_clip: float,
        ccd_log_compression: float,
        minimum_optical_fusion: float,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.max_tokens = int(max_tokens)
        self.detector_size = int(detector_size)
        self.detector_output_size = int(detector_output_size)
        self.ccd_relative_clip = float(ccd_relative_clip)
        self.ccd_log_compression = float(ccd_log_compression)
        self.minimum_optical_fusion = float(minimum_optical_fusion)
        self.embedding_dim = int(embedding_dim)
        if token_mixer_type != "depthwise_conv1d":
            raise ValueError("Offline Language tail requires depthwise_conv1d")
        if not 0.0 <= self.minimum_optical_fusion < 1.0:
            raise ValueError("minimum_optical_fusion must lie in [0,1)")
        if self.width <= 0 or self.max_tokens <= 0 or self.detector_size <= 0:
            raise ValueError("Offline tail dimensions must be positive")
        if self.detector_output_size < self.max_tokens:
            raise ValueError("Detector token rows must cover max_tokens")
        if self.ccd_relative_clip <= 0.0 or self.ccd_log_compression <= 0.0:
            raise ValueError("Offline CCD normalization constants must be positive")

        self.block2 = OfflineElectronicResidualMLPBlock(
            self.width,
            float(expansion),
            float(dropout),
            float(initial_residual_weight),
            bool(token_mixer_enabled),
            int(token_mixer_kernel_size),
        )
        self.ccd_readout = OfflineFullPlaneReadout(
            detector_size=self.detector_size,
            output_size=self.detector_output_size,
            layernorm_eps=float(detector_layernorm_eps),
            layernorm_affine=bool(detector_layernorm_affine),
            layernorm_scope=str(detector_layernorm_scope),
            nonlinearity=str(detector_nonlinearity),
        )
        self.optical_output_adapter = nn.Linear(self.detector_output_size, self.width)
        self.output_norm = nn.LayerNorm(self.width)
        self.block2_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        detector_feature_size = 2 * self.width
        self.retrieval_norm = nn.LayerNorm(detector_feature_size)
        self.retrieval_projection = nn.Linear(detector_feature_size, self.embedding_dim)

    @property
    def block2_optical_fusion(self) -> torch.Tensor:
        return _bounded_fusion(
            self.block2_optical_fusion_logit, self.minimum_optical_fusion
        )

    def _pad_groups(
        self, groups: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not groups or any(group.ndim != 2 for group in groups):
            raise ValueError("Offline tail expects a non-empty list of [L,width] tensors")
        lengths = [len(group) for group in groups]
        if any(
            length <= 0
            or length > self.max_tokens
            or group.shape[-1] != self.width
            for length, group in zip(lengths, groups)
        ):
            raise ValueError("Offline tail received an invalid token length or width")
        max_length = max(lengths)
        padded = torch.zeros(
            len(groups),
            max_length,
            self.width,
            device=groups[0].device,
            dtype=torch.float32,
        )
        padding_mask = torch.ones(
            len(groups), max_length, dtype=torch.bool, device=groups[0].device
        )
        for index, group in enumerate(groups):
            if group.device != groups[0].device:
                raise ValueError("Every cached token group must use the same device")
            padded[index, : len(group)] = group.float()
            padding_mask[index, : len(group)] = False
        return padded, padding_mask, lengths

    def _normalize_ccd(self, intensity: torch.Tensor) -> torch.Tensor:
        expected = (self.detector_size, self.detector_size)
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != expected:
            raise ValueError(
                f"Offline CCD must be [B,{expected[0]},{expected[1]}], "
                f"got {tuple(intensity.shape)}"
            )
        value = intensity.float().clamp_min(0.0)
        if not torch.isfinite(value).all():
            raise RuntimeError("Offline CCD intensity contains NaN or Inf")
        frame_mean = value.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        relative = (value / frame_mean).clamp_max(self.ccd_relative_clip)
        return torch.log1p(self.ccd_log_compression * relative)

    def detector_features(
        self, block2_input_groups: list[torch.Tensor], ccd: torch.Tensor
    ) -> torch.Tensor:
        padded, padding_mask, lengths = self._pad_groups(block2_input_groups)
        if len(block2_input_groups) != len(ccd):
            raise ValueError(
                f"Cached group count {len(block2_input_groups)} does not match "
                f"CCD batch {len(ccd)}"
            )
        electronic = self.block2(padded, padding_mask=padding_mask, causal=True)
        readout = self.ccd_readout(self._normalize_ccd(ccd))
        optical_delta = self.optical_output_adapter(readout[:, : padded.shape[1]])
        latent = self.output_norm(
            electronic + self.block2_optical_fusion * optical_delta
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return torch.stack(
            [
                torch.cat(
                    (
                        latent[index, :length].mean(dim=0),
                        latent[index, :length].amax(dim=0),
                    ),
                    dim=0,
                )
                for index, length in enumerate(lengths)
            ],
            dim=0,
        )

    def forward(
        self, block2_input_groups: list[torch.Tensor], ccd: torch.Tensor
    ) -> torch.Tensor:
        features = self.detector_features(block2_input_groups, ccd)
        raw = self.retrieval_projection(self.retrieval_norm(features.float()))
        if not torch.isfinite(raw).all() or torch.any(raw.norm(dim=-1) <= 1.0e-12):
            raise RuntimeError("Offline retrieval head produced an invalid embedding")
        return F.normalize(raw, p=2, dim=-1)


__all__ = ["LanguageGlobalOfflineTail", "OfflineElectronicResidualMLPBlock"]
