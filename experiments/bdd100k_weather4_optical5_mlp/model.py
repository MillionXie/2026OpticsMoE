from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .optics import OpticalDetectionLayer


class Optical5MLPWeatherClassifier(nn.Module):
    """Grayscale amplitude -> five optical conversions -> electronic MLP readout."""

    def __init__(
        self,
        input_size: int = 224,
        optical_field_size: int = 256,
        optical_padding_size: int = 400,
        wavelength_nm: float = 532.0,
        pixel_pitch_um: float = 17.0,
        mask_distance_cm: float = 5.0,
        phase_init: str = "uniform",
        amplitude_mask_enabled: bool = True,
        detector_pool_size: int = 16,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 4,
        optical_layers: int = 5,
        phase_dropout: object | None = None,
    ) -> None:
        super().__init__()
        if optical_layers != 5:
            raise ValueError("Optical5MLPWeatherClassifier requires five optical layers")
        self.input_size = int(input_size)
        self.optical_field_size = int(optical_field_size)
        self.detector_pool_size = int(detector_pool_size)
        if phase_dropout is None:
            from .settings import PhaseDropoutSettings
            phase_dropout = PhaseDropoutSettings(enabled=False)
        self.optical_layers = nn.ModuleList([
            OpticalDetectionLayer(
                optical_field_size, optical_padding_size, wavelength_nm, pixel_pitch_um,
                mask_distance_cm, phase_init, amplitude_mask_enabled, phase_dropout,
            ) for _ in range(5)
        ])
        readout_dim = self.detector_pool_size ** 2
        self.detector_pool = nn.AdaptiveAvgPool2d((self.detector_pool_size, self.detector_pool_size))
        self.mlp = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def set_epoch(self, epoch: int) -> None:
        cfg = self.optical_layers[0].phase_dropout
        active = bool(cfg.enabled and epoch >= int(cfg.start_epoch))
        for layer in self.optical_layers:
            layer.set_phase_dropout_active(active)

    def encode_amplitude(self, grayscale: torch.Tensor) -> torch.Tensor:
        if grayscale.ndim != 4 or grayscale.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W], got {tuple(grayscale.shape)}")
        value = grayscale.float().clamp(0.0, 1.0)
        rms = value.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        amplitude = value / rms
        return F.interpolate(amplitude, size=(self.optical_field_size, self.optical_field_size), mode="bilinear", align_corners=False)[:, 0]

    def forward(self, grayscale: torch.Tensor, return_diagnostics: bool = False):
        amplitude = self.encode_amplitude(grayscale)
        input_amplitude = amplitude
        intensities: list[torch.Tensor] = []
        for layer in self.optical_layers:
            amplitude, intensity = layer(amplitude)
            if return_diagnostics:
                intensities.append(intensity)
        detector_input = intensity
        pooled = self.detector_pool(detector_input.unsqueeze(1)).flatten(1)
        logits = self.mlp(pooled)
        if not return_diagnostics:
            return logits
        return logits, {
            "input_amplitude": input_amplitude,
            "layer_intensities": intensities,
            "detector_input": detector_input,
            "detector_pooled": pooled,
        }

    def parameter_summary(self) -> dict[str, int]:
        phase = sum(layer.raw_phase.numel() for layer in self.optical_layers)
        amplitude = sum(layer.raw_amplitude.numel() for layer in self.optical_layers if layer.raw_amplitude is not None)
        mlp = sum(parameter.numel() for parameter in self.mlp.parameters())
        total = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "phase_mask_parameter_count": phase,
            "amplitude_mask_parameter_count": amplitude,
            "mlp_parameter_count": mlp,
            "total_trainable_parameter_count": total,
        }
