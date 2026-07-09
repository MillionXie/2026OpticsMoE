from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .optics import AngularSpectrumPropagator


class ClassRegionDetector(nn.Module):
    """Fixed non-overlapping class regions; the region energies are the logits source."""

    def __init__(
        self,
        field_size: int,
        class_names: Sequence[str],
        region_size: int,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.field_size = int(field_size)
        self.class_names = list(class_names)
        self.region_size = int(region_size)
        self.temperature = float(temperature)
        self.eps = float(eps)
        masks = torch.zeros(len(self.class_names), self.field_size, self.field_size)
        boxes = []
        center_y = self.field_size / 2
        for index, name in enumerate(self.class_names):
            center_x = self.field_size * (index + 1) / (len(self.class_names) + 1)
            x0 = int(round(center_x - self.region_size / 2))
            y0 = int(round(center_y - self.region_size / 2))
            x0 = max(0, min(x0, self.field_size - self.region_size))
            y0 = max(0, min(y0, self.field_size - self.region_size))
            x1 = x0 + self.region_size
            y1 = y0 + self.region_size
            masks[index, y0:y1, x0:x1] = 1
            boxes.append(
                {
                    "class_index": index,
                    "class_name": name,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": self.region_size,
                    "height": self.region_size,
                }
            )
        if torch.any(masks.sum(0) > 1):
            raise ValueError("Detector class regions overlap; reduce detector_region_size")
        self.register_buffer("region_masks", masks, persistent=True)
        self.boxes = boxes

    def forward(self, intensity: torch.Tensor) -> dict[str, torch.Tensor]:
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != (self.field_size, self.field_size):
            raise ValueError(f"Expected [B,{self.field_size},{self.field_size}]")
        value = intensity.float().clamp_min(0)
        region_energy = torch.einsum("bhw,khw->bk", value, self.region_masks.float())
        total = value.sum((-2, -1)).clamp_min(self.eps)
        fractions = region_energy / total[:, None]
        detector_fraction = fractions.sum(1).clamp(max=1)
        distribution = region_energy / region_energy.sum(1, keepdim=True).clamp_min(self.eps)
        logits = torch.log(region_energy.clamp_min(self.eps)) / self.temperature
        return {
            "region_energy": region_energy,
            "region_fractions": fractions,
            "region_distribution": distribution,
            "region_logits": logits,
            "detector_fraction": detector_fraction,
            "outside_fraction": (1 - detector_fraction).clamp_min(0),
        }

    def specification(self) -> dict[str, Any]:
        return {
            "layout": "horizontal_center",
            "field_size": self.field_size,
            "region_size": self.region_size,
            "temperature": self.temperature,
            "class_order": self.class_names,
            "boxes": self.boxes,
        }


class PhaseOnlyPropagationLayer(nn.Module):
    def __init__(
        self,
        field_size: int,
        padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_cm: float,
        phase_init: str = "uniform",
        phase_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.field_size = int(field_size)
        self.phase_mask = nn.Parameter(torch.empty(field_size, field_size))
        if phase_init == "uniform":
            nn.init.uniform_(self.phase_mask, 0, 2 * math.pi)
        elif phase_init == "zeros":
            nn.init.zeros_(self.phase_mask)
        elif phase_init in {"normal", "small_normal"}:
            nn.init.normal_(self.phase_mask, mean=0.0, std=phase_init_std)
        else:
            raise ValueError("phase_init must be uniform, zeros, normal, or small_normal")
        self.propagator = AngularSpectrumPropagator(
            field_size, padding_size, wavelength_nm, pixel_pitch_um, distance_cm
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.field_size, self.field_size):
            raise ValueError(f"Expected complex field [B,{self.field_size},{self.field_size}]")
        if not torch.is_complex(field):
            raise ValueError("PhaseOnlyPropagationLayer expects a complex field")
        modulation = torch.exp(1j * self.phase_mask.float()).to(torch.complex64).unsqueeze(0)
        return self.propagator(field.to(torch.complex64) * modulation)

    def wrapped_phase(self) -> torch.Tensor:
        return torch.remainder(self.phase_mask, 2 * math.pi)


class StandardD2NNTimeOfDayClassifier(nn.Module):
    """Phase-only D2NN; final detector-region energies are the class scores."""

    def __init__(
        self,
        field_size: int,
        padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_cm: float,
        phase_init: str,
        phase_init_std: float,
        num_classes: int,
        class_names: Sequence[str],
        optical_layers: int,
        detector_region_size: int,
        detector_region_temperature: float,
        input_energy_normalization: str = "rms",
    ) -> None:
        super().__init__()
        self.field_size = int(field_size)
        self.input_energy_normalization = input_energy_normalization
        self.layers = nn.ModuleList(
            [
                PhaseOnlyPropagationLayer(
                    field_size,
                    padding_size,
                    wavelength_nm,
                    pixel_pitch_um,
                    distance_cm,
                    phase_init,
                    phase_init_std,
                )
                for _ in range(optical_layers)
            ]
        )
        self.class_detector = ClassRegionDetector(
            field_size, list(class_names), detector_region_size, detector_region_temperature
        )
        if len(class_names) != num_classes:
            raise ValueError("class_names and num_classes must agree")

    def forward(
        self,
        image: torch.Tensor,
        return_diagnostics: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
        amplitude = self.encode(image)
        initial_intensity = amplitude.square()
        field = torch.complex(amplitude, torch.zeros_like(amplitude))
        intermediates: list[torch.Tensor] = []
        for layer in self.layers:
            field = layer(field)
            if return_diagnostics:
                intermediates.append(field.abs().square().float())
        detected = field.abs().square().float()
        aux = self.class_detector(detected)
        logits = aux["region_logits"]
        if return_diagnostics:
            diagnostics: dict[str, Any] = {
                "input_intensity": initial_intensity,
                "after_layers": intermediates,
                "detector_input": detected,
                **aux,
            }
            return logits, diagnostics
        if return_aux:
            return logits, aux
        return logits

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("Expected [B,C,H,W] image tensor")
        if image.shape[1] == 1:
            value = image[:, 0].float().clamp(0, 1)
        elif image.shape[1] == 3:
            weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            value = (image.float().clamp(0, 1) * weights).sum(1)
        else:
            raise ValueError("D2NN input must have one or three channels")
        value = F.interpolate(
            value.unsqueeze(1),
            size=(self.field_size, self.field_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        if self.input_energy_normalization == "rms":
            return value / value.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        if self.input_energy_normalization == "mean":
            return value / value.mean((-2, -1), keepdim=True).clamp_min(1e-6)
        return value


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(value))


def build_model(settings: Any) -> nn.Module:
    if settings.model_type == "standard_d2nn":
        return StandardD2NNTimeOfDayClassifier(
            settings.optical_field_size,
            settings.optical_padding_size,
            settings.wavelength_nm,
            settings.pixel_pitch_um,
            settings.mask_distance_cm,
            settings.phase_init,
            settings.phase_init_std,
            settings.num_classes,
            settings.class_names,
            settings.optical_layers,
            settings.detector_region_size,
            settings.detector_region_temperature,
            settings.input_energy_normalization,
        )
    if settings.model_type == "lenet5":
        return LeNet5(settings.num_classes)
    return _build_torchvision_model(settings.model_type, settings.num_classes, settings.pretrained)


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report: dict[str, Any] = {
        "model_name": type(model).__name__,
        "parameters": total,
        "trainable_parameters": trainable,
    }
    if isinstance(model, StandardD2NNTimeOfDayClassifier):
        phase_parameters = sum(layer.phase_mask.numel() for layer in model.layers)
        report.update(
            {
                "optical_layers": len(model.layers),
                "optical_field_size": model.field_size,
                "phase_mask_parameters": phase_parameters,
                "amplitude_mask_parameters": 0,
                "readout_parameters": 0,
                "readout": "fixed_detector_region_energy",
                "amplitude_modulation": False,
                "inter_layer_detection": False,
                "inter_layer_nonlinearity": False,
                "final_square_law_detection": True,
                "class_region_detector": model.class_detector.specification(),
                "input_energy_normalization": model.input_energy_normalization,
            }
        )
    return report


def _build_torchvision_model(model_type: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        from torchvision import models
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("A compatible torchvision installation is required") from exc

    if model_type == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_type == "vgg11_bn":
        weights = models.VGG11_BN_Weights.DEFAULT if pretrained else None
        model = models.vgg11_bn(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_type == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported torchvision model_type: {model_type}")
