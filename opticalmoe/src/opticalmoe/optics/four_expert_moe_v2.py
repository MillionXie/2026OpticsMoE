import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .detectors import DetectorArray
from .four_expert_geometry import FourExpertLayout
from .microlens_prompt import MicrolensArrayPrompt
from .phase_layers import PhaseLayer
from .readout import ElectronicReadout


class TrainableMicrolensArrayPromptV2(nn.Module):
    """Verified four-cell microlens geometry with trainable scalar controls."""

    def __init__(
        self,
        layout: FourExpertLayout,
        wavelength_m: float,
        pixel_size_m: float,
        focal_length_m: float,
        input_to_prompt_m: float,
        amplitude_init_logits: float = 2.0,
        train_phase_biases: bool = True,
    ) -> None:
        super().__init__()
        fixed = MicrolensArrayPrompt(
            layout=layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            focal_length_m=focal_length_m,
            input_to_prompt_m=input_to_prompt_m,
        )
        self.layout = layout
        self.amplitude_logits = nn.Parameter(
            torch.full((4,), float(amplitude_init_logits), dtype=torch.float32)
        )
        phase_biases = torch.zeros(4, dtype=torch.float32)
        if train_phase_biases:
            self.phase_biases = nn.Parameter(phase_biases)
        else:
            self.register_buffer("phase_biases", phase_biases)

        self.register_buffer(
            "cell_masks", fixed.cell_masks.detach().clone(), persistent=False
        )
        self.register_buffer(
            "lens_phases", fixed.lens_phases.detach().clone(), persistent=False
        )
        self.register_buffer(
            "grating_phases", fixed.grating_phases.detach().clone(), persistent=False
        )

    def amplitudes(self) -> torch.Tensor:
        return torch.sigmoid(self.amplitude_logits)

    def powers(self) -> torch.Tensor:
        return self.amplitudes().square()

    def normalized_powers(self) -> torch.Tensor:
        powers = self.powers()
        return powers / (powers.sum() + 1e-8)

    def amplitude_map(self) -> torch.Tensor:
        return torch.sum(
            self.cell_masks * self.amplitudes().view(4, 1, 1),
            dim=0,
        )

    def phase_map(self) -> torch.Tensor:
        phase = self.lens_phases + self.grating_phases
        phase = phase + self.cell_masks * self.phase_biases.view(4, 1, 1)
        return phase.sum(dim=0)

    def transmission(self) -> torch.Tensor:
        phase = (
            self.lens_phases
            + self.grating_phases
            + self.cell_masks * self.phase_biases.view(4, 1, 1)
        )
        cells = (
            self.cell_masks
            * self.amplitudes().view(4, 1, 1)
            * torch.exp(1j * phase).to(torch.complex64)
        )
        # Outside the four cells the complex transmission stays zero.
        return cells.sum(dim=0).to(torch.complex64)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field.to(torch.complex64) * self.transmission().unsqueeze(0)


class FourExpertPhaseLayer(nn.Module):
    """Four independent local phase masks embedded in fixed expert apertures."""

    def __init__(
        self,
        layout: FourExpertLayout,
        phase_param: str = "unconstrained",
        phase_init: str = "uniform_0_2pi",
        init_std: float = 0.02,
        aperture_mode: str = "hard",
    ) -> None:
        super().__init__()
        if aperture_mode not in {"hard", "transparent"}:
            raise ValueError("aperture_mode must be 'hard' or 'transparent'.")
        self.layout = layout
        self.aperture_mode = aperture_mode
        self.local_phases = nn.ModuleList(
            [
                PhaseLayer(
                    grid_size=layout.expert_size,
                    parameterization=phase_param,
                    init=phase_init,
                    init_std=init_std,
                )
                for _ in range(4)
            ]
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        output = (
            torch.zeros_like(field, dtype=torch.complex64)
            if self.aperture_mode == "hard"
            else field.to(torch.complex64).clone()
        )
        for aperture, phase_layer in zip(self.layout.experts, self.local_phases):
            local = field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
            output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = (
                phase_layer(local)
            )
        return output

    def get_phase_wrapped(self) -> torch.Tensor:
        return torch.stack(
            [layer.get_phase_wrapped() for layer in self.local_phases], dim=0
        )


class GlobalFCPhaseMask(nn.Module):
    """One trainable phase-only mask spanning the full optical canvas."""

    def __init__(
        self,
        canvas_shape: Tuple[int, int],
        phase_param: str = "unconstrained",
        phase_init: str = "identity",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.phase = PhaseLayer(
            grid_size=canvas_shape,
            parameterization=phase_param,
            init=phase_init,
            init_std=init_std,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.phase(field)

    def get_phase_wrapped(self) -> torch.Tensor:
        return self.phase.get_phase_wrapped()


class FourExpertMoEClassifierV2(nn.Module):
    """Trainable four-expert classifier using the verified microlens geometry."""

    def __init__(
        self,
        num_classes: int = 10,
        layout: Optional[FourExpertLayout] = None,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        input_size: int = 200,
        num_layers: int = 5,
        distances_m: Optional[Dict[str, float]] = None,
        focal_length_m: float = 0.10,
        aperture_mode: str = "hard",
        phase_param: str = "unconstrained",
        expert_phase_init: str = "uniform_0_2pi",
        expert_init_std: float = 0.02,
        global_fc_phase_init: str = "identity",
        global_fc_init_std: float = 0.02,
        prompt_amplitude_init_logits: float = 2.0,
        train_prompt_phase_biases: bool = True,
        detector_size: int = 32,
        detector_layout: str = "grid",
        normalize_detector_energy: bool = True,
        readout_type: str = "optical_only",
        logit_scale: float = 10.0,
        readout_hidden_dim: int = 64,
        readout_activation: str = "relu",
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.layout = (
            FourExpertLayout(prompt_cell_size=300) if layout is None else layout
        )
        self.layout.validate()
        self.canvas_shape = self.layout.canvas_shape
        self.aperture_mode = aperture_mode

        default_distances = {
            "input_to_prompt": 0.20,
            "prompt_to_expert": 0.20,
            "inter_layer": 0.05,
            "layer5_to_fc": 0.05,
            "fc_to_detector": 0.05,
        }
        self.distances_m = dict(default_distances)
        if distances_m:
            self.distances_m.update(
                {key: float(value) for key, value in distances_m.items()}
            )

        self.input_to_prompt = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["input_to_prompt"],
            evanescent_mode=evanescent_mode,
        )
        self.prompt = TrainableMicrolensArrayPromptV2(
            layout=self.layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            focal_length_m=focal_length_m,
            input_to_prompt_m=self.distances_m["input_to_prompt"],
            amplitude_init_logits=prompt_amplitude_init_logits,
            train_phase_biases=train_prompt_phase_biases,
        )
        self.prompt_to_expert = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["prompt_to_expert"],
            evanescent_mode=evanescent_mode,
        )
        self.expert_layers = nn.ModuleList(
            [
                FourExpertPhaseLayer(
                    layout=self.layout,
                    phase_param=phase_param,
                    phase_init=expert_phase_init,
                    init_std=expert_init_std,
                    aperture_mode=aperture_mode,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.inter_layer_propagators = nn.ModuleList(
            [
                AngularSpectrumPropagator(
                    wavelength_m=wavelength_m,
                    pixel_size_m=pixel_size_m,
                    grid_size=self.canvas_shape,
                    distance_m=self.distances_m["inter_layer"],
                    evanescent_mode=evanescent_mode,
                )
                for _ in range(max(0, self.num_layers - 1))
            ]
        )
        self.layer5_to_fc = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["layer5_to_fc"],
            evanescent_mode=evanescent_mode,
        )
        self.global_fc = GlobalFCPhaseMask(
            canvas_shape=self.canvas_shape,
            phase_param=phase_param,
            phase_init=global_fc_phase_init,
            init_std=global_fc_init_std,
        )
        self.fc_to_detector = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["fc_to_detector"],
            evanescent_mode=evanescent_mode,
        )
        self.detector = DetectorArray(
            num_classes=self.num_classes,
            grid_size=self.canvas_shape,
            detector_size=detector_size,
            layout=detector_layout,
            normalize_total_energy=normalize_detector_energy,
        )
        self.readout = ElectronicReadout(
            num_classes=self.num_classes,
            readout_type=readout_type,
            logit_scale=logit_scale,
            hidden_dim=readout_hidden_dim,
            activation=readout_activation,
        )
        self.register_buffer(
            "expert_masks", self.layout.expert_masks(), persistent=False
        )
        self.register_buffer(
            "expert_union_mask", self.layout.expert_union_mask(), persistent=False
        )

    @property
    def num_propagation_segments(self) -> int:
        return self.num_layers + 3

    def prepare_canvas_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.ndim != 4:
            raise ValueError(
                f"Expected images [B,C,H,W] or [B,H,W], got {tuple(images.shape)}"
            )
        images = images.float()
        if images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        images = images.clamp(0.0, 1.0)
        canvas = torch.zeros(
            images.shape[0],
            self.canvas_shape[0],
            self.canvas_shape[1],
            dtype=torch.float32,
            device=images.device,
        )
        aperture = self.layout.input_aperture
        canvas[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = images[:, 0]
        return canvas.to(torch.complex64)

    def _expert_energy_diagnostics(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        intensity = torch.abs(field.to(torch.complex64)).square()
        energies = torch.einsum(
            "bhw,khw->bk", intensity, self.expert_masks
        )
        total = intensity.sum(dim=(-2, -1))
        outside = (total - energies.sum(dim=1)).clamp_min(0.0)
        ratios = energies / (total.unsqueeze(1) + 1e-8)
        return {
            "intensity": intensity,
            "energy": energies,
            "ratios": ratios,
            "outside_ratio": outside / (total + 1e-8),
        }

    def forward(
        self,
        images: torch.Tensor,
        return_intermediates: bool = False,
    ):
        canvas_input = self.prepare_canvas_input(images)
        after_input_to_prompt = self.input_to_prompt(canvas_input)
        after_prompt = self.prompt(after_input_to_prompt)
        expert_entrance_raw = self.prompt_to_expert(after_prompt)
        entrance_diag = self._expert_energy_diagnostics(expert_entrance_raw)
        expert_entrance = expert_entrance_raw
        if self.aperture_mode == "hard":
            expert_entrance = (
                expert_entrance
                * self.expert_union_mask.unsqueeze(0).to(torch.complex64)
            )

        field = expert_entrance
        layer_fields = []
        for index, layer in enumerate(self.expert_layers):
            field = layer(field)
            if return_intermediates:
                layer_fields.append(field)
            if index < self.num_layers - 1:
                field = self.inter_layer_propagators[index](field)

        after_layer5_to_fc = self.layer5_to_fc(field)
        after_global_fc = self.global_fc(after_layer5_to_fc)
        detector_field = self.fc_to_detector(after_global_fc)
        detector_intensity = torch.abs(detector_field).square()
        detector_energies = self.detector(detector_field)
        logits = self.readout(detector_energies)

        if not return_intermediates:
            return logits

        prompt_amplitudes = self.prompt.amplitudes()
        prompt_powers = self.prompt.powers()
        intermediates = {
            "input_amplitude": canvas_input.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_prompt": after_prompt,
            "expert_entrance_field": expert_entrance_raw,
            "expert_entrance_after_aperture": expert_entrance,
            "expert_entrance_intensity": entrance_diag["intensity"],
            "expert_energy": entrance_diag["energy"],
            "expert_energy_ratios": entrance_diag["ratios"],
            "outside_energy_ratio": entrance_diag["outside_ratio"],
            "prompt_amplitudes": prompt_amplitudes,
            "prompt_powers": prompt_powers,
            "normalized_prompt_powers": self.prompt.normalized_powers(),
            "prompt_phase": self.prompt.phase_map(),
            "prompt_amplitude_map": self.prompt.amplitude_map(),
            "after_each_layer": layer_fields,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "detector_field": detector_field,
            "detector_intensity": detector_intensity,
            "detector_energies": detector_energies,
            "logits": logits,
        }
        for index, layer_field in enumerate(layer_fields, start=1):
            intermediates[f"after_expert_layer_{index}"] = layer_field
        return logits, intermediates

    def optical_parameter_count(self) -> int:
        modules = [self.prompt, self.expert_layers, self.global_fc]
        return sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
        )

    def electronic_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.readout.parameters())
