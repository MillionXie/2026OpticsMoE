from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .detectors import DetectorArray
from .four_expert_geometry import FourExpertLayout
from .four_expert_moe_v2 import FourExpertPhaseLayer, GlobalFCPhaseMask
from .microlens_prompt import MicrolensArrayPrompt
from .readout import ElectronicReadout


class TaskPromptBank(nn.Module):
    """Independent prompt amplitudes and phase biases for each named task."""

    def __init__(
        self,
        task_names: Sequence[str],
        amplitude_init_logits: float = 2.0,
        train_phase_biases: bool = True,
    ) -> None:
        super().__init__()
        names = [str(name).lower() for name in task_names]
        if not names or len(set(names)) != len(names):
            raise ValueError("task_names must be non-empty and unique.")
        self.task_names = names
        self.amplitude_logits = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.full(
                        (4,),
                        float(amplitude_init_logits),
                        dtype=torch.float32,
                    )
                )
                for name in names
            }
        )
        self.phase_biases = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.zeros(4, dtype=torch.float32),
                    requires_grad=bool(train_phase_biases),
                )
                for name in names
            }
        )

    def normalize_name(self, task_name: str) -> str:
        name = str(task_name).lower()
        if name not in self.amplitude_logits:
            raise KeyError(
                f"Unknown task '{task_name}'. Available tasks: {self.task_names}"
            )
        return name

    def resolve_name(
        self,
        task_name: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> str:
        if task_name is not None and task_id is not None:
            raise ValueError("Provide task_name or task_id, not both.")
        if task_name is not None:
            return self.normalize_name(task_name)
        if task_id is None:
            raise ValueError(
                "The multitask model requires task_name or task_id."
            )
        index = int(task_id)
        if index < 0 or index >= len(self.task_names):
            raise IndexError(
                f"task_id {index} is outside [0, {len(self.task_names) - 1}]."
            )
        return self.task_names[index]

    def controls(self, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.normalize_name(task_name)
        return self.amplitude_logits[name], self.phase_biases[name]

    def amplitudes(self, task_name: str) -> torch.Tensor:
        logits, _ = self.controls(task_name)
        return torch.sigmoid(logits)

    def powers(self, task_name: str) -> torch.Tensor:
        return self.amplitudes(task_name).square()

    def normalized_powers(self, task_name: str) -> torch.Tensor:
        powers = self.powers(task_name)
        return powers / (powers.sum() + 1e-8)


class SharedMicrolensPromptGeometry(nn.Module):
    """Fixed lens/grating geometry modulated by task-specific scalar controls."""

    def __init__(
        self,
        layout: FourExpertLayout,
        wavelength_m: float,
        pixel_size_m: float,
        focal_length_m: float,
        input_to_prompt_m: float,
    ) -> None:
        super().__init__()
        fixed = MicrolensArrayPrompt(
            layout=layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            focal_length_m=focal_length_m,
            input_to_prompt_m=input_to_prompt_m,
        )
        self.register_buffer(
            "cell_masks", fixed.cell_masks.detach().clone(), persistent=False
        )
        self.register_buffer(
            "lens_phases", fixed.lens_phases.detach().clone(), persistent=False
        )
        self.register_buffer(
            "grating_phases", fixed.grating_phases.detach().clone(), persistent=False
        )

    def amplitude_map(self, amplitudes: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            self.cell_masks * amplitudes.view(4, 1, 1),
            dim=0,
        )

    def phase_map(self, phase_biases: torch.Tensor) -> torch.Tensor:
        phase = (
            self.lens_phases
            + self.grating_phases
            + self.cell_masks * phase_biases.view(4, 1, 1)
        )
        return phase.sum(dim=0)

    def transmission(
        self,
        amplitudes: torch.Tensor,
        phase_biases: torch.Tensor,
    ) -> torch.Tensor:
        phase = (
            self.lens_phases
            + self.grating_phases
            + self.cell_masks * phase_biases.view(4, 1, 1)
        )
        cells = (
            self.cell_masks
            * amplitudes.view(4, 1, 1)
            * torch.exp(1j * phase).to(torch.complex64)
        )
        # The region outside the four 300 x 300 cells remains blocked.
        return cells.sum(dim=0).to(torch.complex64)

    def forward(
        self,
        field: torch.Tensor,
        amplitudes: torch.Tensor,
        phase_biases: torch.Tensor,
    ) -> torch.Tensor:
        return field.to(torch.complex64) * self.transmission(
            amplitudes, phase_biases
        ).unsqueeze(0)


class FourExpertMultitaskMoEClassifier(nn.Module):
    """Shared optical MoE backbone with one prompt controller per task."""

    def __init__(
        self,
        task_names: Sequence[str],
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
        self.layout = (
            FourExpertLayout(prompt_cell_size=300) if layout is None else layout
        )
        self.layout.validate()
        self.canvas_shape = self.layout.canvas_shape
        self.aperture_mode = aperture_mode

        defaults = {
            "input_to_prompt": 0.20,
            "prompt_to_expert": 0.20,
            "inter_layer": 0.05,
            "layer5_to_fc": 0.05,
            "fc_to_detector": 0.05,
        }
        self.distances_m = dict(defaults)
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
        self.prompt_geometry = SharedMicrolensPromptGeometry(
            layout=self.layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            focal_length_m=focal_length_m,
            input_to_prompt_m=self.distances_m["input_to_prompt"],
        )
        self.task_prompt_bank = TaskPromptBank(
            task_names=task_names,
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

    def prepare_canvas_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(images.shape)}")
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
        canvas = torch.zeros(
            images.shape[0],
            self.canvas_shape[0],
            self.canvas_shape[1],
            dtype=torch.float32,
            device=images.device,
        )
        aperture = self.layout.input_aperture
        canvas[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = (
            images[:, 0].clamp(0.0, 1.0)
        )
        return canvas.to(torch.complex64)

    def _expert_diagnostics(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        intensity = torch.abs(field.to(torch.complex64)).square()
        energies = torch.einsum("bhw,khw->bk", intensity, self.expert_masks)
        total = intensity.sum(dim=(-2, -1))
        outside = (total - energies.sum(dim=1)).clamp_min(0.0)
        return {
            "intensity": intensity,
            "energy": energies,
            "ratios": energies / (total.unsqueeze(1) + 1e-8),
            "outside_ratio": outside / (total + 1e-8),
        }

    def forward(
        self,
        images: torch.Tensor,
        task_name: Optional[str] = None,
        task_id: Optional[int] = None,
        return_intermediates: bool = False,
    ):
        task_name = self.task_prompt_bank.resolve_name(
            task_name=task_name,
            task_id=task_id,
        )
        _, phase_biases = self.task_prompt_bank.controls(task_name)
        amplitudes = self.task_prompt_bank.amplitudes(task_name)

        canvas_input = self.prepare_canvas_input(images)
        after_input_to_prompt = self.input_to_prompt(canvas_input)
        after_prompt = self.prompt_geometry(
            after_input_to_prompt, amplitudes, phase_biases
        )
        expert_entrance_raw = self.prompt_to_expert(after_prompt)
        entrance_diag = self._expert_diagnostics(expert_entrance_raw)
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

        intermediates = {
            "task_name": task_name,
            "input_amplitude": canvas_input.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_prompt": after_prompt,
            "expert_entrance_field": expert_entrance_raw,
            "expert_entrance_after_aperture": expert_entrance,
            "expert_entrance_intensity": entrance_diag["intensity"],
            "expert_energy": entrance_diag["energy"],
            "expert_energy_ratios": entrance_diag["ratios"],
            "outside_energy_ratio": entrance_diag["outside_ratio"],
            "prompt_amplitudes": amplitudes,
            "prompt_powers": self.task_prompt_bank.powers(task_name),
            "normalized_prompt_powers": self.task_prompt_bank.normalized_powers(
                task_name
            ),
            "prompt_phase": self.prompt_geometry.phase_map(phase_biases),
            "prompt_amplitude_map": self.prompt_geometry.amplitude_map(amplitudes),
            "after_each_layer": layer_fields,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "detector_field": detector_field,
            "detector_intensity": detector_intensity,
            "detector_energies": detector_energies,
            "logits": logits,
        }
        return logits, intermediates

    def optical_parameter_count(self) -> int:
        modules = [self.task_prompt_bank, self.expert_layers, self.global_fc]
        return sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
        )

    def prompt_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.task_prompt_bank.parameters()
        )

    def electronic_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.readout.parameters())
