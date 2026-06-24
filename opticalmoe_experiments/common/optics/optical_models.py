from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .detector import DetectorArray
from .expert_layout import ExpertLayout
from .expert_phase_layer import ExpertPhaseLayer, GlobalFCPhaseMask
from .global_router_prompt import GlobalRouterPrompt
from .phase_layers import PhaseLayer
from .readout import ElectronicReadout


class CenterWindowPhaseLayer(nn.Module):
    """A full optical layer with trainable phase restricted to a center window."""

    def __init__(
        self,
        canvas_shape,
        phase_grid_size: int,
        phase_param: str,
        phase_init: str,
        init_std: float,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ) -> None:
        super().__init__()
        self.canvas_shape = tuple(canvas_shape)
        self.phase_grid_size = int(phase_grid_size)
        cy, cx = self.canvas_shape[0] // 2, self.canvas_shape[1] // 2
        half = self.phase_grid_size // 2
        self.y0 = cy - half
        self.y1 = self.y0 + self.phase_grid_size
        self.x0 = cx - half
        self.x1 = self.x0 + self.phase_grid_size
        self.phase = PhaseLayer(
            (self.phase_grid_size, self.phase_grid_size),
            parameterization=phase_param,
            init=phase_init,
            init_std=init_std,
            phase_dropout_mode=phase_dropout_mode,
            phase_dropout_p=phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = field.to(torch.complex64)
        local = field[:, self.y0:self.y1, self.x0:self.x1]
        local_out = self.phase(local)

        embedded = torch.zeros_like(field, dtype=torch.complex64)
        embedded[:, self.y0:self.y1, self.x0:self.x1] = local_out

        outside_mask = torch.ones(
            field.shape[-2:],
            dtype=torch.float32,
            device=field.device,
        )
        outside_mask[self.y0:self.y1, self.x0:self.x1] = 0.0
        return field * outside_mask.unsqueeze(0).to(torch.complex64) + embedded

    def get_phase_wrapped(self) -> torch.Tensor:
        return self.phase.get_phase_wrapped()

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase.set_phase_dropout_active(active)


class ASGlobalRouterMoEClassifier(nn.Module):
    """Single-task OpticalMoE using the validated AS global-router prompt."""

    def __init__(
        self,
        num_classes: int,
        layout: ExpertLayout,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        num_layers: int = 5,
        distances_m: Optional[Dict[str, float]] = None,
        focal_length_m: float = 0.10,
        aperture_mode: str = "hard",
        phase_param: str = "unconstrained",
        expert_phase_init: str = "identity",
        expert_init_std: float = 0.02,
        global_fc_phase_init: str = "identity",
        global_fc_init_std: float = 0.02,
        prompt_mode: str = "complex_order_router",
        prompt_amplitude_init_logits: float = 2.0,
        train_prompt_amplitudes: bool = True,
        train_prompt_phase_biases: bool = True,
        grating_scale: float = 1.0,
        grating_sign_x: float = 1.0,
        grating_sign_y: float = 1.0,
        prompt_normalize: str = "sum_amplitude",
        detector_size: int = 32,
        detector_layout: str = "grid",
        normalize_detector_energy: bool = True,
        readout_type: str = "mlp",
        logit_scale: float = 10.0,
        readout_hidden_dim: int = 64,
        readout_activation: str = "gelu",
        readout_input_norm: str = "layernorm",
        readout_norm_affine: bool = True,
        readout_hidden_layers: int = 1,
        readout_dropout: float = 0.1,
        expert_phase_dropout_mode: str = "none",
        expert_phase_dropout_p: float = 0.0,
        global_fc_phase_dropout_mode: str = "none",
        global_fc_phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        layout.validate()
        self.num_classes = int(num_classes)
        self.layout = layout
        self.input_size = int(layout.input_size)
        self.num_layers = int(num_layers)
        self.aperture_mode = aperture_mode
        self.canvas_shape = layout.canvas_shape
        defaults = {
            "input_to_prompt": 0.20,
            "prompt_to_expert": 0.20,
            "inter_layer": 0.05,
            "layer5_to_fc": 0.05,
            "fc_to_detector": 0.05,
        }
        self.distances_m = dict(defaults)
        if distances_m:
            self.distances_m.update({key: float(value) for key, value in distances_m.items()})

        prop_args = {
            "wavelength_m": wavelength_m,
            "pixel_size_m": pixel_size_m,
            "grid_size": self.canvas_shape,
            "evanescent_mode": evanescent_mode,
        }
        self.input_to_prompt = AngularSpectrumPropagator(distance_m=self.distances_m["input_to_prompt"], **prop_args)
        self.prompt = GlobalRouterPrompt(
            layout=layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            prompt_to_expert_m=self.distances_m["prompt_to_expert"],
            focal_length_m=focal_length_m,
            mode=prompt_mode,
            amplitude_init_logits=prompt_amplitude_init_logits,
            train_amplitudes=train_prompt_amplitudes,
            train_phase_biases=train_prompt_phase_biases,
            grating_scale=grating_scale,
            grating_sign_x=grating_sign_x,
            grating_sign_y=grating_sign_y,
            normalize=prompt_normalize,
        )
        self.prompt_to_expert = AngularSpectrumPropagator(distance_m=self.distances_m["prompt_to_expert"], **prop_args)
        self.expert_layers = nn.ModuleList(
            [
                ExpertPhaseLayer(
                    layout,
                    phase_param=phase_param,
                    phase_init=expert_phase_init,
                    init_std=expert_init_std,
                    aperture_mode=aperture_mode,
                    phase_dropout_mode=expert_phase_dropout_mode,
                    phase_dropout_p=expert_phase_dropout_p,
                    phase_dropout_block_size=phase_dropout_block_size,
                    phase_dropout_batch_shared=phase_dropout_batch_shared,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.inter_layer_propagators = nn.ModuleList(
            [AngularSpectrumPropagator(distance_m=self.distances_m["inter_layer"], **prop_args) for _ in range(max(0, self.num_layers - 1))]
        )
        self.layer5_to_fc = AngularSpectrumPropagator(distance_m=self.distances_m["layer5_to_fc"], **prop_args)
        self.global_fc = GlobalFCPhaseMask(
            self.canvas_shape,
            phase_param=phase_param,
            phase_init=global_fc_phase_init,
            init_std=global_fc_init_std,
            phase_dropout_mode=global_fc_phase_dropout_mode,
            phase_dropout_p=global_fc_phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )
        self.fc_to_detector = AngularSpectrumPropagator(distance_m=self.distances_m["fc_to_detector"], **prop_args)
        self.detector = DetectorArray(num_classes, self.canvas_shape, detector_size, detector_layout, normalize_detector_energy)
        self.readout = ElectronicReadout(
            num_classes=num_classes,
            readout_type=readout_type,
            logit_scale=logit_scale,
            hidden_dim=readout_hidden_dim,
            activation=readout_activation,
            input_norm=readout_input_norm,
            norm_affine=readout_norm_affine,
            hidden_layers=readout_hidden_layers,
            dropout=readout_dropout,
        )
        self.register_buffer("expert_masks", layout.expert_masks(), persistent=False)
        self.register_buffer("expert_union_mask", layout.expert_union_mask(), persistent=False)

    def prepare_canvas_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        images = F.interpolate(images.float(), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        aperture = self.layout.input_aperture
        canvas = torch.zeros(images.shape[0], *self.canvas_shape, dtype=torch.float32, device=images.device)
        canvas[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = images[:, 0].clamp(0.0, 1.0)
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

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        canvas_input = self.prepare_canvas_input(images)
        after_input_to_prompt = self.input_to_prompt(canvas_input)
        after_prompt = self.prompt(after_input_to_prompt)
        expert_entrance_raw = self.prompt_to_expert(after_prompt)
        entrance_diag = self._expert_diagnostics(expert_entrance_raw)
        expert_entrance = expert_entrance_raw
        if self.aperture_mode == "hard":
            expert_entrance = expert_entrance * self.expert_union_mask.unsqueeze(0).to(torch.complex64)
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
        detector_energies = self.detector(detector_field)
        logits = self.readout(detector_energies)
        if not return_intermediates:
            return logits
        maps = self.prompt.prompt_maps()
        intermediates = {
            "input_amplitude": canvas_input.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_prompt": after_prompt,
            "expert_entrance_before_aperture": expert_entrance_raw,
            "expert_entrance_after_aperture": expert_entrance,
            "expert_entrance_intensity": entrance_diag["intensity"],
            "expert_energy": entrance_diag["energy"],
            "expert_energy_ratios": entrance_diag["ratios"],
            "outside_energy_ratio": entrance_diag["outside_ratio"],
            "prompt_amplitudes": self.prompt.amplitudes(),
            "prompt_powers": self.prompt.powers(),
            "normalized_prompt_powers": self.prompt.normalized_powers(),
            "after_each_layer": layer_fields,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "detector_field": detector_field,
            "detector_intensity": torch.abs(detector_field).square(),
            "detector_energies": detector_energies,
            "logits": logits,
        }
        intermediates.update(maps)
        for layer_index, value in enumerate(layer_fields, start=1):
            intermediates[f"after_expert_layer_{layer_index}"] = value
        return logits, intermediates

    def optical_parameter_count(self) -> int:
        return sum(p.numel() for module in [self.prompt, self.expert_layers, self.global_fc] for p in module.parameters())

    def prompt_parameter_count(self) -> int:
        return sum(p.numel() for p in self.prompt.parameters())

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.readout.parameters())

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.expert_layers:
            layer.set_phase_dropout_active(active)
        self.global_fc.set_phase_dropout_active(active)


class GeneralD2NNClassifier(nn.Module):
    """Non-MoE D2NN baseline with parameter-matched center phase windows."""

    def __init__(
        self,
        num_classes: int,
        canvas_size: int = 1000,
        input_size: int = 134,
        d2nn_phase_grid_size: int = 402,
        num_layers: int = 5,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        distances_m: Optional[Dict[str, float]] = None,
        phase_param: str = "unconstrained",
        phase_init: str = "identity",
        init_std: float = 0.02,
        detector_size: int = 32,
        detector_layout: str = "grid",
        normalize_detector_energy: bool = True,
        readout_type: str = "mlp",
        readout_hidden_dim: int = 64,
        readout_activation: str = "gelu",
        readout_input_norm: str = "layernorm",
        readout_dropout: float = 0.1,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.canvas_shape = (int(canvas_size), int(canvas_size))
        self.num_layers = int(num_layers)
        self.d2nn_phase_grid_size = int(d2nn_phase_grid_size)
        defaults = {"input_to_prompt": 0.20, "inter_layer": 0.05, "layer5_to_fc": 0.05, "fc_to_detector": 0.05}
        self.distances_m = dict(defaults)
        if distances_m:
            self.distances_m.update({key: float(value) for key, value in distances_m.items()})
        prop_args = {
            "wavelength_m": wavelength_m,
            "pixel_size_m": pixel_size_m,
            "grid_size": self.canvas_shape,
            "evanescent_mode": evanescent_mode,
        }
        self.first_prop = AngularSpectrumPropagator(distance_m=self.distances_m["input_to_prompt"], **prop_args)
        self.layers = nn.ModuleList(
            [
                CenterWindowPhaseLayer(
                    self.canvas_shape,
                    self.d2nn_phase_grid_size,
                    phase_param,
                    phase_init,
                    init_std,
                    phase_dropout_mode=phase_dropout_mode,
                    phase_dropout_p=phase_dropout_p,
                    phase_dropout_block_size=phase_dropout_block_size,
                    phase_dropout_batch_shared=phase_dropout_batch_shared,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.inter_props = nn.ModuleList([AngularSpectrumPropagator(distance_m=self.distances_m["inter_layer"], **prop_args) for _ in range(max(0, self.num_layers - 1))])
        self.layer5_to_fc = AngularSpectrumPropagator(distance_m=self.distances_m["layer5_to_fc"], **prop_args)
        self.global_fc = GlobalFCPhaseMask(self.canvas_shape, phase_param=phase_param, phase_init=phase_init, init_std=init_std)
        self.fc_to_detector = AngularSpectrumPropagator(distance_m=self.distances_m["fc_to_detector"], **prop_args)
        self.detector = DetectorArray(num_classes, self.canvas_shape, detector_size, detector_layout, normalize_detector_energy)
        self.readout = ElectronicReadout(num_classes, readout_type=readout_type, hidden_dim=readout_hidden_dim, activation=readout_activation, input_norm=readout_input_norm, dropout=readout_dropout)

    def prepare_canvas_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        images = F.interpolate(images.float(), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        cy, cx = self.canvas_shape[0] // 2, self.canvas_shape[1] // 2
        half = self.input_size // 2
        canvas = torch.zeros(images.shape[0], *self.canvas_shape, dtype=torch.float32, device=images.device)
        canvas[:, cy - half:cy - half + self.input_size, cx - half:cx - half + self.input_size] = images[:, 0].clamp(0.0, 1.0)
        return canvas.to(torch.complex64)

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        canvas_input = self.prepare_canvas_input(images)
        after_input_to_prompt = self.first_prop(canvas_input)
        field = after_input_to_prompt
        layer_fields = []
        for index, layer in enumerate(self.layers):
            field = layer(field)
            if return_intermediates:
                layer_fields.append(field)
            if index < self.num_layers - 1:
                field = self.inter_props[index](field)
        after_layer5_to_fc = self.layer5_to_fc(field)
        after_global_fc = self.global_fc(after_layer5_to_fc)
        detector_field = self.fc_to_detector(after_global_fc)
        detector_energies = self.detector(detector_field)
        logits = self.readout(detector_energies)
        if not return_intermediates:
            return logits
        return logits, {
            "input_amplitude": canvas_input.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_each_layer": layer_fields,
            "after_expert_layer_1": layer_fields[0] if layer_fields else field,
            "after_expert_layer_last": layer_fields[-1] if layer_fields else field,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "detector_field": detector_field,
            "detector_intensity": torch.abs(detector_field).square(),
            "detector_energies": detector_energies,
            "logits": logits,
        }

    def optical_parameter_count(self) -> int:
        return sum(p.numel() for module in [self.layers, self.global_fc] for p in module.parameters())

    def prompt_parameter_count(self) -> int:
        return 0

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.readout.parameters())

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.layers:
            layer.set_phase_dropout_active(active)
        self.global_fc.set_phase_dropout_active(active)
