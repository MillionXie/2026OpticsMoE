from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .as_global_router_prompt import ASGlobalRouterPromptBank
from .detectors import DetectorArray
from .four_expert_moe_v2 import GlobalFCPhaseMask
from .nine_expert_geometry import NineExpertFair134Layout
from .nine_expert_phase_layer import NineExpertPhaseLayer
from .readout import ElectronicReadout


class NineExpertASGlobalRouterMultitaskMoEClassifier(nn.Module):
    """Trainable 9-expert fair134 AS global-router multitask OpticalMoE."""

    def __init__(
        self,
        task_names: Sequence[str],
        task_num_classes: Dict[str, int],
        task_head_configs: Optional[Dict[str, Dict]] = None,
        layout: Optional[NineExpertFair134Layout] = None,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        input_size: int = 134,
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
        train_prompt_phase_biases: bool = True,
        grating_scale: float = 1.0,
        grating_sign_x: float = 1.0,
        grating_sign_y: float = 1.0,
        prompt_normalize: str = "sum_amplitude",
        detector_size: int = 32,
        detector_layout: str = "grid",
        normalize_detector_energy: bool = True,
        readout_type: str = "optical_only",
        logit_scale: float = 10.0,
        readout_hidden_dim: int = 64,
        readout_activation: str = "relu",
        readout_input_norm: str = "none",
        readout_norm_affine: bool = True,
        readout_hidden_layers: int = 1,
        readout_dropout: float = 0.0,
        expert_phase_dropout_mode: str = "none",
        expert_phase_dropout_p: float = 0.0,
        global_fc_phase_dropout_mode: str = "none",
        global_fc_phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.task_names = [str(name).lower() for name in task_names]
        if not self.task_names:
            raise ValueError("task_names must be non-empty.")
        self.task_num_classes = {
            str(name).lower(): int(count) for name, count in task_num_classes.items()
        }
        if set(self.task_names) != set(self.task_num_classes):
            raise ValueError("task_num_classes must match task_names exactly.")
        self.num_classes = max(self.task_num_classes.values())
        self.input_size = int(input_size)
        self.num_layers = int(num_layers)
        self.layout = NineExpertFair134Layout() if layout is None else layout
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
            self.distances_m.update({key: float(value) for key, value in distances_m.items()})

        self.input_to_prompt = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["input_to_prompt"],
            evanescent_mode=evanescent_mode,
        )
        self.prompt_bank = ASGlobalRouterPromptBank(
            task_names=self.task_names,
            layout=self.layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            prompt_to_expert_m=self.distances_m["prompt_to_expert"],
            focal_length_m=focal_length_m,
            amplitude_init_logits=prompt_amplitude_init_logits,
            train_phase_biases=train_prompt_phase_biases,
            prompt_mode=prompt_mode,
            grating_scale=grating_scale,
            grating_sign_x=grating_sign_x,
            grating_sign_y=grating_sign_y,
            normalize=prompt_normalize,
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
                NineExpertPhaseLayer(
                    layout=self.layout,
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
            phase_dropout_mode=global_fc_phase_dropout_mode,
            phase_dropout_p=global_fc_phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )
        self.fc_to_detector = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["fc_to_detector"],
            evanescent_mode=evanescent_mode,
        )

        task_head_configs = task_head_configs or {}
        detectors = {}
        readouts = {}
        resolved = {}
        for task_name, num_classes in self.task_num_classes.items():
            head = dict(task_head_configs.get(task_name, {}))
            settings = {
                "detector_size": int(head.get("detector_size", detector_size)),
                "detector_layout": head.get("detector_layout", detector_layout),
                "normalize_detector_energy": bool(
                    head.get("normalize_detector_energy", normalize_detector_energy)
                ),
                "readout_type": head.get("readout_type", readout_type),
                "logit_scale": float(head.get("logit_scale", logit_scale)),
                "hidden_dim": int(head.get("hidden_dim", readout_hidden_dim)),
                "activation": head.get("activation", readout_activation),
                "input_norm": head.get("input_norm", readout_input_norm),
                "norm_affine": bool(head.get("norm_affine", readout_norm_affine)),
                "hidden_layers": int(head.get("hidden_layers", readout_hidden_layers)),
                "dropout": float(head.get("dropout", readout_dropout)),
            }
            detectors[task_name] = DetectorArray(
                num_classes=num_classes,
                grid_size=self.canvas_shape,
                detector_size=settings["detector_size"],
                layout=settings["detector_layout"],
                normalize_total_energy=settings["normalize_detector_energy"],
            )
            readouts[task_name] = ElectronicReadout(
                num_classes=num_classes,
                readout_type=settings["readout_type"],
                logit_scale=settings["logit_scale"],
                hidden_dim=settings["hidden_dim"],
                activation=settings["activation"],
                input_norm=settings["input_norm"],
                norm_affine=settings["norm_affine"],
                hidden_layers=settings["hidden_layers"],
                dropout=settings["dropout"],
            )
            resolved[task_name] = settings
        self.task_head_configs = resolved
        self.task_detectors = nn.ModuleDict(detectors)
        self.task_readouts = nn.ModuleDict(readouts)
        self.register_buffer("expert_masks", self.layout.expert_masks(), persistent=False)
        self.register_buffer("expert_union_mask", self.layout.expert_union_mask(), persistent=False)

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
        aperture = self.layout.input_aperture
        canvas = torch.zeros(
            images.shape[0],
            self.canvas_shape[0],
            self.canvas_shape[1],
            dtype=torch.float32,
            device=images.device,
        )
        canvas[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = images[:, 0].clamp(0.0, 1.0)
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
        prompt_task_name: Optional[str] = None,
        readout_task_name: Optional[str] = None,
        return_intermediates: bool = False,
    ):
        selected_task = self.prompt_bank.resolve_name(task_name=task_name, task_id=task_id)
        prompt_task = selected_task if prompt_task_name is None else self.prompt_bank.normalize_name(prompt_task_name)
        readout_task = selected_task if readout_task_name is None else self.prompt_bank.normalize_name(readout_task_name)

        canvas_input = self.prepare_canvas_input(images)
        after_input_to_prompt = self.input_to_prompt(canvas_input)
        prompt_transmission = self.prompt_bank.transmission(prompt_task)
        after_prompt = after_input_to_prompt * prompt_transmission.unsqueeze(0)
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
        detector_intensity = torch.abs(detector_field).square()
        detector_energies = self.task_detectors[readout_task](detector_field)
        logits = self.task_readouts[readout_task](detector_energies)
        if not return_intermediates:
            return logits

        maps = self.prompt_bank.prompt_maps(prompt_task)
        intermediates = {
            "task_name": selected_task,
            "prompt_task_name": prompt_task,
            "readout_task_name": readout_task,
            "task_num_classes": self.task_num_classes[readout_task],
            "input_amplitude": canvas_input.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_prompt": after_prompt,
            "expert_entrance_field": expert_entrance_raw,
            "expert_entrance_after_aperture": expert_entrance,
            "expert_entrance_intensity": entrance_diag["intensity"],
            "expert_energy": entrance_diag["energy"],
            "expert_energy_ratios": entrance_diag["ratios"],
            "outside_energy_ratio": entrance_diag["outside_ratio"],
            "prompt_amplitudes": self.prompt_bank.amplitudes(prompt_task),
            "prompt_powers": self.prompt_bank.powers(prompt_task),
            "normalized_prompt_powers": self.prompt_bank.normalized_powers(prompt_task),
            "prompt_router_amplitude": maps["prompt_router_amplitude"],
            "prompt_router_phase": maps["prompt_router_phase"],
            "prompt_total_amplitude": maps["prompt_total_amplitude"],
            "prompt_total_phase": maps["prompt_total_phase"],
            "prompt_amplitude_map": maps["prompt_total_amplitude"],
            "prompt_phase": maps["prompt_total_phase"],
            "prompt_channel_table": self.prompt_bank.channel_table(),
            "after_each_layer": layer_fields,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "detector_field": detector_field,
            "detector_intensity": detector_intensity,
            "detector_energies": detector_energies,
            "logits": logits,
        }
        for layer_index, layer_field in enumerate(layer_fields, start=1):
            intermediates[f"after_expert_layer_{layer_index}"] = layer_field
        return logits, intermediates

    def optical_parameter_count(self) -> int:
        modules = [self.prompt_bank, self.expert_layers, self.global_fc]
        return sum(p.numel() for module in modules for p in module.parameters())

    def prompt_parameter_count(self) -> int:
        return sum(p.numel() for p in self.prompt_bank.parameters())

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.task_readouts.parameters())

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.expert_layers:
            layer.set_phase_dropout_active(active)
        self.global_fc.set_phase_dropout_active(active)
