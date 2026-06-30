from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .detector import DetectorArray
from .expert_layout import ExpertLayout
from .expert_phase_layer import ExpertPhaseLayer, GlobalFCPhaseMask
from .global_router_prompt import GlobalRouterPrompt
from .optical_models import CenterWindowPhaseLayer
from .readout import ElectronicReadout


class DatasetPromptBank(nn.Module):
    """Task-specific prompt amplitudes/biases over the same AS global-router geometry."""

    def __init__(
        self,
        task_names: Sequence[str],
        layout: ExpertLayout,
        wavelength_m: float,
        pixel_size_m: float,
        prompt_to_expert_m: float,
        focal_length_m: float,
        mode: str = "complex_order_router",
        amplitude_init_logits: float = 2.0,
        train_amplitudes: bool = True,
        train_phase_biases: bool = True,
        grating_scale: float = 1.0,
        grating_sign_x: float = 1.0,
        grating_sign_y: float = 1.0,
        normalize: str = "sum_amplitude",
    ) -> None:
        super().__init__()
        self.task_names = [str(name).lower() for name in task_names]
        if not self.task_names:
            raise ValueError("task_names must be non-empty.")
        self.prompts = nn.ModuleDict(
            {
                name: GlobalRouterPrompt(
                    layout=layout,
                    wavelength_m=wavelength_m,
                    pixel_size_m=pixel_size_m,
                    prompt_to_expert_m=prompt_to_expert_m,
                    focal_length_m=focal_length_m,
                    mode=mode,
                    amplitude_init_logits=amplitude_init_logits,
                    train_amplitudes=train_amplitudes,
                    train_phase_biases=train_phase_biases,
                    grating_scale=grating_scale,
                    grating_sign_x=grating_sign_x,
                    grating_sign_y=grating_sign_y,
                    normalize=normalize,
                )
                for name in self.task_names
            }
        )

    def normalize_name(self, task_name: str) -> str:
        name = str(task_name).lower()
        if name not in self.prompts:
            raise KeyError(f"Unknown task {task_name!r}; valid tasks: {self.task_names}")
        return name

    def resolve_name(self, task_name: Optional[str] = None, task_id: Optional[int] = None) -> str:
        if task_name is not None:
            return self.normalize_name(task_name)
        if task_id is not None:
            return self.task_names[int(task_id)]
        return self.task_names[0]

    def prompt(self, task_name: str) -> GlobalRouterPrompt:
        return self.prompts[self.normalize_name(task_name)]

    def transmission(self, task_name: str) -> torch.Tensor:
        return self.prompt(task_name).transmission()

    def amplitudes(self, task_name: str) -> torch.Tensor:
        return self.prompt(task_name).amplitudes()

    def powers(self, task_name: str) -> torch.Tensor:
        return self.prompt(task_name).powers()

    def normalized_powers(self, task_name: str) -> torch.Tensor:
        return self.prompt(task_name).normalized_powers()

    def prompt_maps(self, task_name: str) -> Dict[str, torch.Tensor]:
        return self.prompt(task_name).prompt_maps()

    def channel_table(self):
        return self.prompt(self.task_names[0]).channel_table()


class DatasetSwitchingASGlobalRouterMoEClassifier(nn.Module):
    """Shared 9-expert AS global-router MoE with task-specific prompt/readout heads."""

    def __init__(
        self,
        task_names: Sequence[str],
        task_num_classes: Dict[str, int],
        task_head_configs: Optional[Dict[str, Dict]] = None,
        layout: Optional[ExpertLayout] = None,
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
        global_fc_phase_mode: str = "center_window",
        global_fc_phase_size: Optional[int] = None,
        global_fc_padding_mode: str = "transparent",
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
        self.task_names = [str(name).lower() for name in task_names]
        self.task_num_classes = {str(key).lower(): int(value) for key, value in task_num_classes.items()}
        if set(self.task_names) != set(self.task_num_classes):
            raise ValueError("task_num_classes must match task_names exactly.")
        self.layout = layout or ExpertLayout(num_experts=9)
        self.layout.validate()
        self.canvas_shape = self.layout.canvas_shape
        self.input_size = int(self.layout.input_size)
        self.num_layers = int(num_layers)
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
        prop_args = {
            "wavelength_m": wavelength_m,
            "pixel_size_m": pixel_size_m,
            "grid_size": self.canvas_shape,
            "evanescent_mode": evanescent_mode,
        }
        self.input_to_prompt = AngularSpectrumPropagator(distance_m=self.distances_m["input_to_prompt"], **prop_args)
        self.prompt_bank = DatasetPromptBank(
            task_names=self.task_names,
            layout=self.layout,
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
                    self.layout,
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
            phase_size=global_fc_phase_size or self.layout.active_window_size,
            phase_mode=global_fc_phase_mode,
            padding_mode=global_fc_padding_mode,
            phase_param=phase_param,
            phase_init=global_fc_phase_init,
            init_std=global_fc_init_std,
            phase_dropout_mode=global_fc_phase_dropout_mode,
            phase_dropout_p=global_fc_phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )
        self.fc_to_detector = AngularSpectrumPropagator(distance_m=self.distances_m["fc_to_detector"], **prop_args)
        task_head_configs = task_head_configs or {}
        unknown_heads = sorted(set(task_head_configs) - set(self.task_names))
        if unknown_heads:
            raise ValueError(f"task_head_configs contains unknown tasks: {unknown_heads}; valid tasks: {self.task_names}")
        detectors = {}
        readouts = {}
        resolved = {}
        for task_name in self.task_names:
            head = dict(task_head_configs.get(task_name, {}))
            settings = {
                "detector_size": int(head.get("detector_size", detector_size)),
                "detector_layout": head.get("detector_layout", detector_layout),
                "normalize_detector_energy": bool(head.get("normalize_detector_energy", normalize_detector_energy)),
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
                num_classes=self.task_num_classes[task_name],
                grid_size=self.canvas_shape,
                detector_size=settings["detector_size"],
                layout=settings["detector_layout"],
                normalize_total_energy=settings["normalize_detector_energy"],
            )
            readouts[task_name] = ElectronicReadout(
                num_classes=self.task_num_classes[task_name],
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

    def _normalize_task(self, task_name: str) -> str:
        return self.prompt_bank.normalize_name(task_name)

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
        prompt_task = selected_task if prompt_task_name is None else self._normalize_task(prompt_task_name)
        readout_task = selected_task if readout_task_name is None else self._normalize_task(readout_task_name)
        canvas_input = self.prepare_canvas_input(images)
        after_input_to_prompt = self.input_to_prompt(canvas_input)
        after_prompt = after_input_to_prompt * self.prompt_bank.transmission(prompt_task).unsqueeze(0)
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
            "expert_entrance_before_aperture": expert_entrance_raw,
            "expert_entrance_after_aperture": expert_entrance,
            "expert_entrance_intensity": entrance_diag["intensity"],
            "expert_energy": entrance_diag["energy"],
            "expert_energy_ratios": entrance_diag["ratios"],
            "outside_energy_ratio": entrance_diag["outside_ratio"],
            "prompt_amplitudes": self.prompt_bank.amplitudes(prompt_task),
            "prompt_powers": self.prompt_bank.powers(prompt_task),
            "normalized_prompt_powers": self.prompt_bank.normalized_powers(prompt_task),
            "after_each_layer": layer_fields,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "global_fc_phase_region": self.global_fc.phase_region(),
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
        return self.expert_phase_parameter_count() + self.global_fc_parameter_count()

    def expert_phase_parameter_count(self) -> int:
        return sum(p.numel() for p in self.expert_layers.parameters())

    def global_fc_parameter_count(self) -> int:
        return int(self.global_fc.trainable_parameter_count())

    def prompt_parameter_count(self) -> int:
        return sum(p.numel() for p in self.prompt_bank.parameters())

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.task_readouts.parameters())

    def task_readout_parameter_counts(self) -> Dict[str, int]:
        return {name: sum(p.numel() for p in self.task_readouts[name].parameters()) for name in self.task_names}

    def task_detector_configs(self) -> Dict[str, Dict]:
        return {
            name: {
                "detector_size": self.task_head_configs[name]["detector_size"],
                "detector_layout": self.task_head_configs[name]["detector_layout"],
                "normalize_detector_energy": self.task_head_configs[name]["normalize_detector_energy"],
            }
            for name in self.task_names
        }

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.expert_layers:
            layer.set_phase_dropout_active(active)
        self.global_fc.set_phase_dropout_active(active)


class DatasetSwitchingSharedD2NNClassifier(nn.Module):
    """Shared non-MoE D2NN backbone with task-specific detector/readout heads."""

    def __init__(
        self,
        task_names: Sequence[str],
        task_num_classes: Dict[str, int],
        task_head_configs: Optional[Dict[str, Dict]] = None,
        canvas_size: int = 520,
        input_size: int = 120,
        d2nn_phase_grid_size: int = 360,
        num_layers: int = 5,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        distances_m: Optional[Dict[str, float]] = None,
        phase_param: str = "unconstrained",
        phase_init: str = "identity",
        init_std: float = 0.02,
        global_fc_phase_mode: str = "center_window",
        global_fc_phase_size: Optional[int] = None,
        global_fc_padding_mode: str = "transparent",
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
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        global_fc_phase_dropout_mode: str = "none",
        global_fc_phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.task_names = [str(name).lower() for name in task_names]
        self.task_num_classes = {str(key).lower(): int(value) for key, value in task_num_classes.items()}
        self.canvas_shape = (int(canvas_size), int(canvas_size))
        self.input_size = int(input_size)
        self.num_layers = int(num_layers)
        self.d2nn_phase_grid_size = int(d2nn_phase_grid_size)
        self.global_fc_phase_mode = str(global_fc_phase_mode)
        self.global_fc_phase_size = int(global_fc_phase_size or min(min(self.canvas_shape), 450))
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
        self.inter_props = nn.ModuleList(
            [AngularSpectrumPropagator(distance_m=self.distances_m["inter_layer"], **prop_args) for _ in range(max(0, self.num_layers - 1))]
        )
        self.layer5_to_fc = AngularSpectrumPropagator(distance_m=self.distances_m["layer5_to_fc"], **prop_args)
        self.global_fc = GlobalFCPhaseMask(
            self.canvas_shape,
            phase_size=self.global_fc_phase_size,
            phase_mode=self.global_fc_phase_mode,
            padding_mode=global_fc_padding_mode,
            phase_param=phase_param,
            phase_init=phase_init,
            init_std=init_std,
            phase_dropout_mode=global_fc_phase_dropout_mode,
            phase_dropout_p=global_fc_phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )
        self.fc_to_detector = AngularSpectrumPropagator(distance_m=self.distances_m["fc_to_detector"], **prop_args)
        task_head_configs = task_head_configs or {}
        unknown_heads = sorted(set(task_head_configs) - set(self.task_names))
        if unknown_heads:
            raise ValueError(f"task_head_configs contains unknown tasks: {unknown_heads}; valid tasks: {self.task_names}")
        self.task_detectors = nn.ModuleDict()
        self.task_readouts = nn.ModuleDict()
        self.task_head_configs = {}
        for task_name in self.task_names:
            head = dict(task_head_configs.get(task_name, {}))
            settings = {
                "detector_size": int(head.get("detector_size", detector_size)),
                "detector_layout": head.get("detector_layout", detector_layout),
                "normalize_detector_energy": bool(head.get("normalize_detector_energy", normalize_detector_energy)),
                "readout_type": head.get("readout_type", readout_type),
                "logit_scale": float(head.get("logit_scale", logit_scale)),
                "hidden_dim": int(head.get("hidden_dim", readout_hidden_dim)),
                "activation": head.get("activation", readout_activation),
                "input_norm": head.get("input_norm", readout_input_norm),
                "norm_affine": bool(head.get("norm_affine", readout_norm_affine)),
                "hidden_layers": int(head.get("hidden_layers", readout_hidden_layers)),
                "dropout": float(head.get("dropout", readout_dropout)),
            }
            self.task_detectors[task_name] = DetectorArray(self.task_num_classes[task_name], self.canvas_shape, settings["detector_size"], settings["detector_layout"], settings["normalize_detector_energy"])
            self.task_readouts[task_name] = ElectronicReadout(
                self.task_num_classes[task_name],
                readout_type=settings["readout_type"],
                logit_scale=settings["logit_scale"],
                hidden_dim=settings["hidden_dim"],
                activation=settings["activation"],
                input_norm=settings["input_norm"],
                norm_affine=settings["norm_affine"],
                hidden_layers=settings["hidden_layers"],
                dropout=settings["dropout"],
            )
            self.task_head_configs[task_name] = settings

    def _normalize_task(self, task_name: Optional[str]) -> str:
        name = self.task_names[0] if task_name is None else str(task_name).lower()
        if name not in self.task_readouts:
            raise KeyError(f"Unknown task {task_name!r}; valid tasks: {self.task_names}")
        return name

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

    def forward(self, images: torch.Tensor, task_name: Optional[str] = None, prompt_task_name: Optional[str] = None, readout_task_name: Optional[str] = None, return_intermediates: bool = False):
        selected_task = self._normalize_task(task_name)
        readout_task = self._normalize_task(readout_task_name or selected_task)
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
        detector_energies = self.task_detectors[readout_task](detector_field)
        logits = self.task_readouts[readout_task](detector_energies)
        if not return_intermediates:
            return logits
        intermediates = {
            "task_name": selected_task,
            "prompt_task_name": "",
            "readout_task_name": readout_task,
            "task_num_classes": self.task_num_classes[readout_task],
            "input_amplitude": canvas_input.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_each_layer": layer_fields,
            "after_expert_layer_1": layer_fields[0] if layer_fields else field,
            "after_expert_layer_last": layer_fields[-1] if layer_fields else field,
            "after_layer5_to_fc": after_layer5_to_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase_wrapped(),
            "global_fc_phase_region": self.global_fc.phase_region(),
            "detector_field": detector_field,
            "detector_intensity": torch.abs(detector_field).square(),
            "detector_energies": detector_energies,
            "logits": logits,
        }
        return logits, intermediates

    def optical_parameter_count(self) -> int:
        return int(self.num_layers) * int(self.d2nn_phase_grid_size) * int(self.d2nn_phase_grid_size) + int(self.global_fc.trainable_parameter_count())

    def prompt_parameter_count(self) -> int:
        return 0

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.task_readouts.parameters())

    def task_readout_parameter_counts(self) -> Dict[str, int]:
        return {name: sum(p.numel() for p in self.task_readouts[name].parameters()) for name in self.task_names}

    def task_detector_configs(self) -> Dict[str, Dict]:
        return {
            name: {
                "detector_size": self.task_head_configs[name]["detector_size"],
                "detector_layout": self.task_head_configs[name]["detector_layout"],
                "normalize_detector_energy": self.task_head_configs[name]["normalize_detector_energy"],
            }
            for name in self.task_names
        }

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.layers:
            layer.set_phase_dropout_active(active)
        self.global_fc.set_phase_dropout_active(active)
