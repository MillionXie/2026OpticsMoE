import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .angular_spectrum import AngularSpectrumPropagator
from .grating import (
    build_aperture_mask,
    build_detilt_phase_for_aperture,
    build_expert_aperture_union,
    build_linear_grating_phase,
    compute_steering_params,
)
from .moe_layout import MoeLayout
from .phase_layers import PhaseLayer
from .translated_detectors import TranslatedDetectorArray


class ExpertBankPhaseLayer(nn.Module):
    """One optical layer containing local left/right expert phase masks."""

    def __init__(
        self,
        layout: MoeLayout,
        parameterization: str = "unconstrained",
        init: str = "uniform",
    ) -> None:
        super().__init__()
        self.layout = layout
        self.left_phase = PhaseLayer(layout.expert_size, parameterization=parameterization, init=init)
        self.right_phase = PhaseLayer(layout.expert_size, parameterization=parameterization, init=init)

    def _apply_side(self, out: torch.Tensor, field: torch.Tensor, side: str) -> None:
        aperture = self.layout.aperture_for_side(side)
        layer = self.left_phase if side == "left" else self.right_phase
        phase = layer.get_phase().to(field.device)
        modulation = torch.exp(1j * phase).to(torch.complex64)
        out[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = (
            field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1].to(torch.complex64)
            * modulation
        )

    def forward(self, field: torch.Tensor, use_aperture_masks: bool = True) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected field shape [B, H, W], got {tuple(field.shape)}")

        out = torch.zeros_like(field.to(torch.complex64)) if use_aperture_masks else field.to(torch.complex64).clone()
        self._apply_side(out, field, "left")
        self._apply_side(out, field, "right")
        return out.to(torch.complex64)

    def set_side_requires_grad(self, side: str, requires_grad: bool) -> None:
        layer = self.left_phase if side == "left" else self.right_phase
        for param in layer.parameters():
            param.requires_grad = bool(requires_grad)

    def get_phase_wrapped(self, side: str) -> torch.Tensor:
        layer = self.left_phase if side == "left" else self.right_phase
        return layer.get_phase_wrapped()


class OpticalMoEClassifier(nn.Module):
    """Large-canvas optical MoE classifier with translated paired detectors.

    The first implementation supports fixed grating routing, entrance de-tilt,
    single-side bank evaluation/training, and paired 10-class detector summation.
    It intentionally does not implement output grating recombination.
    """

    def __init__(
        self,
        num_classes: int = 10,
        layout: Optional[MoeLayout] = None,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        distances_m: Optional[Dict[str, float]] = None,
        num_layers: int = 5,
        phase_param: str = "unconstrained",
        phase_init: str = "uniform",
        detector_size: int = 32,
        detector_layout: str = "grid",
        readout_mode: str = "auto",
        detector_normalization: str = "local",
        logit_scale: float = 10.0,
        mode: str = "single_side",
        prompt_mode: str = "fixed_grating",
        prompt_init: str = "fixed_grating",
        target_side: Optional[str] = "left",
        prompt_slope_sign: int = 1,
        use_entrance_detilt: bool = True,
        use_aperture_masks: bool = True,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.layout = layout or MoeLayout()
        self.layout.validate()
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.num_layers = int(num_layers)
        self.phase_param = phase_param
        self.mode = mode
        self.prompt_mode = prompt_mode
        self.prompt_init = prompt_init
        self.target_side = target_side
        self.prompt_slope_sign = int(prompt_slope_sign)
        self.use_entrance_detilt = bool(use_entrance_detilt)
        self.use_aperture_masks = bool(use_aperture_masks)
        self.detector_normalization = detector_normalization
        self.logit_scale = float(logit_scale)
        self.migration_summaries: List[Dict] = []

        if mode not in {"single_side", "dual_expert_paired_sum", "diagnostic"}:
            raise ValueError("mode must be single_side, dual_expert_paired_sum, or diagnostic")
        if target_side is not None and target_side not in {"left", "right"}:
            raise ValueError("target_side must be 'left', 'right', or None")

        distances_m = distances_m or {
            "input_to_prompt": 0.01,
            "prompt_to_first_layer": 0.24,
            "inter_layer": 0.05,
            "last_layer_to_detector": 0.05,
        }
        self.distances_m = dict(distances_m)

        shift_pixels = self.layout.left_shift_pixels if target_side == "left" else self.layout.right_shift_pixels
        self.steering_params = compute_steering_params(
            wavelength_m=self.wavelength_m,
            pixel_size_m=self.pixel_size_m,
            shift_pixels=shift_pixels,
            distance_m=self.distances_m["prompt_to_first_layer"],
            inter_layer_m=self.distances_m["inter_layer"],
        )

        self.expert_layers = nn.ModuleList(
            [
                ExpertBankPhaseLayer(
                    layout=self.layout,
                    parameterization=phase_param,
                    init=phase_init,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.detector = TranslatedDetectorArray(
            num_classes=num_classes,
            layout=self.layout,
            detector_size=detector_size,
            detector_layout=detector_layout,
        )
        self.detector_size = int(detector_size)
        self.detector_layout = detector_layout

        self.propagators = nn.ModuleList(self._build_propagators(evanescent_mode))
        if len(self.propagators) != self.num_layers + 2:
            raise RuntimeError("Number of propagation segments must equal num_layers + 2.")

        aperture_union = build_expert_aperture_union(self.layout.canvas_shape, self.layout.left, self.layout.right)
        self.register_buffer("aperture_union_mask", aperture_union, persistent=False)
        self.register_buffer("left_aperture_mask", build_aperture_mask(self.layout.canvas_shape, self.layout.left), persistent=False)
        self.register_buffer("right_aperture_mask", build_aperture_mask(self.layout.canvas_shape, self.layout.right), persistent=False)

        fixed_prompt = self._build_fixed_prompt_phase(target_side=target_side)
        self.register_buffer("fixed_prompt_phase", fixed_prompt, persistent=False)
        detilt_phase = self._build_detilt_phase(target_side=target_side)
        self.register_buffer("entrance_detilt_phase", detilt_phase, persistent=False)

        self.prompt_raw_phase: Optional[nn.Parameter]
        if prompt_mode == "trainable_phase":
            self.prompt_raw_phase = nn.Parameter(self._init_prompt_parameter(prompt_init, fixed_prompt))
        elif prompt_mode == "trainable_residual_on_grating":
            self.prompt_raw_phase = nn.Parameter(torch.zeros(self.layout.canvas_shape, dtype=torch.float32))
        else:
            self.prompt_raw_phase = None

        self.readout_mode = self._resolve_readout_mode(readout_mode)

    @property
    def num_propagation_segments(self) -> int:
        return len(self.propagators)

    def _build_propagators(self, evanescent_mode: str) -> List[AngularSpectrumPropagator]:
        distances = [
            self.distances_m["input_to_prompt"],
            self.distances_m["prompt_to_first_layer"],
        ]
        distances.extend([self.distances_m["inter_layer"]] * max(0, self.num_layers - 1))
        distances.append(self.distances_m["last_layer_to_detector"])
        return [
            AngularSpectrumPropagator(
                wavelength_m=self.wavelength_m,
                pixel_size_m=self.pixel_size_m,
                grid_size=self.layout.canvas_shape,
                distance_m=distance,
                evanescent_mode=evanescent_mode,
            )
            for distance in distances
        ]

    def _build_fixed_prompt_phase(self, target_side: Optional[str]) -> torch.Tensor:
        if target_side not in {"left", "right"}:
            return torch.zeros(self.layout.canvas_shape, dtype=torch.float32)
        return build_linear_grating_phase(
            canvas_shape=self.layout.canvas_shape,
            period_px=self.steering_params.grating_period_px,
            direction=target_side,
            slope_sign=self.prompt_slope_sign,
        )

    def _build_detilt_phase(self, target_side: Optional[str]) -> torch.Tensor:
        if target_side not in {"left", "right"} or not self.use_entrance_detilt:
            return torch.zeros(self.layout.canvas_shape, dtype=torch.float32)
        aperture = self.layout.aperture_for_side(target_side)
        return build_detilt_phase_for_aperture(
            canvas_shape=self.layout.canvas_shape,
            aperture=aperture,
            period_px=self.steering_params.grating_period_px,
            direction=target_side,
            prompt_slope_sign=self.prompt_slope_sign,
        )

    def _init_prompt_parameter(self, prompt_init: str, fixed_prompt: torch.Tensor) -> torch.Tensor:
        if prompt_init == "zeros":
            return torch.zeros_like(fixed_prompt)
        if prompt_init == "random":
            return torch.empty_like(fixed_prompt).uniform_(0.0, 2.0 * math.pi)
        if prompt_init == "fixed_grating":
            return fixed_prompt.clone()
        if prompt_init == "fixed_grating_plus_noise":
            return fixed_prompt.clone() + 0.01 * torch.randn_like(fixed_prompt)
        raise ValueError(f"Unsupported prompt_init: {prompt_init}")

    def _resolve_readout_mode(self, readout_mode: str) -> str:
        if readout_mode != "auto":
            return readout_mode
        if self.mode == "single_side" and self.target_side == "left":
            return "left_only"
        if self.mode == "single_side" and self.target_side == "right":
            return "right_only"
        return "paired_sum_global"

    def get_prompt_phase(self) -> torch.Tensor:
        if self.prompt_mode == "identity":
            return torch.zeros_like(self.fixed_prompt_phase)
        if self.prompt_mode == "fixed_grating":
            return self.fixed_prompt_phase
        if self.prompt_mode == "trainable_phase":
            return self.prompt_raw_phase
        if self.prompt_mode == "trainable_residual_on_grating":
            return self.fixed_prompt_phase + self.prompt_raw_phase
        raise ValueError(f"Unsupported prompt_mode: {self.prompt_mode}")

    def set_fixed_routing_side(self, target_side: str) -> None:
        """Switch fixed grating/de-tilt routing side for diagnostic evaluation.

        This is intended for oracle/task-id evaluation before a trainable router
        exists. It does not change trainable expert phases.
        """

        if target_side not in {"left", "right"}:
            raise ValueError("target_side must be 'left' or 'right'")
        self.target_side = target_side
        shift_pixels = self.layout.left_shift_pixels if target_side == "left" else self.layout.right_shift_pixels
        self.steering_params = compute_steering_params(
            wavelength_m=self.wavelength_m,
            pixel_size_m=self.pixel_size_m,
            shift_pixels=shift_pixels,
            distance_m=self.distances_m["prompt_to_first_layer"],
            inter_layer_m=self.distances_m["inter_layer"],
        )
        device = self.fixed_prompt_phase.device
        self.fixed_prompt_phase = self._build_fixed_prompt_phase(target_side=target_side).to(device)
        self.entrance_detilt_phase = self._build_detilt_phase(target_side=target_side).to(device)
        if self.readout_mode in {"left_only", "right_only"}:
            self.readout_mode = f"{target_side}_only"

    def prepare_canvas_input(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.ndim == 4:
            if images.shape[1] != 1:
                raise ValueError(f"Expected one-channel images, got {tuple(images.shape)}")
            images = images[:, 0]
        elif images.ndim != 3:
            raise ValueError(f"Expected images [B,1,H,W] or [B,H,W], got {tuple(images.shape)}")

        images = images.float()
        if images.shape[-2:] != (self.layout.input_size, self.layout.input_size):
            images = F.interpolate(
                images.unsqueeze(1),
                size=(self.layout.input_size, self.layout.input_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        amplitude = torch.clamp(images, 0.0, 1.0)

        batch_size = amplitude.shape[0]
        canvas = torch.zeros(
            batch_size,
            self.layout.canvas_height,
            self.layout.canvas_width,
            dtype=torch.float32,
            device=amplitude.device,
        )
        aperture = self.layout.input_aperture
        canvas[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = amplitude
        return amplitude, canvas.to(torch.complex64)

    def _apply_aperture_union(self, field: torch.Tensor) -> torch.Tensor:
        if not self.use_aperture_masks:
            return field.to(torch.complex64)
        return field.to(torch.complex64) * self.aperture_union_mask.to(field.device).to(torch.complex64)

    def _readout_logits(self, detector_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.readout_mode
        if mode == "left_only":
            key = "left_local_norm" if self.detector_normalization == "local" else "left_global_norm"
            logits = detector_outputs[key]
        elif mode == "right_only":
            key = "right_local_norm" if self.detector_normalization == "local" else "right_global_norm"
            logits = detector_outputs[key]
        elif mode == "paired_sum_global":
            logits = detector_outputs["paired_sum_global"]
        elif mode == "paired_sum_raw":
            logits = detector_outputs["paired_sum_raw"]
        elif mode == "energy_gated_local":
            logits = detector_outputs["energy_gated_local"]
        else:
            raise ValueError(f"Unsupported readout_mode: {mode}")
        return logits * self.logit_scale

    def _batch_centroid(self, field: torch.Tensor) -> torch.Tensor:
        image = torch.abs(field.to(torch.complex64)) ** 2 if torch.is_complex(field) else field.float()
        height, width = image.shape[-2:]
        y = torch.arange(height, dtype=image.dtype, device=image.device).view(1, height, 1)
        x = torch.arange(width, dtype=image.dtype, device=image.device).view(1, 1, width)
        total = image.sum(dim=(-2, -1)) + 1e-8
        cy = (image * y).sum(dim=(-2, -1)) / total
        cx = (image * x).sum(dim=(-2, -1)) / total
        return torch.stack([cy, cx], dim=1)

    def _edge_energy_ratio(self, field: torch.Tensor, border: int = 50) -> torch.Tensor:
        image = torch.abs(field.to(torch.complex64)) ** 2 if torch.is_complex(field) else field.float()
        mask = torch.zeros(image.shape[-2:], dtype=image.dtype, device=image.device)
        mask[:border, :] = 1.0
        mask[-border:, :] = 1.0
        mask[:, :border] = 1.0
        mask[:, -border:] = 1.0
        total = image.sum(dim=(-2, -1)) + 1e-8
        return (image * mask).sum(dim=(-2, -1)) / total

    def _add_plane_diagnostics(self, diagnostics: Dict, name: str, field: torch.Tensor) -> None:
        diagnostics["centroid_per_plane"][name] = self._batch_centroid(field).detach()
        diagnostics["edge_energy_ratio_per_plane"][name] = self._edge_energy_ratio(field).detach()

    def optical_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def electronic_parameter_count(self) -> int:
        return 0

    def set_all_requires_grad(self, requires_grad: bool) -> None:
        for param in self.parameters():
            param.requires_grad = bool(requires_grad)

    def set_side_requires_grad(self, side: str, requires_grad: bool) -> None:
        for layer in self.expert_layers:
            layer.set_side_requires_grad(side, requires_grad)

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        intermediates = {"centroid_per_plane": {}, "edge_energy_ratio_per_plane": {}} if return_intermediates else None

        input_amplitude, field = self.prepare_canvas_input(images)
        if return_intermediates:
            intermediates["input_amplitude"] = input_amplitude.detach()
            intermediates["padded_input_on_canvas"] = field.detach()
            self._add_plane_diagnostics(intermediates, "padded_input_on_canvas", field)

        field = self.propagators[0](field)
        if return_intermediates:
            intermediates["after_input_to_prompt"] = field.detach()
            self._add_plane_diagnostics(intermediates, "after_input_to_prompt", field)

        prompt_phase = self.get_prompt_phase().to(field.device)
        field = field * torch.exp(1j * prompt_phase).to(torch.complex64)
        if return_intermediates:
            intermediates["prompt_phase"] = prompt_phase.detach()
            intermediates["after_prompt"] = field.detach()
            self._add_plane_diagnostics(intermediates, "after_prompt", field)

        field = self.propagators[1](field)
        if return_intermediates:
            intermediates["after_prompt_to_first_layer"] = field.detach()
            intermediates["before_entrance_detilt"] = field.detach()
            self._add_plane_diagnostics(intermediates, "after_prompt_to_first_layer", field)

        field = self._apply_aperture_union(field)
        if self.use_entrance_detilt:
            detilt_phase = self.entrance_detilt_phase.to(field.device)
            field = field * torch.exp(1j * detilt_phase).to(torch.complex64)
        if return_intermediates:
            intermediates["after_entrance_detilt"] = field.detach()
            self._add_plane_diagnostics(intermediates, "after_entrance_detilt", field)

        for layer_idx, layer in enumerate(self.expert_layers):
            one_based_idx = layer_idx + 1
            field = layer(field, use_aperture_masks=self.use_aperture_masks)
            if return_intermediates:
                intermediates[f"after_layer_{one_based_idx}_modulation"] = field.detach()
                self._add_plane_diagnostics(intermediates, f"after_layer_{one_based_idx}_modulation", field)

            if layer_idx < self.num_layers - 1:
                field = self.propagators[2 + layer_idx](field)
                field = self._apply_aperture_union(field)
            else:
                field = self.propagators[-1](field)

            if return_intermediates:
                intermediates[f"after_layer_{one_based_idx}_propagation"] = field.detach()
                self._add_plane_diagnostics(intermediates, f"after_layer_{one_based_idx}_propagation", field)

        detector_field = field
        detector_intensity = torch.abs(detector_field.to(torch.complex64)) ** 2
        detector_outputs = self.detector(detector_field)
        logits = self._readout_logits(detector_outputs)

        if return_intermediates:
            intermediates["detector_field"] = detector_field.detach()
            intermediates["detector_intensity"] = detector_intensity.detach()
            intermediates["detector_energies_left_raw"] = detector_outputs["left_raw"].detach()
            intermediates["detector_energies_right_raw"] = detector_outputs["right_raw"].detach()
            intermediates["detector_energies_left_local_norm"] = detector_outputs["left_local_norm"].detach()
            intermediates["detector_energies_right_local_norm"] = detector_outputs["right_local_norm"].detach()
            intermediates["detector_energies_left_global_norm"] = detector_outputs["left_global_norm"].detach()
            intermediates["detector_energies_right_global_norm"] = detector_outputs["right_global_norm"].detach()
            intermediates["detector_energies_paired_sum"] = detector_outputs["paired_sum_global"].detach()
            intermediates["logits"] = logits.detach()
            total = detector_outputs["total_energy"] + 1e-8
            left_ratio = detector_outputs["left_aperture_energy"] / total
            right_ratio = detector_outputs["right_aperture_energy"] / total
            outside_ratio = detector_outputs["outside_energy"] / total
            intermediates["branch_energy_left"] = detector_outputs["left_aperture_energy"].detach()
            intermediates["branch_energy_right"] = detector_outputs["right_aperture_energy"].detach()
            intermediates["branch_energy_outside"] = detector_outputs["outside_energy"].detach()
            intermediates["branch_energy_ratios"] = torch.stack([left_ratio, right_ratio, outside_ratio], dim=1).detach()
            self._add_plane_diagnostics(intermediates, "detector_field", detector_field)
            return logits, intermediates

        return logits

    def load_single_expert_checkpoint_into_side(
        self,
        ckpt_path: str,
        side: str,
        old_config_path: Optional[str] = None,
        strict_geometry_check: bool = True,
        load_readout: bool = True,
    ) -> Dict:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")

        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        loaded_keys = []
        missing_keys = []
        warnings = []

        old_config = None
        if old_config_path:
            with open(old_config_path, "r", encoding="utf-8") as f:
                old_config = yaml.safe_load(f)
            warnings.extend(self._check_old_config_compatibility(old_config, strict_geometry_check))

        for layer_idx in range(self.num_layers):
            old_key = f"phase_layers.{layer_idx}.raw_phase"
            if old_key not in state_dict:
                missing_keys.append(old_key)
                continue
            tensor = state_dict[old_key].detach().cpu().float()
            expected_shape = (self.layout.expert_size, self.layout.expert_size)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{old_key} has shape {tuple(tensor.shape)}, expected {expected_shape}. "
                    "Checkpoint migration does not resize, rotate, or transpose phase masks."
                )
            target_layer = self.expert_layers[layer_idx].left_phase if side == "left" else self.expert_layers[layer_idx].right_phase
            target_layer.raw_phase.data.copy_(tensor.to(target_layer.raw_phase.device))
            loaded_keys.append(old_key)

        if load_readout and old_config is not None:
            readout_type = old_config.get("readout", {}).get("type", "optical_only")
            if readout_type != "optical_only":
                warnings.append(
                    f"Old readout_type={readout_type} is not migrated in this first OpticalMoE version; using optical detector readout."
                )

        unexpected_keys = [
            key
            for key in state_dict.keys()
            if key.startswith("phase_layers.") and key not in loaded_keys
        ]
        summary = {
            "source_checkpoint_path": str(ckpt_path),
            "source_config_path": str(old_config_path) if old_config_path else None,
            "side": side,
            "loaded_keys": loaded_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "warnings": warnings,
        }
        self.migration_summaries.append(summary)
        return summary

    def load_moe_checkpoint_side_into_side(
        self,
        ckpt_path: str,
        source_side: str,
        target_side: str,
    ) -> Dict:
        """Copy one side of a checkpoint produced by OpticalMoEClassifier.

        This is useful when left and right experts were trained in separate
        large-canvas runs and later need to be assembled into one paired model.
        """

        if source_side not in {"left", "right"} or target_side not in {"left", "right"}:
            raise ValueError("source_side and target_side must be 'left' or 'right'")

        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        source_name = f"{source_side}_phase"
        target_name = f"{target_side}_phase"
        loaded_keys = []
        missing_keys = []

        for layer_idx in range(self.num_layers):
            old_key = f"expert_layers.{layer_idx}.{source_name}.raw_phase"
            if old_key not in state_dict:
                missing_keys.append(old_key)
                continue
            tensor = state_dict[old_key].detach().cpu().float()
            expected_shape = (self.layout.expert_size, self.layout.expert_size)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{old_key} has shape {tuple(tensor.shape)}, expected {expected_shape}.")
            target_layer = self.expert_layers[layer_idx].left_phase if target_side == "left" else self.expert_layers[layer_idx].right_phase
            target_layer.raw_phase.data.copy_(tensor.to(target_layer.raw_phase.device))
            loaded_keys.append(old_key)

        summary = {
            "type": "optical_moe_side",
            "source_checkpoint_path": str(ckpt_path),
            "source_side": source_side,
            "target_side": target_side,
            "loaded_keys": loaded_keys,
            "missing_keys": missing_keys,
            "epoch": payload.get("epoch"),
        }
        self.migration_summaries.append(summary)
        return summary

    def _check_old_config_compatibility(self, old_config: Dict, strict: bool) -> List[str]:
        warnings = []
        optics = old_config.get("optics", {})
        detector = old_config.get("detector", {})
        readout = old_config.get("readout", {})
        checks = [
            ("optics.wavelength_nm", optics.get("wavelength_nm"), self.wavelength_m * 1e9),
            ("optics.pixel_size_um", optics.get("pixel_size_um"), self.pixel_size_m * 1e6),
            ("optics.input_size", optics.get("input_size"), self.layout.input_size),
            ("optics.grid_size", optics.get("grid_size"), self.layout.expert_size),
            ("optics.num_layers", optics.get("num_layers"), self.num_layers),
            ("optics.phase_param", optics.get("phase_param"), self.phase_param),
            ("detector.detector_size", detector.get("detector_size"), self.detector_size),
            ("detector.layout", detector.get("layout"), self.detector_layout),
            ("readout.type", readout.get("type", "optical_only"), "optical_only"),
        ]
        for name, old_value, current_value in checks:
            if old_value is None:
                warnings.append(f"{name} missing in old config; compatibility could not be checked.")
                continue
            same = old_value == current_value
            if isinstance(current_value, float):
                same = abs(float(old_value) - float(current_value)) < 1e-6
            if not same:
                message = f"Config mismatch for {name}: old={old_value}, current={current_value}"
                if strict:
                    raise ValueError(message)
                warnings.append(message)
        return warnings

    def geometry_summary(self) -> Dict:
        return {
            "layout": self.layout.to_dict(),
            "wavelength_nm": self.wavelength_m * 1e9,
            "pixel_size_um": self.pixel_size_m * 1e6,
            "distances_m": self.distances_m,
            "num_layers": self.num_layers,
            "prompt_slope_sign": self.prompt_slope_sign,
            "steering": self.steering_params.to_dict(),
            "readout_mode": self.readout_mode,
            "prompt_mode": self.prompt_mode,
            "target_side": self.target_side,
            "use_entrance_detilt": self.use_entrance_detilt,
            "use_aperture_masks": self.use_aperture_masks,
            "input_placement": self.layout.input_aperture.to_dict(),
        }
