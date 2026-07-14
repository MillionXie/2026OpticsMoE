"""CIFAR-10 deep heterogeneous nine-expert linear optical MoE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experts import HeterogeneousExpertBank
from layout import MoELayout
from optics import AngularSpectrumPropagator, DetectorArray, PhaseLayer
from prompt import GlobalRouterPrompt


class GlobalFCPhaseLayer(nn.Module):
    """One trainable 450x450 phase-only plane."""

    def __init__(self, layout, optics_cfg, dropout_cfg):
        super().__init__()
        self.layout = layout
        enabled = bool(dropout_cfg.get("enabled", False))
        self.phase = PhaseLayer(
            layout.active_size,
            parameterization=optics_cfg.get("phase_param", "sigmoid"),
            init=optics_cfg.get("phase_init", "zeros"),
            init_std=float(optics_cfg.get("init_std", 0.02)),
            phase_dropout_mode=dropout_cfg.get("mode", "none") if enabled else "none",
            phase_dropout_p=float(dropout_cfg.get("p", 0.0)) if enabled else 0.0,
            phase_dropout_block_size=int(dropout_cfg.get("block_size", 8)),
            phase_dropout_batch_shared=bool(dropout_cfg.get("batch_shared", True)),
        )

    def forward(self, field):
        aperture = self.layout.active_aperture
        output = field.to(torch.complex64).clone()
        crop = field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
        output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = self.phase(crop)
        return output

    def get_phase(self):
        return self.phase.get_phase_wrapped()

    def set_phase_dropout_active(self, active):
        self.phase.set_phase_dropout_active(active)


class DeepHeterogeneousOpticalMoEClassifier(nn.Module):
    def __init__(self, config, num_classes=4):
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        model_cfg = config.get("model", {})
        optics = config.get("optics", {})
        detector = config.get("detector", {})
        self.layout = MoELayout(
            canvas_size=int(model_cfg.get("canvas_size", 480)),
            active_size=int(model_cfg.get("active_size", 450)),
            input_size=int(model_cfg.get("input_size", 120)),
            image_size=int(model_cfg.get("image_size", 100)),
            num_experts=int(model_cfg.get("num_experts", 9)),
            expert_size=int(model_cfg.get("expert_size", 120)),
            expert_pitch=int(model_cfg.get("expert_pitch", 150)),
        )
        self.layout.validate()
        wavelength = float(optics.get("wavelength_m", 5.32e-7))
        pixel_size = float(optics.get("pixel_size_m", 16.0e-6))
        distances = optics.get("distances_m", {})
        common = {
            "wavelength_m": wavelength,
            "pixel_size_m": pixel_size,
            "grid_size": self.layout.canvas_size,
            "evanescent_mode": str(optics.get("evanescent_mode", "zero")),
            "k_space_constraint_enabled": bool(optics.get("k_space_constraint_enabled", False)),
            "theta_max_deg": float(optics.get("theta_max_deg", 1.0)),
        }
        input_to_prompt_distance = float(distances.get("input_to_prompt", 0.1444))
        prompt_to_expert_distance = float(distances.get("prompt_to_expert", 0.1444))
        focal_length = float(optics.get("prompt_focal_length_m", prompt_to_expert_distance / 2.0))
        prompt_cfg = config.get("prompt", {})
        if bool(prompt_cfg.get("enforce_global_convolution_geometry", True)):
            expected = 2.0 * focal_length
            tolerance = float(prompt_cfg.get("convolution_relative_tolerance", 0.02))
            if abs(prompt_to_expert_distance - expected) / expected > tolerance:
                raise ValueError(
                    f"prompt_to_expert={prompt_to_expert_distance:.6f} m is incompatible with global fan-out; "
                    f"expected 2*f={expected:.6f} m"
                )
        self.input_to_prompt = AngularSpectrumPropagator(
            distance_m=input_to_prompt_distance,
            **common,
        )
        self.prompt = GlobalRouterPrompt(
            self.layout,
            wavelength,
            pixel_size,
            input_to_prompt_distance,
            prompt_to_expert_distance,
            focal_length,
            top_k=int(prompt_cfg.get("top_k", 3)),
            pool_size=int(prompt_cfg.get("router_pool_size", 10)),
            temperature=float(prompt_cfg.get("temperature", 1.0)),
            grating_sign_x=float(prompt_cfg.get("grating_sign_x", 1.0)),
            grating_sign_y=float(prompt_cfg.get("grating_sign_y", 1.0)),
            min_grating_period_pixels=float(prompt_cfg.get("min_grating_period_pixels", 0.0)),
            mode=str(prompt_cfg.get("mode", "region_amplitude_global_lens")),
        )
        self.prompt_to_expert = AngularSpectrumPropagator(distance_m=prompt_to_expert_distance, **common)
        self.expert_bank = HeterogeneousExpertBank(self.layout, config.get("expert_bank", {}), optics)
        self.expert_to_global_fc = AngularSpectrumPropagator(
            distance_m=float(distances.get("expert_to_global_fc", distances.get("last_expert_to_global_fc", 0.05))),
            **common,
        )
        dropout = config.get("regularization", {}).get("phase_dropout", {})
        self.global_fc = GlobalFCPhaseLayer(self.layout, optics, dropout)
        self.to_detector = AngularSpectrumPropagator(
            distance_m=float(distances.get("global_fc_to_detector", 0.10)),
            **common,
        )
        self.detector = DetectorArray(
            self.num_classes,
            self.layout.canvas_size,
            int(detector.get("detector_size", 50)),
            detector.get("layout", "fixed_2x2"),
            bool(detector.get("normalize_detector_energy", True)),
            start_pos_x=int(detector.get("start_pos_x", 115)),
            start_pos_y=int(detector.get("start_pos_y", 115)),
            n_det_sets=detector.get("N_det_sets", [2, 2]),
            det_steps_x=detector.get("det_steps_x", [150, 150]),
            det_steps_y=int(detector.get("det_steps_y", 150)),
            start_pos_x_per_row=detector.get("start_pos_x_per_row"),
        )

    def prepare_canvas_input(self, images):
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.shape[1] != 1:
            images = images.mean(1, keepdim=True)
        if tuple(images.shape[-2:]) != (self.layout.input_size, self.layout.input_size):
            dataset_cfg = self.config.get("dataset", {})
            mode = str(dataset_cfg.get("resize_interpolation", "bicubic")).lower()
            if mode not in {"nearest", "bilinear", "bicubic"}:
                raise ValueError("dataset.resize_interpolation must be nearest, bilinear, or bicubic")
            kwargs = {"mode": mode}
            if mode in {"bilinear", "bicubic"}:
                kwargs.update({"align_corners": False, "antialias": bool(dataset_cfg.get("resize_antialias", True))})
            images = F.interpolate(images.float(), size=(self.layout.image_size, self.layout.image_size), **kwargs).clamp(0, 1)
            total_pad = self.layout.input_size - self.layout.image_size
            left = total_pad // 2
            right = total_pad - left
            images = F.pad(images, (left, right, left, right), mode="constant", value=0.0)
        aperture = self.layout.input_aperture
        canvas = torch.zeros(images.shape[0], self.layout.canvas_size, self.layout.canvas_size, device=images.device)
        canvas[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = images[:, 0].clamp(0, 1)
        return canvas.to(torch.complex64)

    def expert_energy_ratios(self, field):
        intensity = field.to(torch.complex64).abs().square()
        energies = []
        for aperture in self.layout.expert_apertures:
            energies.append(intensity[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1].sum((-2, -1)))
        energies = torch.stack(energies, dim=1)
        return energies / intensity.sum((-2, -1))[:, None].clamp_min(1.0e-12)

    @staticmethod
    def global_fanout_convolution(field, prompt_transmission):
        field = field.to(torch.complex64)
        prompt_transmission = prompt_transmission.to(torch.complex64)
        flipped = torch.flip(field, dims=(-2, -1))
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.fft2(flipped) * torch.fft.fft2(prompt_transmission)),
            dim=(-2, -1),
        ).to(torch.complex64)

    def forward(self, images, return_intermediates=False, capture_expert_outputs=True):
        canvas = self.prepare_canvas_input(images)
        routing = self.prompt.routing(images)
        prompt_transmission = routing["transmission"]
        expert_entrance = self.global_fanout_convolution(canvas, prompt_transmission)
        entrance_energy_ratios = self.expert_energy_ratios(expert_entrance)
        if return_intermediates:
            bank_output, bank_details = self.expert_bank(
                expert_entrance,
                return_details=True,
                capture_outputs=capture_expert_outputs,
            )
        else:
            bank_output = self.expert_bank(expert_entrance)
            bank_details = None
        at_global_fc = self.expert_to_global_fc(bank_output)
        after_global_fc = self.global_fc(at_global_fc)
        detector_field = self.to_detector(after_global_fc)
        detector_intensity = detector_field.abs().square()
        logits = self.detector(detector_field)
        if not return_intermediates:
            return logits
        return logits, {
            "input_canvas": canvas,
            "at_prompt": canvas,
            "prompt_amplitude": routing["prompt_amplitude"],
            "prompt_phase": routing["prompt_phase"],
            "prompt_transmission": prompt_transmission,
            "routing_logits": routing["logits"],
            "routing_probabilities": routing["probabilities"],
            "routing_weights": routing["weights"],
            "routing_selected_mask": routing["selected_mask"],
            "routing_selected_indices": routing["selected_indices"],
            "router_balance_loss": routing["balance_loss"],
            "router_importance_loss": routing["importance_loss"],
            "router_normalized_entropy": routing["normalized_entropy"],
            "router_importance": routing["importance"],
            "router_load": routing["load"],
            "expert_entrance": expert_entrance,
            "expert_entrance_energy_ratios": entrance_energy_ratios,
            "expert_bank_output": bank_output,
            "expert_local_outputs": bank_details["local_outputs"],
            "expert_input_power": bank_details["input_power"],
            "expert_output_power": bank_details["output_power"],
            "fiber_coupling_efficiency": bank_details["fiber_coupling_efficiency"],
            "fiber_effective_mode_number": bank_details["fiber_effective_mode_number"],
            "fiber_mode_power_distribution": bank_details["fiber_mode_power_distribution"],
            "expert_details": bank_details["expert_details"],
            "expert_types": bank_details["expert_types"],
            "at_global_fc": at_global_fc,
            "after_global_fc": after_global_fc,
            "global_fc_phase": self.global_fc.get_phase(),
            "detector_field": detector_field,
            "detector_intensity": detector_intensity,
            "detector_energies": logits,
        }

    def set_phase_dropout_active(self, active):
        self.global_fc.set_phase_dropout_active(active)

    def expert_parameter_report(self):
        return self.expert_bank.parameter_report()

    def expert_parameter_count(self):
        return self.expert_bank.trainable_parameter_count()

    def global_fc_parameter_count(self):
        return self.global_fc.phase.raw_phase.numel()

    def router_parameter_count(self):
        return sum(parameter.numel() for parameter in self.prompt.router_network.parameters())

    def optical_parameter_count(self):
        return self.expert_parameter_count() + self.global_fc_parameter_count()

    def electronic_parameter_count(self):
        return self.router_parameter_count()
