import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OPTICS_PATH = Path(__file__).resolve().parent.parent / "d2nn_mnist4_amp_1phase400" / "optics.py"
_spec = importlib.util.spec_from_file_location("kadid_regression_reliable_optics", OPTICS_PATH)
if _spec is None or _spec.loader is None: raise ImportError(f"Cannot load optics from {OPTICS_PATH}")
_module = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_module)
AngularSpectrumPropagator = _module.AngularSpectrumPropagator
DetectorArray = _module.DetectorArray
PhaseLayer = _module.PhaseLayer


class FullOpticalIQARegressor(nn.Module):
    """No-reference IQA: six phase planes and fixed detector-anchor readout."""

    def __init__(self, config):
        super().__init__(); self.config = config
        optics = config.get("optics", {}); detector = config.get("detector", {}); dropout = config.get("regularization", {}).get("phase_dropout", {})
        self.input_size = int(optics.get("input_size", 300)); self.canvas_size = int(optics.get("canvas_size", 360)); self.num_layers = int(optics.get("num_layers", 6))
        if self.num_layers != 6: raise ValueError("KADID pure-optical baseline requires exactly six phase planes")
        if self.input_size > self.canvas_size or (self.canvas_size - self.input_size) % 2: raise ValueError("canvas_size-input_size must be nonnegative and even")
        enabled = bool(dropout.get("enabled", False)); mode = dropout.get("mode", "none") if enabled else "none"; probability = float(dropout.get("p", 0.0)) if enabled else 0.0
        self.phase_layers = nn.ModuleList([
            PhaseLayer(self.canvas_size, parameterization=optics.get("phase_param", "sigmoid"), init=optics.get("phase_init", "zeros"), init_std=float(optics.get("phase_init_std", .02)), phase_dropout_mode=mode, phase_dropout_p=probability, phase_dropout_block_size=int(dropout.get("block_size", 8)), phase_dropout_batch_shared=bool(dropout.get("batch_shared", True)))
            for _ in range(self.num_layers)
        ])
        propagation = dict(wavelength_m=float(optics.get("wavelength_m", 5.32e-7)), pixel_size_m=float(optics.get("pixel_size_m", 16e-6)), grid_size=self.canvas_size, evanescent_mode=optics.get("evanescent_mode", "zero"), k_space_constraint_enabled=bool(optics.get("k_space_constraint_enabled", False)), theta_max_deg=float(optics.get("theta_max_deg", 1.0)))
        self.input_prop = AngularSpectrumPropagator(distance_m=float(optics.get("input_to_layer_distance_m", 0.0)), **propagation)
        self.inter_props = nn.ModuleList([AngularSpectrumPropagator(distance_m=float(optics.get("inter_layer_distance_m", .05)), **propagation) for _ in range(self.num_layers - 1)])
        self.detector_prop = AngularSpectrumPropagator(distance_m=float(optics.get("detector_distance_m", .10)), **propagation)
        anchors = int(detector.get("quality_anchor_count", 10))
        self.detector = DetectorArray(anchors, self.canvas_size, detector_size=int(detector.get("detector_size", 30)), layout=detector.get("layout", "fixed_2x2"), normalize_total_energy=bool(detector.get("normalize_detector_energy", True)), start_pos_x=int(detector.get("start_pos_x", 90)), start_pos_y=int(detector.get("start_pos_y", 105)), n_det_sets=detector.get("N_det_sets", [3, 3, 4]), det_steps_x=detector.get("det_steps_x", [20, 20, 20]), det_steps_y=int(detector.get("det_steps_y", 30)), start_pos_x_per_row=detector.get("start_pos_x_per_row", [115, 115, 90]))
        self.register_buffer("quality_anchors", torch.linspace(0.0, 1.0, anchors), persistent=True)

    def prepare_input(self, images):
        if images.ndim == 3: images = images.unsqueeze(1)
        if images.shape[1] != 1: images = images.mean(1, keepdim=True)
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size): images = F.interpolate(images.float(), size=(self.input_size, self.input_size), mode="bicubic", align_corners=False, antialias=True)
        images = images.float().clamp(0, 1); pad = (self.canvas_size - self.input_size) // 2
        return F.pad(images[:, 0], (pad, pad, pad, pad), value=0.0).to(torch.complex64)

    def forward(self, images, return_intermediates=False, capture_layer_fields=True):
        canvas = self.prepare_input(images); field = self.input_prop(canvas); fields = []
        for index, layer in enumerate(self.phase_layers):
            field = layer(field)
            if index < len(self.inter_props): field = self.inter_props[index](field)
            if return_intermediates and capture_layer_fields: fields.append(field)
        detector_field = self.detector_prop(field); region_energies = self.detector(detector_field)
        region_probabilities = region_energies / (region_energies.sum(1, keepdim=True) + 1e-8)
        prediction = torch.sum(region_probabilities * self.quality_anchors.to(region_probabilities), dim=1)
        if not return_intermediates: return prediction
        return prediction, {"input_canvas": canvas, "after_each_layer": fields, "detector_field": detector_field, "detector_intensity": detector_field.abs().square(), "region_energies": region_energies, "region_probabilities": region_probabilities, "prediction": prediction}

    def phase_stack_wrapped(self): return torch.stack([layer.get_phase_wrapped() for layer in self.phase_layers])
    def set_phase_dropout_active(self, active):
        for layer in self.phase_layers: layer.set_phase_dropout_active(active)
    def optical_parameter_count(self): return sum(layer.raw_phase.numel() for layer in self.phase_layers)
    @staticmethod
    def electronic_parameter_count(): return 0


def soft_quality_targets(targets, anchors, sigma):
    sigma = float(sigma)
    if sigma <= 0: raise ValueError("target_distribution_sigma must be positive")
    values = torch.exp(-0.5 * ((targets[:, None] - anchors[None, :]) / sigma).square())
    return values / (values.sum(1, keepdim=True) + 1e-8)
