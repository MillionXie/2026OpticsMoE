import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OPTICS_PATH = Path(__file__).resolve().parent.parent / "d2nn_mnist4_amp_1phase400" / "optics.py"
_spec = importlib.util.spec_from_file_location("fulloptical_reliable_optics", OPTICS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load angular-spectrum optics from {OPTICS_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
AngularSpectrumPropagator = _module.AngularSpectrumPropagator
DetectorArray = _module.DetectorArray
PhaseLayer = _module.PhaseLayer


class SquareDetectionLayerNormReload(nn.Module):
    """Square-law detection -> spatial LayerNorm -> nonlinearity."""

    def __init__(self, eps=1e-5, nonlinearity="relu", field_size=None, elementwise_affine=False):
        super().__init__(); self.eps=float(eps); self.nonlinearity=str(nonlinearity).lower()
        if self.eps<=0: raise ValueError("optoelectronic_interlayers.eps must be positive")
        if self.nonlinearity not in {"relu","softplus"}: raise ValueError("optoelectronic_interlayers.nonlinearity must be relu or softplus")
        self.elementwise_affine=bool(elementwise_affine);self.field_size=int(field_size) if field_size is not None else None
        if self.elementwise_affine:
            if self.field_size is None or self.field_size<=0:raise ValueError("field_size is required when elementwise_affine=true")
            self.affine_weight=nn.Parameter(torch.ones(self.field_size,self.field_size,dtype=torch.float32))
            self.affine_bias=nn.Parameter(torch.zeros(self.field_size,self.field_size,dtype=torch.float32))
        else:self.register_parameter("affine_weight",None);self.register_parameter("affine_bias",None)

    def forward(self, field, return_details=False):
        intensity=field.to(torch.complex64).abs().square().float()
        normalized=F.layer_norm(intensity,intensity.shape[-2:],weight=None,bias=None,eps=self.eps)
        if self.affine_weight is not None:normalized=normalized*self.affine_weight+self.affine_bias
        amplitude=F.relu(normalized) if self.nonlinearity=="relu" else F.softplus(normalized)
        reloaded=torch.complex(amplitude,torch.zeros_like(amplitude))
        if not return_details:return reloaded
        return reloaded,{
            "detector_intensity":intensity,"layer_normalized":normalized,"reloaded_amplitude":amplitude,
            "normalization_mean":intensity.mean((-2,-1)),
            "normalization_std":torch.sqrt(intensity.var((-2,-1),unbiased=False)+self.eps),
            "normalization_scope":"spatial_per_sample","elementwise_affine":self.elementwise_affine,
            "affine_sharing":"per_stage","routing_amplitude_reapplied":False,
        }


class FullOpticalD2NNClassifier(nn.Module):
    """Six phase-only planes followed directly by detector-region readout."""

    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        optics = config.get("optics", {})
        detector = config.get("detector", {})
        dropout = config.get("regularization", {}).get("phase_dropout", {})
        self.input_size = int(optics.get("input_size", 300))
        self.canvas_size = int(optics.get("canvas_size", 360))
        self.num_layers = int(optics.get("num_layers", 6))
        conversion=config.get("optoelectronic_interlayers",{})
        self.optoelectronic_enabled=bool(conversion.get("enabled",False))
        if self.optoelectronic_enabled:
            expected={"detection":"square_law","normalization":"layer_norm","normalization_scope":"spatial_per_sample","reload_as":"amplitude"}
            for key,value in expected.items():
                if str(conversion.get(key,value)).lower()!=value:raise ValueError(f"optoelectronic_interlayers.{key} must be {value!r}")
            if bool(conversion.get("reapply_routing_amplitude",False)):raise ValueError("Full-optical model has no routing amplitude to reapply")
        if self.input_size > self.canvas_size or (self.canvas_size - self.input_size) % 2:
            raise ValueError("optics.canvas_size-input_size must be nonnegative and even")
        if self.num_layers != 6:
            raise ValueError("This baseline is defined as exactly six phase planes")
        self.interlayer_conversions=nn.ModuleList([
            SquareDetectionLayerNormReload(
                float(conversion.get("eps",1e-5)),str(conversion.get("nonlinearity","relu")),
                field_size=self.canvas_size,
                elementwise_affine=bool(conversion.get("elementwise_affine",conversion.get("affine",True))),
            ) for _ in range(self.num_layers-1)
        ]) if self.optoelectronic_enabled else nn.ModuleList()
        enabled = bool(dropout.get("enabled", False))
        mode = str(dropout.get("mode", "none")) if enabled else "none"
        probability = float(dropout.get("p", 0.0)) if enabled else 0.0
        self.phase_layers = nn.ModuleList([
            PhaseLayer(
                self.canvas_size,
                parameterization=str(optics.get("phase_param", "sigmoid")),
                init=str(optics.get("phase_init", "zeros")),
                init_std=float(optics.get("phase_init_std", 0.02)),
                phase_dropout_mode=mode,
                phase_dropout_p=probability,
                phase_dropout_block_size=int(dropout.get("block_size", 8)),
                phase_dropout_batch_shared=bool(dropout.get("batch_shared", True)),
            )
            for _ in range(self.num_layers)
        ])
        propagation = dict(
            wavelength_m=float(optics.get("wavelength_m", 5.32e-7)),
            pixel_size_m=float(optics.get("pixel_size_m", 16e-6)),
            grid_size=self.canvas_size,
            evanescent_mode=str(optics.get("evanescent_mode", "zero")),
            k_space_constraint_enabled=bool(optics.get("k_space_constraint_enabled", False)),
            theta_max_deg=float(optics.get("theta_max_deg", 1.0)),
        )
        self.input_prop = AngularSpectrumPropagator(
            distance_m=float(optics.get("input_to_layer_distance_m", 0.0)), **propagation
        )
        self.inter_props = nn.ModuleList([
            AngularSpectrumPropagator(
                distance_m=float(optics.get("inter_layer_distance_m", 0.05)), **propagation
            )
            for _ in range(self.num_layers - 1)
        ])
        self.detector_prop = AngularSpectrumPropagator(
            distance_m=float(optics.get("detector_distance_m", 0.10)), **propagation
        )
        self.detector = DetectorArray(
            self.num_classes,
            self.canvas_size,
            detector_size=int(detector.get("detector_size", 40)),
            layout=str(detector.get("layout", "fixed_2x2")),
            normalize_total_energy=bool(detector.get("normalize_detector_energy", True)),
            start_pos_x=int(detector.get("start_pos_x", 100)),
            start_pos_y=int(detector.get("start_pos_y", 100)),
            n_det_sets=detector.get("N_det_sets", [2, 2]),
            det_steps_x=detector.get("det_steps_x", [80, 80]),
            det_steps_y=int(detector.get("det_steps_y", 80)),
            start_pos_x_per_row=detector.get("start_pos_x_per_row"),
        )

    def prepare_input(self, images):
        if images.ndim == 3: images = images.unsqueeze(1)
        if images.shape[1] != 1: images = images.mean(1, keepdim=True)
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            dataset = self.config.get("dataset", {})
            mode = str(dataset.get("interpolation", "bicubic")).lower()
            kwargs = {"mode": mode}
            if mode in {"bilinear", "bicubic"}:
                kwargs.update({"align_corners": False, "antialias": bool(dataset.get("antialias", True))})
            images = F.interpolate(images.float(), size=(self.input_size, self.input_size), **kwargs)
        images = images.float().clamp(0.0, 1.0)
        pad = (self.canvas_size - self.input_size) // 2
        return F.pad(images[:, 0], (pad, pad, pad, pad), value=0.0).to(torch.complex64)

    def forward(self, images, return_intermediates=False, capture_layer_fields=True):
        canvas = self.prepare_input(images)
        field = self.input_prop(canvas)
        fields = []; interlayer_detector_intensities=[];interlayer_normalized=[];interlayer_reloaded_amplitudes=[]
        for index, layer in enumerate(self.phase_layers):
            field = layer(field)
            if index < len(self.inter_props):
                field = self.inter_props[index](field)
                if self.optoelectronic_enabled:
                    conversion_layer=self.interlayer_conversions[index]
                    if return_intermediates and capture_layer_fields:
                        field,conversion=conversion_layer(field,return_details=True)
                        interlayer_detector_intensities.append(conversion["detector_intensity"])
                        interlayer_normalized.append(conversion["layer_normalized"])
                        interlayer_reloaded_amplitudes.append(conversion["reloaded_amplitude"])
                    else:field=conversion_layer(field)
            if return_intermediates and capture_layer_fields:
                fields.append(field)
        detector_field = self.detector_prop(field)
        logits = self.detector(detector_field)
        if not return_intermediates:
            return logits
        return logits, {
            "input_canvas": canvas,
            "after_each_layer": fields,
            "detector_field": detector_field,
            "detector_intensity": detector_field.abs().square(),
            "detector_energies": logits,
            "optoelectronic_interlayers_enabled":self.optoelectronic_enabled,
            "interlayer_detector_intensities":interlayer_detector_intensities,
            "interlayer_layer_normalized":interlayer_normalized,
            "interlayer_reloaded_amplitudes":interlayer_reloaded_amplitudes,
        }

    def phase_stack_wrapped(self):
        return torch.stack([layer.get_phase_wrapped() for layer in self.phase_layers])

    def set_phase_dropout_active(self, active):
        for layer in self.phase_layers: layer.set_phase_dropout_active(active)

    def optical_parameter_count(self):
        return sum(layer.raw_phase.numel() for layer in self.phase_layers)

    def interlayer_conversion_parameter_count(self):
        return sum(parameter.numel() for parameter in self.interlayer_conversions.parameters() if parameter.requires_grad)

    def electronic_parameter_count(self):
        return self.interlayer_conversion_parameter_count()
