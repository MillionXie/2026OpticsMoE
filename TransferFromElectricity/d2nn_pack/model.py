import torch
import torch.nn as nn
import torch.nn.functional as F

from optics import AngularSpectrumPropagator, DetectorArray, PhaseLayer


class CenteredLocalPhaseLayer(nn.Module):
    """Trainable local phase mask embedded transparently in a larger canvas."""

    def __init__(
        self,
        canvas_size: int,
        phase_mask_size: int,
        parameterization: str = "unconstrained",
        init: str = "uniform_0_2pi",
        init_std: float = 0.02,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ):
        super().__init__()
        self.canvas_size = int(canvas_size)
        self.phase_mask_size = int(phase_mask_size)
        if self.phase_mask_size > self.canvas_size:
            raise ValueError("phase_mask_size must be <= canvas_size.")
        self.y0 = (self.canvas_size - self.phase_mask_size) // 2
        self.x0 = (self.canvas_size - self.phase_mask_size) // 2
        self.y1 = self.y0 + self.phase_mask_size
        self.x1 = self.x0 + self.phase_mask_size
        self.local_phase = PhaseLayer(
            self.phase_mask_size,
            parameterization=parameterization,
            init=init,
            init_std=init_std,
            phase_dropout_mode=phase_dropout_mode,
            phase_dropout_p=phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )

    @property
    def raw_phase(self):
        return self.local_phase.raw_phase

    def phase_mask_region(self):
        return [self.y0, self.y1, self.x0, self.x1]

    def set_phase_dropout_active(self, active: bool) -> None:
        self.local_phase.set_phase_dropout_active(active)

    def get_phase_wrapped(self) -> torch.Tensor:
        return self.local_phase.get_phase_wrapped()

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        if tuple(field.shape[-2:]) != (self.canvas_size, self.canvas_size):
            raise ValueError(f"Expected canvas {(self.canvas_size, self.canvas_size)}, got {tuple(field.shape[-2:])}")
        field = field.to(torch.complex64)
        crop = field[:, self.y0 : self.y1, self.x0 : self.x1]
        modulated_crop = self.local_phase(crop)
        delta = modulated_crop - crop
        return field + F.pad(delta, (self.x0, self.canvas_size - self.x1, self.y0, self.canvas_size - self.y1))


class ElectronicReadout(nn.Module):
    def __init__(self, num_classes: int, cfg: dict):
        super().__init__()
        self.readout_type = cfg.get("type", "mlp")
        self.logit_scale = float(cfg.get("logit_scale", 10.0))
        input_norm = cfg.get("input_norm", "layernorm")
        norm_affine = bool(cfg.get("norm_affine", True))
        if input_norm in {None, "none"}:
            norm = nn.Identity()
        elif input_norm == "layernorm":
            norm = nn.LayerNorm(num_classes, elementwise_affine=norm_affine)
        else:
            raise ValueError(f"Unsupported input_norm: {input_norm}")
        if self.readout_type == "optical_only":
            self.net = norm
        elif self.readout_type == "linear":
            self.net = nn.Sequential(norm, nn.Linear(num_classes, num_classes))
        elif self.readout_type == "mlp":
            hidden_dim = int(cfg.get("hidden_dim", 64))
            hidden_layers = int(cfg.get("hidden_layers", 1))
            dropout = float(cfg.get("dropout", 0.0))
            layers = [norm, nn.Linear(num_classes, hidden_dim)]
            for index in range(hidden_layers):
                layers.append(self._activation(cfg.get("activation", "gelu")))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                if index < hidden_layers - 1:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported readout.type: {self.readout_type}")

    @staticmethod
    def _activation(name: str):
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        logits = self.net(energies)
        return logits * self.logit_scale if self.readout_type == "optical_only" else logits


class D2NNClassifier(nn.Module):
    def __init__(self, config: dict, num_classes: int = 10):
        super().__init__()
        optics_cfg = config.get("optics", {})
        det_cfg = config.get("detector", {})
        readout_cfg = config.get("readout", {})
        dropout_cfg = config.get("regularization", {}).get("phase_dropout", {})
        enabled = bool(dropout_cfg.get("enabled", False))
        dropout_mode = dropout_cfg.get("mode", "none") if enabled else "none"
        dropout_p = float(dropout_cfg.get("p", 0.0)) if enabled else 0.0

        self.num_classes = int(num_classes)
        self.canvas_size = int(optics_cfg.get("canvas_size", 400))
        self.input_size = int(optics_cfg.get("input_size", 256))
        self.phase_mask_size = int(optics_cfg.get("phase_mask_size", self.input_size))
        self.phase_mask_mode = optics_cfg.get("phase_mask_mode", "centered_local")
        self.num_layers = int(optics_cfg.get("num_layers", 5))
        if self.phase_mask_size > self.canvas_size:
            raise ValueError("phase_mask_size must be <= canvas_size.")
        self.phase_layers = nn.ModuleList(
            [self._make_phase_layer(optics_cfg, dropout_cfg, dropout_mode, dropout_p) for _ in range(self.num_layers)]
        )
        self.input_to_layer_prop = AngularSpectrumPropagator(
            wavelength_m=float(optics_cfg.get("wavelength_m", 5.32e-7)),
            pixel_size_m=float(optics_cfg.get("pixel_size_m", 8.0e-6)),
            grid_size=self.canvas_size,
            distance_m=float(optics_cfg.get("input_to_layer_distance_m", 0.05)),
            evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
        )
        self.inter_prop = AngularSpectrumPropagator(
            wavelength_m=float(optics_cfg.get("wavelength_m", 5.32e-7)),
            pixel_size_m=float(optics_cfg.get("pixel_size_m", 8.0e-6)),
            grid_size=self.canvas_size,
            distance_m=float(optics_cfg.get("inter_layer_distance_m", 0.05)),
            evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
        )
        self.detector_prop = AngularSpectrumPropagator(
            wavelength_m=float(optics_cfg.get("wavelength_m", 5.32e-7)),
            pixel_size_m=float(optics_cfg.get("pixel_size_m", 8.0e-6)),
            grid_size=self.canvas_size,
            distance_m=float(optics_cfg.get("detector_distance_m", 0.05)),
            evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
        )
        self.detector = DetectorArray(
            num_classes=self.num_classes,
            grid_size=self.canvas_size,
            detector_size=int(det_cfg.get("detector_size", 32)),
            layout=det_cfg.get("layout", "grid"),
            normalize_total_energy=bool(det_cfg.get("normalize_detector_energy", True)),
        )
        self.readout = ElectronicReadout(self.num_classes, readout_cfg)

    def _make_phase_layer(self, optics_cfg, dropout_cfg, dropout_mode, dropout_p):
        kwargs = {
            "parameterization": optics_cfg.get("phase_param", "unconstrained"),
            "init": optics_cfg.get("phase_init", "uniform_0_2pi"),
            "init_std": float(optics_cfg.get("init_std", 0.02)),
            "phase_dropout_mode": dropout_mode,
            "phase_dropout_p": dropout_p,
            "phase_dropout_block_size": int(dropout_cfg.get("block_size", 8)),
            "phase_dropout_batch_shared": bool(dropout_cfg.get("batch_shared", True)),
        }
        if self.phase_mask_mode == "centered_local":
            return CenteredLocalPhaseLayer(self.canvas_size, self.phase_mask_size, **kwargs)
        if self.phase_mask_mode == "full_canvas":
            return PhaseLayer(self.canvas_size, **kwargs)
        raise ValueError(f"Unsupported phase_mask_mode: {self.phase_mask_mode}")

    def phase_mask_region(self):
        if self.phase_mask_mode == "centered_local":
            return self.phase_layers[0].phase_mask_region()
        return [0, self.canvas_size, 0, self.canvas_size]

    def prepare_canvas_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        images = images.float()
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        canvas = torch.zeros(images.shape[0], self.canvas_size, self.canvas_size, device=images.device, dtype=torch.float32)
        y0 = (self.canvas_size - self.input_size) // 2
        x0 = (self.canvas_size - self.input_size) // 2
        canvas[:, y0 : y0 + self.input_size, x0 : x0 + self.input_size] = images[:, 0]
        return canvas.to(torch.complex64)

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.phase_layers:
            layer.set_phase_dropout_active(active)

    def phase_stack_wrapped(self) -> torch.Tensor:
        return torch.stack([layer.get_phase_wrapped().detach().cpu() for layer in self.phase_layers], dim=0)

    def optical_parameter_count(self) -> int:
        return sum(layer.raw_phase.numel() for layer in self.phase_layers)

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.readout.parameters())

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        intermediates = {}
        if return_intermediates:
            intermediates["input_256"] = images.detach()
        field = self.prepare_canvas_input(images)
        if return_intermediates:
            intermediates["canvas_input_400"] = field.detach()
        field = self.input_to_layer_prop(field)
        if return_intermediates:
            intermediates["after_input_to_layer"] = field.detach()
        for index, layer in enumerate(self.phase_layers):
            field = layer(field)
            if return_intermediates:
                intermediates[f"after_phase_modulation_{index + 1}"] = field.detach()
            if index < self.num_layers - 1:
                field = self.inter_prop(field)
                if return_intermediates:
                    intermediates[f"after_propagation_{index + 1}"] = field.detach()
        detector_field = self.detector_prop(field)
        detector_energies = self.detector(detector_field)
        logits = self.readout(detector_energies)
        if not return_intermediates:
            return logits
        intermediates["detector_field"] = detector_field.detach()
        intermediates["detector_intensity"] = torch.abs(detector_field.detach()).square()
        intermediates["detector_energies"] = detector_energies.detach()
        intermediates["logits"] = logits.detach()
        return logits, intermediates
