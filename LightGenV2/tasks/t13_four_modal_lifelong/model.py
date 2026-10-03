"""Fixed-geometry optical MoE and full-aperture D2NN.

The MoE preallocates sixteen 224-square expert slots.  Four slots are enabled
for each incoming task, while the canvas, propagation kernel, OEO, global phase
and full-plane CCD/MLP interface stay fixed for the complete task sequence.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ARCHIVE = Path(__file__).resolve().parents[2] / "demo_check/adrenal_softsign_code_export_20260915_145336/code"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))
from optical_reference.optics import AngularSpectrumPropagator

TASK_ORDER = ("eurosat", "clevr", "speech", "physical")


def normalize_power(x, power=1.0):
    return x * (power / x.square().sum((-2, -1), keepdim=True).clamp_min(1e-20)).sqrt()


class CrossModalOptics(nn.Module):
    def __init__(self, architecture="moe", seed=17, phase_dropout=0.05,
                 readout_grid=16, head_width=64, head_bottleneck=0, optical_layers=2,
                 max_experts=16, oeo_activation="intensity_softsign",
                 routing_temperature=1.0):
        super().__init__()
        if architecture not in {"moe", "d2nn"}:
            raise ValueError(architecture)
        self.architecture = architecture
        if max_experts not in (4, 16):
            raise ValueError("max_experts must be 4 or 16")
        self.max_experts = int(max_experts)
        self.grid = 2 if max_experts == 4 else 4
        self.expert_size, self.gap, self.border = 224, 30, 20
        self.height = self.grid * self.expert_size + (self.grid - 1) * self.gap + 2 * self.border
        self.width = self.height
        self.active_height, self.active_width = self.height - 2 * self.border, self.width - 2 * self.border
        # Each group is spatially spread over the aperture. Geometry never changes.
        order = ([(0, 0), (0, 1), (1, 0), (1, 1)] if max_experts == 4 else
                 [(0, 0), (0, 3), (3, 0), (3, 3),
                  (0, 1), (0, 2), (3, 1), (3, 2),
                  (1, 0), (1, 3), (2, 0), (2, 3),
                  (1, 1), (1, 2), (2, 1), (2, 2)])
        self.slots = [(self.border + r * (self.expert_size + self.gap),
                       self.border + c * (self.expert_size + self.gap)) for r, c in order]
        self.router_centers = [(y + 112, x + 112) for y, x in self.slots]
        self.phase_dropout = float(phase_dropout)
        self.readout_grid = int(readout_grid)
        # Keep these legacy constructor arguments loadable, but the formal
        # protocol intentionally permits exactly one electronic readout layer.
        self.head_width = int(head_width)
        self.head_bottleneck = int(head_bottleneck)
        self.optical_layers = int(optical_layers)
        self.routing_temperature = float(routing_temperature)
        if self.routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        if oeo_activation not in {"intensity_softsign", "centered_leaky_softsign"}:
            raise ValueError(oeo_activation)
        self.oeo_activation = oeo_activation
        if self.optical_layers < 2:
            raise ValueError("optical_layers must include expert/input and global phases")
        self.heads = nn.ModuleDict({
            name: self.make_head(classes)
            for name, classes in {"eurosat": 10, "clevr": 2, "speech": 8, "physical": 10}.items()
        })
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            if architecture == "moe":
                self.first_phase = nn.ParameterList([
                    nn.Parameter(torch.randn(224, 224) * 0.02) for _ in range(max_experts)])
                torch.manual_seed(seed + 103)
                self.router_phase = nn.Parameter(torch.randn(224, 224) * 0.02)
            else:
                self.first_phase = nn.Parameter(torch.randn(self.active_height, self.active_width) * 0.02)
                self.register_parameter("router_phase", None)
            torch.manual_seed(seed + 102)
            self.global_phase = nn.Parameter(torch.randn(self.active_height, self.active_width) * 0.02)
            self.additional_phases = nn.ParameterList([
                nn.Parameter(torch.randn(self.active_height, self.active_width) * 0.02)
                for _ in range(self.optical_layers - 2)])
        self.register_buffer("active_count", torch.tensor(4))
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=5.32e-7, pixel_size_m=1.7e-5,
            grid_size=(self.height, self.width), distance_m=0.1)

    def make_head(self, classes):
        features = self.readout_grid ** 2
        return nn.Linear(features, classes)

    def detector_centers(self, classes):
        """Fixed training-only detector locations spread over the active plane."""
        if classes == 2:
            # Matches the geometry that made the earlier CLEVR optical model
            # train reliably, expressed relative to an arbitrary aperture.
            return [(self.height // 2, self.width // 2 - self.active_width // 6),
                    (self.height // 2, self.width // 2 + self.active_width // 6)]
        rows, cols = 2, 5
        ys = torch.linspace(self.border + self.active_height * .3,
                            self.border + self.active_height * .7, rows)
        xs = torch.linspace(self.border + self.active_width * .15,
                            self.border + self.active_width * .85, cols)
        return [(int(y), int(x)) for y in ys for x in xs][:classes]

    @staticmethod
    def transmission(raw):
        return torch.exp(2j * torch.pi * torch.sigmoid(raw))

    @staticmethod
    def detect(intensity, centers, side):
        half = side // 2
        return torch.stack([intensity[:, y-half:y+half, x-half:x+half].sum((-2, -1))
                            for y, x in centers], 1)

    def configure_task(self, task_index, warmup=False):
        if task_index not in (0, 1, 2, 3):
            raise ValueError(task_index)
        self.active_count.fill_(4 * (task_index + 1))
        if self.architecture == "moe":
            start = 4 * task_index
            for i, p in enumerate(self.first_phase):
                p.requires_grad_(start <= i < start + 4)
            self.router_phase.requires_grad_(not warmup)
        else:
            self.first_phase.requires_grad_(True)
        self.global_phase.requires_grad_(not warmup)
        for phase in self.additional_phases:
            phase.requires_grad_(not warmup)
        current = TASK_ORDER[task_index]
        # A new task-specific head has no inherited weights, so it learns during
        # both new-expert warmup and the main stage. Previous heads stay frozen.
        for name, head in self.heads.items():
            head.requires_grad_(name == current)

    def configure_single_task(self, task):
        """Train a fresh four-expert MoE on one task, independent of sequence order."""
        if task not in TASK_ORDER:
            raise ValueError(task)
        if self.architecture != "moe":
            raise ValueError("configure_single_task is only valid for the MoE")
        self.active_count.fill_(4)
        for i, phase in enumerate(self.first_phase):
            phase.requires_grad_(i < 4)
        self.router_phase.requires_grad_(True)
        self.global_phase.requires_grad_(True)
        for phase in self.additional_phases:
            phase.requires_grad_(True)
        for name, head in self.heads.items():
            head.requires_grad_(name == task)

    def phase_mask(self, raw):
        value = self.transmission(raw)
        if self.training and self.phase_dropout:
            b = 8
            h, w = raw.shape[-2:]
            mask = torch.rand((1, 1, (h+b-1)//b, (w+b-1)//b), device=raw.device) < self.phase_dropout
            mask = mask.repeat_interleave(b, -2).repeat_interleave(b, -1)[0, 0, :h, :w]
            value = torch.where(mask, torch.ones_like(value), value)
        return value

    def oeo(self, field):
        b = self.border
        intensity = field[:, b:-b, b:-b].abs().square()
        intensity = intensity / intensity.mean((-2, -1), keepdim=True).clamp_min(1e-20)
        if self.oeo_activation == "intensity_softsign":
            amplitude = F.softsign(intensity)
        else:
            z = F.layer_norm(intensity, intensity.shape[-2:], eps=1e-5)
            amplitude = F.softsign(F.leaky_relu(z, negative_slope=0.1))
        amplitude = normalize_power(amplitude)
        return F.pad(amplitude, (b, b, b, b)).to(torch.complex64)

    def route(self, amplitude, warmup=False, expert_mask=None):
        n = int(self.active_count)
        allowed = torch.arange(self.max_experts, device=amplitude.device) < n
        if expert_mask is not None:
            allowed &= torch.as_tensor(expert_mask, device=amplitude.device, dtype=torch.bool)
        if warmup:
            allowed &= torch.arange(self.max_experts, device=amplitude.device) >= n - 4
        if not bool(allowed.any()):
            raise ValueError("empty expert mask")
        if warmup:
            return allowed.float().expand(len(amplitude), -1) / allowed.sum(), None
        dy, dx = self.height - 224, self.width - 224
        routed = F.pad(amplitude.to(torch.complex64) * self.transmission(self.router_phase),
                       (dx//2, dx-dx//2, dy//2, dy-dy//2))
        capture = self.detect(self.propagator(routed).abs().square(), self.router_centers, 60)
        weights = (capture + 1e-12).pow(1.0 / self.routing_temperature) * allowed
        return weights / weights.sum(1, keepdim=True), capture.sum(1)

    def forward(self, amplitude, task, warmup=False, expert_mask=None):
        if amplitude.ndim != 3 or tuple(amplitude.shape[-2:]) != (224, 224):
            raise ValueError(f"expected Bx224x224, got {tuple(amplitude.shape)}")
        amplitude = normalize_power(amplitude.float())
        if task not in self.heads:
            raise ValueError(task)
        if self.architecture == "d2nn":
            expanded = F.interpolate(amplitude[:, None], (self.active_height, self.active_width),
                                     mode="bilinear", align_corners=False)[:, 0]
            expanded = normalize_power(expanded)
            field = F.pad(expanded * self.phase_mask(self.first_phase), (self.border,) * 4)
            q, router_capture = None, None
        else:
            # New tasks may reuse all experts learned so far. Old tasks retain
            # the capacity available when they were learned, preventing later
            # experts from silently replacing frozen optical memory.
            task_capacity = min(self.max_experts, 4 * (TASK_ORDER.index(task) + 1))
            task_mask = torch.arange(self.max_experts, device=amplitude.device) < task_capacity
            if expert_mask is not None:
                task_mask &= torch.as_tensor(expert_mask, device=amplitude.device, dtype=torch.bool)
            q, router_capture = self.route(amplitude, warmup=warmup, expert_mask=task_mask)
            field = torch.zeros((len(amplitude), self.height, self.width),
                                device=amplitude.device, dtype=torch.complex64)
            for i in range(int(self.active_count)):
                if bool((q[:, i] > 0).any()):
                    y, x = self.slots[i]
                    field[:, y:y+224, x:x+224] = (amplitude * q[:, i, None, None].sqrt()
                                                   * self.phase_mask(self.first_phase[i]))
        field = self.oeo(self.propagator(field))
        for phase in (self.global_phase, *self.additional_phases):
            global_mask = F.pad(self.phase_mask(phase), (self.border,) * 4, value=1)
            field = self.oeo(self.propagator(field * global_mask))
        intensity = field[:, self.border:-self.border, self.border:-self.border].abs().square()
        # A camera samples the complete output plane. Fixed pooling bounds the
        # electronic interface without depending on hand-picked detector windows.
        features = F.adaptive_avg_pool2d(
            intensity[:, None], (self.readout_grid, self.readout_grid))[:, 0].flatten(1)
        features = torch.log1p(features / features.mean(1, keepdim=True).clamp_min(1e-20))
        logits = self.heads[task](features)
        classes = logits.shape[1]
        detector_side = max(32, min(64, self.active_width // 7))
        detector_energy = self.detect(field.abs().square(), self.detector_centers(classes),
                                      detector_side)
        detector_probabilities = ((detector_energy + 1e-12) /
                                  (detector_energy.sum(1, keepdim=True) +
                                   classes * 1e-12))
        return {"probabilities": logits.softmax(1), "logits": logits,
                "ccd_features": features,
                "detector_probabilities": detector_probabilities,
                "route_power": q, "router_capture": router_capture,
                "capture": intensity.sum((-2, -1))}
