"""The t13 optical physics with one task-independent, permanently frozen readout."""

import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.tasks.t13_four_modal_lifelong.model import CrossModalOptics, normalize_power


def fixed_readout_weight(features: int = 784, classes: int = 10, seed: int = 17024):
    """Centered, orthonormal readout rows shared by both optical architectures."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        directions = torch.randn(features, classes)
        directions -= directions.mean(0, keepdim=True)
        basis, _ = torch.linalg.qr(directions, mode="reduced")
    return basis.T.contiguous()


def fixed_window_weight():
    """Ten predetermined CCD regions represented by the same single Linear."""
    weights = torch.zeros(10, 28, 28)
    for label in range(10):
        row, column = divmod(label, 5)
        left = round(column * 28 / 5)
        right = round((column + 1) * 28 / 5)
        weights[label, row * 14:(row + 1) * 14, left:right] = 1.0
    weights = weights.flatten(1)
    weights -= weights.mean(1, keepdim=True)
    return torch.nn.functional.normalize(weights, dim=1)


class SharedReadoutOptics(CrossModalOptics):
    """Use the same fixed 10-output Linear for all tasks, without a task ID in forward.

    Labels retain their published local indices 0..C-1. Every forward returns
    all ten logits; unused output positions are genuine competing classes, not
    masked with knowledge of the current task. A single model can therefore
    be evaluated on every task without selecting a task-specific head or expert
    subset at inference time.
    """

    def __init__(self, architecture: str, *, seed: int = 17,
                 max_experts: int = 16, routing_temperature: float = 1.25,
                 readout_seed: int = 17024, readout_gain: float = 1.0,
                 readout_design: str = "orthogonal"):
        if readout_gain <= 0:
            raise ValueError("readout_gain must be positive")
        if readout_design not in {"orthogonal", "windows"}:
            raise ValueError("unknown fixed readout design")
        super().__init__(architecture=architecture, seed=seed,
                         phase_dropout=0.0, readout_grid=28,
                         head_width=0, head_bottleneck=0, optical_layers=2,
                         max_experts=max_experts,
                         oeo_activation="intensity_softsign",
                         routing_temperature=routing_temperature)
        self.heads = nn.ModuleDict()
        self.shared_head = nn.Linear(784, 10, bias=False)
        with torch.no_grad():
            rows = (fixed_readout_weight(seed=readout_seed)
                    if readout_design == "orthogonal" else fixed_window_weight())
            self.shared_head.weight.copy_(rows * readout_gain)
        self.shared_head.requires_grad_(False)
        self.readout_seed = int(readout_seed)
        self.readout_gain = float(readout_gain)
        self.readout_design = readout_design

    def configure_stage(self, stage_index: int, *, warmup: bool = False):
        if stage_index not in range(4):
            raise ValueError(stage_index)
        if warmup and (self.architecture != "moe" or stage_index == 0):
            raise ValueError("warmup is only for a new MoE expert group")
        active = 4 * (stage_index + 1)
        if active > self.max_experts:
            raise ValueError("stage exceeds preallocated expert capacity")
        self.active_count.fill_(active)
        if self.architecture == "moe":
            start = 4 * stage_index
            for i, phase in enumerate(self.first_phase):
                phase.requires_grad_(start <= i < active)
            self.router_phase.requires_grad_(not warmup)
        else:
            self.first_phase.requires_grad_(True)
        self.global_phase.requires_grad_(not warmup)
        for phase in self.additional_phases:
            phase.requires_grad_(not warmup)
        self.shared_head.requires_grad_(False)

    def forward(self, amplitude: torch.Tensor, *, warmup: bool = False):
        if amplitude.ndim != 3 or tuple(amplitude.shape[-2:]) != (224, 224):
            raise ValueError(f"expected Bx224x224, got {tuple(amplitude.shape)}")
        amplitude = normalize_power(amplitude.float())
        if self.architecture == "d2nn":
            expanded = F.interpolate(amplitude[:, None],
                                     (self.active_height, self.active_width),
                                     mode="bilinear", align_corners=False)[:, 0]
            expanded = normalize_power(expanded)
            field = F.pad(expanded * self.phase_mask(self.first_phase),
                          (self.border,) * 4)
            route_power = None
        else:
            route_power, _ = self.route(amplitude, warmup=warmup,
                                        expert_mask=None)
            field = torch.zeros((len(amplitude), self.height, self.width),
                                device=amplitude.device, dtype=torch.complex64)
            for i in range(int(self.active_count)):
                if bool((route_power[:, i] > 0).any()):
                    y, x = self.slots[i]
                    field[:, y:y + 224, x:x + 224] = (
                        amplitude * route_power[:, i, None, None].sqrt()
                        * self.phase_mask(self.first_phase[i]))
        field = self.oeo(self.propagator(field))
        for phase in (self.global_phase, *self.additional_phases):
            phase_field = F.pad(self.phase_mask(phase),
                                (self.border,) * 4, value=1)
            field = self.oeo(self.propagator(field * phase_field))
        intensity = field[:, self.border:-self.border,
                          self.border:-self.border].abs().square()
        features = F.adaptive_avg_pool2d(intensity[:, None], (28, 28))[:, 0].flatten(1)
        features = torch.log1p(features / features.mean(1, keepdim=True).clamp_min(1e-20))
        logits = self.shared_head(features)
        return {"logits": logits, "probabilities": logits.softmax(1),
                "ccd_features": features, "route_power": route_power}
