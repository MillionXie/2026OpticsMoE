"""Original angular-spectrum path, zero phase start, and direct CCD readout."""

import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.tasks.t13_four_modal_lifelong.model import CrossModalOptics, normalize_power


class DirectCCDOptics(CrossModalOptics):
    """MoE/D2NN share output camera geometry and have no electronic class head.

    The camera sees the original final propagation plane. There is no Fourier
    lens, additional propagation, or trainable element after that propagation.
    """

    def __init__(self, architecture: str, *, router_side=84, output_side=128,
                 x_pitch=200, y_pitch=300, routing_temperature=1.25):
        super().__init__(architecture=architecture, seed=17, phase_dropout=0.0,
                         readout_grid=28, head_width=0, head_bottleneck=0,
                         optical_layers=2, max_experts=16,
                         oeo_activation="intensity_softsign",
                         routing_temperature=routing_temperature)
        self.heads = nn.ModuleDict()
        # Every physical phase mask starts from raw zero. The inherited map
        # exp(2πi sigmoid(raw)) then gives exactly π at every pixel.
        with torch.no_grad():
            if architecture == "moe":
                self.router_phase.zero_()
                for phase in self.first_phase:
                    phase.zero_()
            else:
                self.first_phase.zero_()
            self.global_phase.zero_()
            for phase in self.additional_phases:
                phase.zero_()
        self.router_side = int(router_side)
        if self.router_side <= 0 or self.router_side > 84 or self.router_side % 2:
            raise ValueError("router_side must be even and at most 84")
        c = self.height // 2
        inner = (-self.router_side // 2, self.router_side // 2)
        outer = (-3 * self.router_side // 2, -self.router_side // 2,
                 self.router_side // 2, 3 * self.router_side // 2)
        self.router_centers = ([(c + y, c + x) for y in inner for x in inner]
                               + [(c + y, c + x) for y in outer for x in outer
                                  if y not in inner or x not in inner])
        self.set_output_geometry(output_side, x_pitch, y_pitch)

    def set_output_geometry(self, side: int, x_pitch: int, y_pitch: int):
        """Three-four-three equal squares with physical gaps and row staggering."""
        side, x_pitch, y_pitch = int(side), int(x_pitch), int(y_pitch)
        if side <= 0 or side % 2 or x_pitch < side + 16 or y_pitch < side + 16:
            raise ValueError("windows must be even-sided and separated by >=16 pixels")
        c = self.height // 2
        centers = ([(c - y_pitch, c + k * x_pitch) for k in (-1, 0, 1)]
                   + [(c, c + int(k * x_pitch / 2)) for k in (-3, -1, 1, 3)]
                   + [(c + y_pitch, c + k * x_pitch) for k in (-1, 0, 1)])
        half = side // 2
        if any(y - half < 0 or y + half > self.height or
               x - half < 0 or x + half > self.width for y, x in centers):
            raise ValueError("classification window extends outside CCD")
        self.output_side = side
        self.x_pitch, self.y_pitch = x_pitch, y_pitch
        self.output_centers = centers

    def configure_stage(self, stage_index: int):
        if stage_index not in range(4):
            raise ValueError(stage_index)
        active = 4 * (stage_index + 1)
        self.active_count.fill_(active)
        if self.architecture == "moe":
            for index, phase in enumerate(self.first_phase):
                phase.requires_grad_(4 * stage_index <= index < active)
            self.router_phase.requires_grad_(True)
        else:
            self.first_phase.requires_grad_(True)
        self.global_phase.requires_grad_(True)
        for phase in self.additional_phases:
            phase.requires_grad_(True)

    @staticmethod
    def window_power(intensity, centers, side):
        half = side // 2
        return torch.stack([intensity[:, y-half:y+half, x-half:x+half].sum((-2, -1))
                            for y, x in centers], 1)

    def route_with_efficiency(self, amplitude, return_debug=False):
        active = int(self.active_count)
        dy, dx = self.height - 224, self.width - 224
        field = F.pad(amplitude.to(torch.complex64) * self.transmission(self.router_phase),
                      (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        intensity = self.propagator(field).abs().square()
        power = self.window_power(intensity, self.router_centers, self.router_side)
        scores = (power[:, :active] + 1e-12).pow(1 / self.routing_temperature)
        route = F.pad(scores / scores.sum(1, keepdim=True).clamp_min(1e-20),
                      (0, self.max_experts - active))
        efficiency = power[:, :active].sum(1) / intensity.sum((-2, -1)).clamp_min(1e-20)
        return route, efficiency, intensity if return_debug else None

    def forward(self, amplitude: torch.Tensor, *, return_debug=False):
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
            route, router_efficiency, router_ccd = None, None, None
        else:
            route, router_efficiency, router_ccd = self.route_with_efficiency(
                amplitude, return_debug=return_debug)
            field = torch.zeros((len(amplitude), self.height, self.width),
                                dtype=torch.complex64, device=amplitude.device)
            for index in range(int(self.active_count)):
                y, x = self.slots[index]
                field[:, y:y+224, x:x+224] = (
                    amplitude * route[:, index, None, None].sqrt()
                    * self.phase_mask(self.first_phase[index]))
        # The existing OEO is applied only between phase layers, identically
        # for both architectures; the final propagated complex field reaches CCD.
        field = self.oeo(self.propagator(field))
        phases = (self.global_phase, *self.additional_phases)
        pre_final_field = None
        for index, phase in enumerate(phases):
            mask = F.pad(self.phase_mask(phase), (self.border,) * 4, value=1)
            if return_debug and index == len(phases) - 1:
                pre_final_field = field * mask
            propagated = self.propagator(field * mask)
            field = propagated if index == len(phases) - 1 else self.oeo(propagated)
        intensity = field.abs().square()
        powers = self.window_power(intensity, self.output_centers, self.output_side)
        # A 1e-12 hard floor erased the target-window gradient in the
        # zero-phase four-corner MoE (one window had 1.2e-14 power). A tiny
        # additive floor keeps the physically measured energy differentiable.
        logits = (powers + 1e-20).log()
        readout_efficiency = powers.sum(1) / intensity.sum((-2, -1)).clamp_min(1e-20)
        result = {"logits": logits, "window_power": powers,
                  "readout_efficiency": readout_efficiency,
                  "route_power": route, "router_efficiency": router_efficiency}
        if return_debug:
            result["router_ccd"] = router_ccd
            result["final_ccd"] = intensity
            result["pre_final_field"] = pre_final_field
        return result
