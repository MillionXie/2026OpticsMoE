"""Two optical-CCD-directed image copies on one continuous first phase plane."""
import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.demo_check.pure_optical.models import PhaseOnly
from LightGenV2.tasks.t17_router_classification.model_four import CONTRACT


class SpatialRoutingLG(PhaseOnly):
    """All reported positions are (y,x) top-left pixel offsets on the 478² face."""

    def __init__(self, mode, num_classes=8):
        if mode not in {"fixed", "continuous"}:
            raise ValueError(f"Unknown spatial routing mode: {mode}")
        # Allocate exactly the two full-size phase planes. The additional 224²
        # optical router uses the same zero-phase initialization as T17.
        super().__init__("full_d2nn", dict(CONTRACT))
        self.architecture = "spatial_routing"
        self.mode = mode
        self.router_phase = nn.Parameter(torch.zeros(224, 224))
        self.shared_head = nn.Linear(784, num_classes, bias=False)
        anchors = torch.linspace(0, 254, 17).round().long()
        self.register_buffer("axis", anchors, persistent=False)
        grid = torch.cartesian_prod(anchors, anchors)
        self.register_buffer("anchor_yx", grid, persistent=False)
        corners = torch.tensor([(0, 0), (0, 254), (254, 0), (254, 254)])
        self.register_buffer("corners", corners, persistent=False)
        offsets = torch.arange(-1, 2)
        self.register_buffer("neighbors", torch.cartesian_prod(offsets, offsets), persistent=False)
        pixels = torch.arange(225)
        self.register_buffer("patch_index", pixels[:, None] * 478 + pixels, persistent=False)

    def window_scores(self, intensity):
        # 60² detector windows. Crop starts at the first center (132) minus
        # its half width (30). Integer rim offsets exactly match the four
        # canonical detector apertures.
        integral = intensity[:, 102:416, 102:416].cumsum(1).cumsum(2)
        integral = F.pad(integral, (1, 0, 1, 0))
        lo, hi = self.axis, self.axis + 60
        return (integral[:, hi[:, None], hi] - integral[:, lo[:, None], hi]
                - integral[:, hi[:, None], lo] + integral[:, lo[:, None], lo]).clamp_min(0)

    def continuous_coordinates(self, scores):
        flat = scores.flatten(1)
        first = flat.argmax(1)
        distance = (self.anchor_yx[None] - self.anchor_yx[first, None]).abs().amax(-1)
        # Exclude overlapping router detector windows when locating peak 2.
        second = flat.masked_fill(distance < 60, -torch.inf).argmax(1)
        indices = torch.stack((first, second), 1)
        n = len(self.axis)
        peaks = torch.stack((indices // n, indices % n), -1)
        neighborhood = peaks[:, :, None, :] + self.neighbors
        valid = ((neighborhood >= 0) & (neighborhood < n)).all(-1)
        y, x = neighborhood.clamp(0, n - 1).unbind(-1)
        batch = torch.arange(len(flat), device=flat.device)[:, None, None]
        local_power = scores[batch, y, x]
        # A fixed power-weighted centroid gives sub-pixel positions with a
        # gradient through the local CCD intensities for each selected peak.
        weights = (local_power + 1e-12) * valid
        weights = weights / weights.sum(-1, keepdim=True)
        points = torch.stack((self.axis[y], self.axis[x]), -1)
        positions = (points * weights[..., None]).sum(-2)
        return positions, flat.gather(1, indices)

    def locate(self, amplitude):
        router_field = amplitude * self.transmission(self.router_phase)
        intensity = self.propagator(F.pad(router_field, (147,) * 4)).abs().square()
        if self.mode == "fixed":
            energy = self.detect(intensity, self.router_centers, 60)
            # Use the T17 normalized four-port scores before selecting so even
            # its tie convention is retained.
            dense = (energy + 1e-12) / (energy.sum(1, keepdim=True) + 4e-12)
            indices = dense.topk(2, dim=1).indices
            positions = self.corners[indices].to(amplitude.dtype)
            selected = energy.gather(1, indices)
        else:
            positions, selected = self.continuous_coordinates(self.window_scores(intensity))
        capture = selected.sum(1)
        power = (selected + 1e-12) / (capture[:, None] + 2e-12)
        return positions, power, capture

    def multiplex(self, amplitude, positions, power):
        """Coherently sum sub-pixel translations, then set total incident power."""
        # Use a 225² footprint per bilinear-translated 224² patch. At offset
        # 254, fraction=1 with base=253 keeps the footprint inside the face.
        base = positions.floor().long().clamp(0, 253)
        frac_y, frac_x = (positions - base).unbind(-1)
        wy = frac_y[..., None, None]
        wx = frac_x[..., None, None]
        patch = amplitude[:, None]
        row = (1 - wy) * F.pad(patch, (0, 0, 0, 1)) + wy * F.pad(patch, (0, 0, 1, 0))
        patch = (1 - wx) * F.pad(row, (0, 1, 0, 0)) + wx * F.pad(row, (1, 0, 0, 0))
        patch = patch * power.sqrt()[..., None, None]
        start = base[..., 0] * 478 + base[..., 1]
        indices = start[..., None, None] + self.patch_index
        active = amplitude.new_zeros(len(amplitude), 478 * 478)
        active = active.scatter_add(1, indices.flatten(1), patch.flatten(1))
        active = active / active.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return active.reshape(-1, 478, 478)

    def forward(self, amplitude):
        if amplitude.ndim != 3 or amplitude.shape[-2:] != (224, 224):
            raise ValueError("Expected encoded Bx224x224 amplitudes")
        amplitude = amplitude / amplitude.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)
        positions, power, capture = self.locate(amplitude)
        entrance = self.multiplex(amplitude, positions, power)
        first = F.pad(entrance * self.transmission(self.first_phase), (20,) * 4)
        field = self.oeo(self.propagator(first))
        global_mask = F.pad(self.transmission(self.global_phase), (20,) * 4, value=1)
        field = self.oeo(self.propagator(field * global_mask))
        active = field[:, 20:498, 20:498].abs().square()
        features = F.adaptive_avg_pool2d(active[:, None], 28).flatten(1)
        features = features / features.mean(1, keepdim=True).clamp_min(1e-20)
        return dict(logits=self.shared_head(features), ccd_features=features,
                    positions=positions, power=power, router_capture=capture)
