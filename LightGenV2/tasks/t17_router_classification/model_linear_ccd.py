"""Three-layer coherent D2NN: fixed eight/nine-class CCD, no OEO/learned head."""
import torch
from torch import nn
from torch.nn import functional as F
from .model_four import CONTRACT, ccd_grid_energies
from LightGenV2.demo_check.pure_optical.models import AngularSpectrumPropagator, PhaseOnly


class LinearD2NNCCD(nn.Module):
    def __init__(self, layers=3, num_classes=9):
        super().__init__()
        if layers != 3:
            raise ValueError('This profile fixes three main phase layers')
        if num_classes not in (8, 9):
            raise ValueError('Fixed CCD layouts support eight or nine classes')
        self.num_classes = num_classes
        self.phases = nn.ParameterList([nn.Parameter(torch.zeros(478, 478)) for _ in range(layers)])
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=CONTRACT['wavelength_m'], pixel_size_m=CONTRACT['pixel_size_m'],
            grid_size=518, distance_m=CONTRACT['propagation_m'])

    def propagate_field(self, field):
        """Linear in the encoded complex field; no normalization inside this path."""
        for phase in self.phases:
            mask = F.pad(PhaseOnly.transmission(phase), (20,)*4, value=1)
            field = self.propagator(field * mask)
        return field

    def forward(self, amplitude):
        if amplitude.ndim != 3 or amplitude.shape[-2:] != (224, 224):
            raise ValueError('Expected the original fixed RGB encoding Bx224x224')
        expanded = F.interpolate(amplitude[:, None], (478, 478), mode='bilinear', align_corners=False)[:, 0]
        expanded = expanded / expanded.square().sum((-2, -1), keepdim=True).clamp_min(1e-20).sqrt()
        field = self.propagate_field(F.pad(expanded, (20,)*4).to(torch.complex64))
        intensity = field.abs().square()[:, 20:498, 20:498]
        if self.num_classes == 9:
            energy = ccd_grid_energies(intensity)
        else:
            # Eight disjoint regions cover the entire active CCD, row-major 0..7.
            ys = (0, 239, 478)
            xs = (0, 119, 239, 358, 478)
            energy = torch.stack([intensity[:, ys[r]:ys[r+1], xs[c]:xs[c+1]].sum((-2,-1))
                                  for r in range(2) for c in range(4)], dim=1)
        total = energy.sum(1, keepdim=True).clamp_min(1e-20)
        p = (energy/total + 1e-12)/(1 + self.num_classes*1e-12)
        return dict(logits=p.log(), probabilities=p, class_energies=energy,
                    route_power=None, router_capture=None)
