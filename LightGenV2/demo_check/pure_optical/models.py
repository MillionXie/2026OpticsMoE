"""Phase-only main paths with one shared fixed input encoding and CCD readout.

Dynamic routing retains an optical measurement and electronic normalization / SLM
control. The three-channel packing is a fixed digital input preparation. No claim
is made that those operations are passive optics or hardware-validated.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ARCHIVE = Path(__file__).resolve().parents[1] / 'adrenal_softsign_code_export_20260915_145336/code'
sys.path.insert(0, str(ARCHIVE))
from optical_reference.optics import AngularSpectrumPropagator


def encode(images, power=1.0):
    """NHWC uint8 image -> fixed three-tile 224-square amplitude, no learnt adapter."""
    rgb = images.permute(0, 3, 1, 2).float() / 255.0
    rgb = F.interpolate(rgb, (112, 112), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    top = torch.cat((rgb[:, 0], rgb[:, 1]), dim=-1)
    bottom = torch.cat((rgb[:, 2], torch.zeros_like(rgb[:, 2])), dim=-1)
    field = torch.cat((top, bottom), dim=-2)
    norm = field.square().sum((-2, -1), keepdim=True)
    if bool((norm <= 1e-12).any()):
        raise ValueError('Zero-power input is not allowed')
    return field * (power / norm).sqrt()


class PhaseOnly(nn.Module):
    def __init__(self, architecture, cfg):
        super().__init__()
        self.architecture = architecture
        self.cfg = cfg
        assert architecture in cfg['architectures']
        assert (cfg['input_size'], cfg['active_size'], cfg['canvas_size']) == (224, 478, 518)
        first_shape = (478, 478) if architecture == 'full_d2nn' else (4, 224, 224)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'] + 101)
            self.first_phase = nn.Parameter(torch.randn(first_shape) * cfg['phase_init_std'])
            torch.manual_seed(cfg['seed'] + 102)
            self.global_phase = nn.Parameter(torch.randn(478, 478) * cfg['phase_init_std'])
            if architecture == 'dynamic_four':
                torch.manual_seed(cfg['seed'] + 103)
                self.router_phase = nn.Parameter(torch.randn(224, 224) * cfg['phase_init_std'])
            else:
                self.register_parameter('router_phase', None)
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=cfg['wavelength_m'], pixel_size_m=cfg['pixel_size_m'],
            grid_size=518, distance_m=cfg['propagation_m'])
        self.apertures = [(20, 20), (20, 274), (274, 20), (274, 274)]
        self.class_centers = [(y, x) for y in (160, 358) for x in (80, 170, 259, 348, 438)]
        self.router_centers = [(132, 132), (132, 386), (386, 132), (386, 386)]

    @staticmethod
    def transmission(raw):
        return torch.exp(2j * torch.pi * torch.sigmoid(raw))

    @staticmethod
    def detect(intensity, centers, side):
        half = side // 2
        return torch.stack([intensity[:, y-half:y+half, x-half:x+half].sum((-2, -1))
                            for y, x in centers], dim=1)

    def route(self, amplitude):
        if self.router_phase is None:
            return amplitude.new_full((len(amplitude), 4), 0.25), None
        field = F.pad(amplitude.to(torch.complex64) * self.transmission(self.router_phase), (147,)*4)
        intensity = self.propagator(field).abs().square()
        energy = self.detect(intensity, self.router_centers, self.cfg['router_detector_size'])
        fractions = (energy + 1e-12) / (energy.sum(1, keepdim=True) + 4e-12)
        return fractions, energy.sum(1)

    def forward(self, images):
        amplitude = encode(images, self.cfg['main_input_power'])
        q, router_capture = self.route(amplitude)
        if self.architecture == 'full_d2nn':
            expanded = F.interpolate(amplitude[:, None], (478, 478), mode='bilinear', align_corners=False)[:, 0]
            expanded = expanded * (self.cfg['main_input_power'] / expanded.square().sum((-2,-1), keepdim=True)).sqrt()
            entrance = F.pad(expanded, (20,)*4).to(torch.complex64)
            first = F.pad(expanded * self.transmission(self.first_phase), (20,)*4)
        else:
            entrance = amplitude.new_zeros((len(amplitude), 518, 518), dtype=torch.complex64)
            first = torch.zeros_like(entrance)
            for i, (y, x) in enumerate(self.apertures):
                value = amplitude * q[:, i, None, None].sqrt()
                entrance[:, y:y+224, x:x+224] = value
                first[:, y:y+224, x:x+224] = value * self.transmission(self.first_phase[i])
        # Full-canvas coherent propagation; no independent-window propagation,
        # intermediate detector, re-encoding, or lossless software concatenation.
        propagated = self.propagator(first)
        global_mask = F.pad(self.transmission(self.global_phase), (20,)*4, value=1)
        field = self.propagator(propagated * global_mask)
        intensity = field.abs().square()
        energies = self.detect(intensity, self.class_centers, self.cfg['detector_size'])
        probabilities = (energies + 1e-12) / (energies.sum(1, keepdim=True) + 10e-12)
        return dict(probabilities=probabilities, route_power=q if self.architecture != 'full_d2nn' else None,
                    input_power=entrance.abs().square().sum((-2,-1)),
                    output_power=intensity.sum((-2,-1)), detector_capture=energies.sum(1),
                    router_capture=router_capture)


def objective(output, labels):
    return F.nll_loss(output['probabilities'].clamp_min(1e-12).log(), labels)
