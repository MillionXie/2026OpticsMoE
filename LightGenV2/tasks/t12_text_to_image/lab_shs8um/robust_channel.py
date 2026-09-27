"""Training-only coherent optical channel; no new deployment parameters."""
import math
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .ccd_bridge import attach


PROFILES = json.loads((Path(__file__).resolve().parents[1]/'configs/optical_channel_robust_v2.json').read_text())


def coherent_modulation(modulation, eta, relative_phase):
    """eta is nominal branch POWER fraction, before interference (not amplitude)."""
    return (1-eta).sqrt()*modulation + eta.sqrt()*torch.exp(1j*relative_phase)


class RobustChannel:
    def __init__(self, model):
        if getattr(model, 'bounded_amplitude', None) != {'kind': 'tanh', 'scale': .5}:
            raise ValueError('Require the deployed zero-preserving tanh/0.5 amplitude contract')
        self.active = False
        self.strength = 1.
        self.profile = PROFILES['camera']
        self.paths = (model.text.optical, model.editor.bottleneck.optical)
        self.originals = []
        for path in self.paths:
            self.patch(path, '_apply_coherent_zero_order', self.phase_path(path))
            self.patch(path, '_draw_stage_shifts', lambda stage: self.shifts())
            router = path.core.router
            original = router._phase_modulation
            self.patch(router, '_phase_modulation', lambda batch, original=original: self.mix(original(batch)))
            router.noise_std = 0.
            router.phase_dropout_p = 0.
            for prop in (path.core.propagator, router.propagator):
                original = prop.forward
                self.patch(prop, 'forward', self.propagation(prop, original))
        self.restore_bridge = attach(model, self.camera)

    def patch(self, obj, name, value):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def configure(self, profile=None, strength=1.):
        self.active = profile is not None
        self.strength = strength
        if profile is not None: self.profile = PROFILES[profile]
        for path in self.paths:
            # All implicit old noises/dropouts are disabled. Only the explicit
            # channel is enabled, including router's existing geometric shifts.
            path.eval()
            router = path.core.router
            router.train(self.active)
            shift = self.profile['shift'] if self.active else 0
            router.input_shift_pixels = router.phase_shift_pixels = router.ccd_shift_pixels = shift

    def shifts(self):
        shift = self.profile['shift'] if self.active else 0
        return {name: tuple(int(v) for v in torch.randint(-shift, shift+1, (2,)))
                for name in ('input', 'phase', 'ccd')}

    def mix(self, modulation):
        if not self.active: return modulation
        p = self.profile
        eta = modulation.real.new_empty((len(modulation),1,1)).uniform_(p['power_min'], p['power_max'])*self.strength
        angle = modulation.real.new_empty((len(modulation),1,1)).uniform_(-math.pi,math.pi)
        drop = p['phase_dropout']*self.strength
        if drop:
            coarse = torch.rand(len(modulation),1,math.ceil(modulation.shape[-2]/8),math.ceil(modulation.shape[-1]/8),device=modulation.device)<drop
            bypass = F.interpolate(coarse.float(), modulation.shape[-2:], mode='nearest')[:,0].bool()
            modulation = torch.where(bypass, torch.ones_like(modulation), modulation)
        return coherent_modulation(modulation, eta, angle)

    def phase_path(self, path):
        def apply(field, modulation, *, phase_support):
            if not self.active: return field, modulation
            support = path._phase_support_mask(modulation, phase_support)
            # Leakage carries the SAME bounded incident field; it cannot add
            # light to zero-amplitude pixels. Propagation then forms interference.
            return field, torch.where(support, self.mix(modulation), modulation)
        return apply

    def propagation(self, prop, original):
        n = prop.grid_size
        frequency = torch.fft.fftfreq(n, device=prop.transfer_function.device)
        fy, fx = torch.meshgrid(frequency, frequency, indexing='ij')
        radial = (fx.square()+fy.square())/.5
        astigmatism = (fx.square()-fy.square())/.25
        def forward(field):
            amount = self.profile['kspace']*self.strength if self.active else 0.
            if not amount: return original(field)
            shape = (len(field),1,1)
            attenuation = field.real.new_empty(shape).uniform_(0, amount)
            defocus = field.real.new_empty(shape).uniform_(-amount,amount)
            astig = field.real.new_empty(shape).uniform_(-amount,amount)
            response = torch.exp(-attenuation*radial)*torch.exp(1j*(defocus*radial+astig*astigmatism))
            return torch.fft.ifft2(torch.fft.fft2(field.to(torch.complex64))*prop.transfer_function*response)
        return forward

    def camera(self, stage, amplitude, phase, ideal):
        if not self.active: return ideal
        p, s = self.profile, self.strength
        shape = (len(ideal),1,1)
        gain = ideal.new_empty(shape).uniform_(1-p['gain']*s,1+p['gain']*s)
        bias = ideal.new_empty(shape).uniform_(0,p['bias']*s)
        std = (p['read']*s)**2 + (p['shot']*s)**2*ideal.detach().clamp_min(0)
        # Detector intensity units of bounded incident amplitude, not peak/batch
        # normalized, and not calibrated camera electrons. Bias/noise are CCD-only.
        return (ideal*gain+bias+torch.randn_like(ideal)*std.sqrt()).clamp_min(0)

    def restore(self):
        self.restore_bridge()
        for obj, name, value in reversed(self.originals): setattr(obj, name, value)
