"""Explicit physical-amplitude units at the three original LSP propagators.

Fixed scale, never per-image normalization. Propagate the physical bounded
amplitude and restore complex field units BEFORE the existing detector/readout.
No learned parameters, clipping, weight changes, or device imports.
"""
from __future__ import annotations

import math
from types import MethodType

STAGES = ('router', 'expert', 'global')


def physical_field(field, scale=16., quantize=False):
    import torch
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid fixed amplitude scale')
    amplitude = field.abs()
    if not torch.isfinite(field).all() or amplitude.max() > scale:
        raise ValueError('Amplitude exceeds the fixed physical encoding range; never clip')
    physical = field / scale
    if quantize:
        encoded = torch.round(amplitude / scale * 255) / 255
        # Avoid changing the original phase, including true zero support.
        direction = field / amplitude.clamp_min(torch.finfo(amplitude.dtype).tiny)
        physical = direction * encoded
    return physical


class FieldUnits:
    """Scoped interception with contiguous-prefix measured physical CCD support."""
    def __init__(self, model, *, scale=16., quantize=False, measured=None, planes=None):
        self.model, self.scale, self.quantize = model, scale, quantize
        self.measured = measured or {}
        self.planes = planes
        if tuple(self.measured) != STAGES[:len(self.measured)]:
            raise ValueError('CCDs must be a contiguous physical-stage prefix')
        self.originals, self.amplitudes, self.detectors = [], {}, {}
        self.index = 0

    def __enter__(self):
        core = self.model.core.optical_branch.core
        for owner, prop in (('router', core.router.propagator), ('body', core.propagator)):
            original = prop.forward
            self.originals.append((prop, original))

            def forward(module, field, original=original, owner=owner):
                import torch
                if self.index >= 3 or owner != ('router', 'body', 'body')[self.index]:
                    raise RuntimeError('Unexpected LSP optical propagation order')
                stage = STAGES[self.index]
                self.index += 1
                physical = physical_field(field, self.scale, self.quantize)
                a = core.geometry.active_aperture
                active = physical[:, a.y0:a.y1, a.x0:a.x1]
                if self.planes is not None:
                    phase = torch.as_tensor(self.planes[stage], device=field.device)
                    recovered = active * torch.exp(-1j * phase)
                    if recovered.imag.abs().max() > 2e-5 or recovered.real.min() < -2e-5:
                        raise RuntimeError('Exported phase does not represent the propagated input')
                self.amplitudes[stage] = active.abs().detach()
                if stage in self.measured:
                    detector = self.measured[stage].to(field.device).float()
                    if detector.shape != active.shape or not torch.isfinite(detector).all() or detector.min() < 0:
                        raise ValueError('Invalid physical CCD intensity')
                    output = torch.zeros_like(field)
                    output[:, a.y0:a.y1, a.x0:a.x1] = detector.sqrt().to(field.dtype)
                    self.detectors[stage] = detector.detach()
                else:
                    output = original(physical)
                    self.detectors[stage] = output[:, a.y0:a.y1, a.x0:a.x1].abs().square().detach()
                return output * self.scale

            prop.forward = MethodType(forward, prop)
        return self

    def __exit__(self, *_args):
        for prop, original in self.originals:
            prop.forward = original
