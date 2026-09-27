"""Per-frame detector-floor suppression. Never changes SLM/BMP amplitude encoding."""
import torch


def correct(intensity,spec):
    if not spec:return intensity
    q=float(spec['floor_quantile'])
    if not 0<q<.5:raise ValueError('Detector floor quantile must be in (0,.5)')
    # Detached robust background estimate in CCD intensity units, not peak scaling.
    floor=torch.quantile(intensity.detach().flatten(1),q,dim=1)[:,None,None]
    return (intensity-floor).clamp_min(0)


def install(model,spec):
    from .ccd_bridge import attach
    model.detector_correction=dict(spec)
    model._default_detector_bridge_restore=attach(model,lambda stage,amplitude,phase,ideal:ideal)
