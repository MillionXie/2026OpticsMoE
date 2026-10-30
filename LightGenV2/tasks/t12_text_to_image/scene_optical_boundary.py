"""SDK-free CCD boundary for the real Qwen3-VL single-EG scene generator.

All learned codecs, electronic residuals, routing normalization/top-k, readout,
fusion and generator remain intact. This only replaces physical detector
calculations. A provider must explicitly supply each requested CCD frame.
"""
import torch
from torch.nn import functional as F

STAGES=('vision_router','vision_expert','vision_global',
        'language_router','language_expert','language_global')


def physical_field(field,modulation):
    """Preserve input complex/sign phase; abs(field) alone is not equivalent."""
    amplitude=field.abs()
    phase=torch.remainder(torch.angle(field.to(torch.complex64))+
        torch.angle(modulation.to(torch.complex64)),2*torch.pi)
    if not torch.isfinite(amplitude).all() or not torch.isfinite(phase).all():
        raise ValueError('Nonfinite optical payload')
    if amplitude.max()>1.000001:raise ValueError('Input must already be bounded exactly once')
    return amplitude,phase


def checked_ccd(value,field,active_size):
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(len(field),active_size,active_size):
        raise ValueError('Provider must return the full calibrated active CCD ROI')
    if not torch.isfinite(value).all() or (value<0).any():
        raise ValueError('CCD must be finite, nonnegative and explicitly calibrated')
    return value.to(device=field.device,dtype=torch.float32)


def attach(macros,provider,*,simulation_diagnostic=False):
    """provider(stage, bounded_amplitude, total_phase, ideal_or_None) -> CCD.

    Diagnostic mode computes an explicitly labelled ideal CCD. Replay mode
    executes NO propagation simulation at these boundaries. There is no silent
    ideal fallback when a measured frame is missing. Provider governs cohort
    caching and calibration and may raise to pause at the next capture stage.
    """
    if set(macros)!={'vision','language'} or any(m.pairs!=1 for m in macros.values()):
        raise ValueError('Exactly one router and one E/G pair per modality required')
    originals=[]
    for name in ['vision','language']:
        macro=macros[name];path=macro.paths[0];router=path.core.router;active=path.core.geometry.active_aperture
        if (router.input_size,router.active_size,router.canvas_size)!=(224,478,518):
            raise ValueError('Physical field geometry changed')
        if any(getattr(router,n)!=0 for n in ['input_shift_pixels','phase_shift_pixels','ccd_shift_pixels']):
            raise ValueError('Export requires the actual clean deployment path')
        for holder in [path,router]:
            if not hasattr(holder,'_t12_physical_simulate'):raise ValueError('Bounded native T12 path required')
        original_router=router._t12_physical_simulate;original_detector=path._t12_physical_simulate
        def route(fields,name=name,router=router,original=original_router):
            # Fields passed here are already tanh-bounded. No second tanh or
            # per-frame peak normalization is applied.
            margin=(router.active_size-router.input_size)//2
            padded=F.pad(fields,(margin,margin,margin,margin))
            modulation=torch.exp(1j*router.active_phase()).unsqueeze(0)
            amplitude,phase=physical_field(padded,modulation)
            ideal=original(fields) if simulation_diagnostic else None
            value=checked_ccd(provider(name+'_router',amplitude,phase,ideal),fields,router.active_size)
            router.last_detector_intensity=value.detach()
            return value
        def detect(field,modulation,shifts,*,phase_support=None,name=name,active=active,original=original_detector):
            if phase_support not in ['expert','global'] or any(any(v) for v in shifts.values()):
                raise ValueError('Unexpected detector stage/shift')
            y0,y1,x0,x1=active.y0,active.y1,active.x0,active.x1
            value=field[:,y0:y1,x0:x1];phase_value=modulation[:,y0:y1,x0:x1]
            # The physical active aperture must contain the complete input.
            energy=field.abs().square().sum();inside=value.abs().square().sum()
            if float((energy-inside).abs())>max(1e-5,float(energy)*1e-6):
                raise ValueError('Cannot crop nonzero field outside the physical active aperture')
            amplitude,phase=physical_field(value,phase_value)
            ideal=original(field,modulation,shifts,phase_support=phase_support) if simulation_diagnostic else None
            return checked_ccd(provider(name+'_'+phase_support,amplitude,phase,ideal),field,y1-y0)
        originals.extend([(router,original_router),(path,original_detector)])
        router._t12_physical_simulate=route;path._t12_physical_simulate=detect
    def restore():
        for holder,original in originals:holder._t12_physical_simulate=original
    return restore
