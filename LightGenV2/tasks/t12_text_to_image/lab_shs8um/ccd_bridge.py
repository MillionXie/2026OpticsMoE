"""Replace six optical detector calculations, preserving the trained graph."""
import torch


def attach(model, capture):
    """capture(stage, active_amplitude, active_phase, ideal_ccd) returns CCD.

    Router weighting, electronic residuals, normalization, and the decoder remain
    untouched. The ideal calculation is only a diagnostic/reference, never used
    instead of physical pixels by the laboratory callback.
    """
    originals=[]
    for name,path in [('language',model.text.optical),('vision',model.editor.bottleneck.optical)]:
        router=path.core.router
        original_router=router._simulate
        original_detector=path._simulate_detector_roi
        active=path.core.geometry.active_aperture
        def router_capture(fields,router=router,original=original_router,name=name,active=active):
            ideal=original(fields)
            amplitude=router.last_input_amplitude[:,active.y0:active.y1,active.x0:active.x1].abs()
            phase=torch.zeros_like(amplitude)
            margin=(phase.shape[-1]-router.input_size)//2
            phase[:,margin:margin+router.input_size,margin:margin+router.input_size]=torch.angle(router._phase_modulation(len(fields)))
            measured=capture(name+'_router',amplitude,phase,ideal)
            if measured.shape!=ideal.shape:raise ValueError('Router CCD shape mismatch')
            router.last_detector_intensity=measured.detach()
            return measured
        def detector_capture(field,modulation,shifts,*,phase_support=None,original=original_detector,name=name,active=active):
            ideal=original(field,modulation,shifts,phase_support=phase_support)
            amplitude=field[:,active.y0:active.y1,active.x0:active.x1].abs()
            phase=torch.angle(modulation[:,active.y0:active.y1,active.x0:active.x1])
            measured=capture(name+'_'+phase_support,amplitude,phase,ideal)
            if measured.shape!=ideal.shape:raise ValueError('Expert/global CCD shape mismatch')
            return measured
        originals.extend([(router,'_simulate',original_router),(path,'_simulate_detector_roi',original_detector)])
        router._simulate=router_capture
        path._simulate_detector_roi=detector_capture
    def restore():
        for obj,key,value in originals:setattr(obj,key,value)
    return restore
