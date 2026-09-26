"""Training-only alignment perturbations; eval and measured-CCD paths unchanged."""
import copy
import torch


def shift_zero(value, maximum=1):
    """Per-sample integer shift, zero outside aperture (never circular wrap)."""
    result=[]
    for row in value:
        dy,dx=torch.randint(-maximum,maximum+1,(2,),device=value.device).tolist()
        moved=torch.roll(row,(dy,dx),(-2,-1))
        if dy>0:moved=moved.clone();moved[:dy,:]=0
        if dy<0:moved=moved.clone();moved[dy:,:]=0
        if dx>0:moved=moved.clone();moved[:,:dx]=0
        if dx<0:moved=moved.clone();moved[:,dx:]=0
        result.append(moved)
    return torch.stack(result)


def prepare(payload, profile):
    if 'robust_alpha_min' not in profile:return payload
    result=copy.deepcopy(payload)
    md=result['metadata'];old_low=md['fusion_alpha_min'];high=md['fusion_alpha_max']
    new_low=profile['robust_alpha_min']
    for name,raw in result['state_dict'].items():
        if name.endswith(('block1_optical_fusion_logit','block2_optical_fusion_logit')):
            actual=old_low+(high-old_low)*raw.sigmoid()
            raw.copy_(torch.logit(((actual-new_low)/(high-new_low)).clamp(1e-6,1-1e-6)))
    md['fusion_alpha_min']=new_low
    md['optical_training_noise']=dict(md['optical_training_noise'],
        dc_min=.20,dc_max=.35,gain_min=.8,gain_max=1.2,
        ccd_mean=.015,ccd_std=.03,ccd_low=-.05,ccd_high=.12,
        router_bypass=.01,phase_bypass=.02,router_logit_std=.05)
    md['alignment_training']=dict(incident_shift_pixels=profile.get('pixel_shift',0),
        ccd_shift_pixels=profile.get('pixel_shift',0),coordinate='17um simulation pixels',
        inference_enabled=False)
    return result


def attach(model, profile):
    maximum=profile.get('pixel_shift',0)
    if not maximum:return
    for modality in (model.vision,model.language):
        path=modality.optics
        fanout=path.fanout
        def jittered_fanout(amplitude,weights,path=path,original=fanout):
            field=original(amplitude,weights)
            return shift_zero(field,maximum) if path.training and path.noise_enabled and not path.measured else field
        path.fanout=jittered_fanout
        for propagator,owner in ((path.propagator,path),(path.router.propagator,path.router)):
            original=propagator.forward
            def jittered_detector(field,original=original,owner=owner):
                detector=original(field)
                if owner.training and owner.noise_enabled:
                    detector=shift_zero(detector,maximum)
                return detector
            propagator.forward=jittered_detector
