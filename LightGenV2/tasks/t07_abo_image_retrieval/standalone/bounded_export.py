"""Same pre-phase amplitude as optics; no additional BMP peak scaling."""
import numpy as np
import torch
from .optics import bounded_amplitude

SPEC=dict(kind='tanh',scale=.5)
def stage_active(amplitude,weights=None):
    if weights is None:
        a=bounded_amplitude(amplitude,SPEC)
        result=a.new_zeros(len(a),478,478);result[:,127:351,127:351]=a
    else:
        field=amplitude.new_zeros(len(amplitude),518,518)
        for i,(y,x) in enumerate(((20,20),(20,274),(274,20),(274,274))):
            field[:,y:y+224,x:x+224]=amplitude.float()*weights[:,i,None,None].clamp_min(0)
        result=bounded_amplitude(field,SPEC)[:,20:498,20:498]
    return result.detach().float().cpu().numpy()

def quantize(active):
    a=np.asarray(active,dtype=np.float32)
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1+1e-6:raise ValueError('Amplitude outside physical [0,1]')
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)

def test(model):
    results={}
    for mode in ('vision','language'):
        optics=getattr(model,mode).optics
        a=torch.linspace(0,12,224*224).reshape(1,224,224)
        seen=[]
        h=optics.router.propagator.register_forward_pre_hook(lambda module,args:seen.append(args[0].abs()[:,20:498,20:498].detach()))
        with torch.no_grad():weights=optics.router(a)
        h.remove();expected=stage_active(a);assert np.max(np.abs(seen[0].numpy()-expected))<1e-6
        results[mode+'_router']=float(np.max(np.abs(seen[0].numpy()-expected)))
        for final,label in ((False,'expert'),(True,'global')):
            seen=[];h=optics.propagator.register_forward_pre_hook(lambda module,args:seen.append(args[0].abs()[:,20:498,20:498].detach()))
            with torch.no_grad():optics.propagate(optics.fanout(a,weights),final)
            h.remove();expected=stage_active(a,weights);error=float(np.max(np.abs(seen[0].numpy()-expected)));assert error<1e-6
            assert np.max(np.abs(quantize(expected)/255-expected))<=.5/255+1e-6
            results[mode+'_'+label]=error
    assert not quantize(np.zeros((1,478,478),np.float32)).any()
    return results
