"""One physical amplitude contract, applied before phase modulation and BMP export."""
import torch

def save_bmps(amplitudes,out,stage,ids,flow):
    import numpy as np
    from PIL import Image
    values=np.asarray(amplitudes,dtype=np.float32)
    if not np.isfinite(values).all() or values.min()<0 or values.max()>1.000001:
        raise ValueError('Bounded physical amplitude must already be within [0,1]')
    folder=out/'amplitude'/stage;folder.mkdir(parents=True,exist_ok=True);paths=[]
    for sid,value in zip(ids,values):
        path=folder/(sid+'.bmp')
        gray=np.rint(value.clip(0,1)*255).astype(np.uint8)
        Image.fromarray(flow.active_to_native(gray)).save(path);paths.append(path)
    return paths,1.0

def encode(value,spec):
    scale=float(spec['scale'])
    if scale<=0:raise ValueError('Amplitude scale must be positive')
    amplitude=value.abs()
    if spec['kind']=='rational':return value/(amplitude+scale)
    if spec['kind']=='tanh':
        factor=torch.tanh(amplitude/scale)/amplitude.clamp_min(1e-8)
        return value*factor
    raise ValueError('Unknown bounded amplitude mapping')

def install(model,spec):
    if getattr(model,'bounded_amplitude',None):raise ValueError('Mapping already installed')
    spec=dict(spec);model.bounded_amplitude=spec
    for path in (model.text.optical,model.editor.bottleneck.optical):
        original=path._simulate_detector_roi
        def detector(field,modulation,shifts,*,phase_support=None,original=original):
            return original(encode(field,spec),modulation,shifts,phase_support=phase_support)
        path._simulate_detector_roi=detector
        router=path.core.router;original_router=router._simulate
        def route(fields,original=original_router):return original(encode(fields,spec))
        router._simulate=route
    return model
