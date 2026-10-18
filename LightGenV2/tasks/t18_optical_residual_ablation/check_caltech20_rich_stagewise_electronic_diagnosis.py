"""Independent RMS fusion math, force specificity and max5 resource proof."""
import numpy as np,torch
import caltech20_rich_stagewise_electronic_diagnosis as m
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'stagewise_electronic_diagnosis_CPU')
    for n in ('run','evaluate','identity','sources','claim'):f=getattr(m,n);assert not unresolved(f.__code__,f.__globals__)
    rg=np.random.default_rng(17);e=rg.uniform(.1,1,(3,5,5));o=rg.uniform(.1,1,(3,5,5));re=np.sqrt((e*e).mean((1,2),keepdims=True));ro=np.sqrt((o*o).mean((1,2),keepdims=True))
    for a in (.75,.85,.94):
        mix=(1-a)*e/re+a*o/ro;expected=re*mix/np.sqrt((mix*mix).mean((1,2),keepdims=True));et=torch.tensor(e);ot=torch.tensor(o);mixed=(1-a)*et/m.rms(et)+a*ot/m.rms(ot);actual=m.rms(et)*mixed/m.rms(mixed);assert np.max(np.abs(actual.numpy()-expected))<1e-12
    proof=m.resources.proof();m.closure.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,independent_normalized_fusion=True,five_gpu_guard=proof,TRAIN_images_read=0,VAL_images_read=0,TEST_images_read=0,real_model_forward_calls=0,optimizer_steps=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
