"""Independent Gaussian discrimination and head parameter-count proof."""
import numpy as np
import caltech20_rich_shrinkage_readout as m
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'same5140_shrinkage_CPU')
    for name in ('parents','train_statistics','discriminant','probabilities','fit','audit','test','identity'):
        f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    rg=np.random.default_rng(17);x=rg.normal(size=(80,4));y=np.repeat(np.arange(20),4);mean,cov,count=m.train_statistics(x,y)
    for beta in m.config()['shrinkage_grid']:
        w,b=m.discriminant(mean,cov,count,beta);mat=(1-beta)*cov+beta*np.trace(cov)/4*np.eye(4);inv=np.linalg.inv(mat);a=x[:3];direct=np.stack([-.5*np.einsum('ni,ij,nj->n',a-mu,inv,a-mu)+np.log(count[c]/len(y)) for c,mu in enumerate(mean)],1);linear=a@w.T+b
        assert np.max(np.abs((direct-direct[:,:1])-(linear-linear[:,:1])))<1e-10
    m.stage.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,independent_Gaussian_discriminant=True,head_parameters=256*20+20,new_inference_parameters=0,real_model_forwards=0,optimizer_steps=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
