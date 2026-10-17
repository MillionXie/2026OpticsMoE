"""CPU admission: exact frozen parents and independent temperature-KD gradient."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech10_equal60_teacherKD as m
def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and m.base.slots()['total']==0;m.claim(out.with_name(out.name+'.claim.json'),'own_KD_CPU');controls=m.admit();cfg=m.config();target=m.teacher_targets();assert target.shape==(761,10) and np.allclose(target.sum(1),1,atol=2e-7)
    rng=np.random.default_rng(17);z=rng.normal(size=(7,10));q=rng.uniform(.01,1,(7,10));q/=q.sum(1,keepdims=True);zt=torch.tensor(z,dtype=torch.float64,requires_grad=True);qt=torch.tensor(q,dtype=torch.float64);T=4.;loss=T*T*(qt*(qt.log()-torch.log_softmax(zt/T,1))).sum(1).mean();g=torch.autograd.grad(loss,zt)[0].numpy();p=np.exp(z/T-(z/T).max(1,keepdims=True));p/=p.sum(1,keepdims=True);error=float(np.abs(g-T*(p-q)/7).max());assert error<1e-12
    parents={}
    for key in m.CASES:
        last=torch.load(m.BASE/key/'last.pt',map_location='cpu',weights_only=False);assert last['epoch']==60 and last['steps']==2880 and last['scheduler']['last_epoch']==60 and all(float(v['step'])==2880 for v in last['optimizer']['state'].values());parents[key]=dict(raw=m.s.digest(last['model']),ema=m.s.digest(last['ema']),moments=m.moments(last['optimizer']))
    out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,analytic_temperature_KD_gradient_maxabs=error,parent_states=parents,frozen_controls=controls,teacher_audit_sha256=cfg['teacher_science_sha256'],model_objects=0,model_forward_calls=0,optimizer_steps=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
