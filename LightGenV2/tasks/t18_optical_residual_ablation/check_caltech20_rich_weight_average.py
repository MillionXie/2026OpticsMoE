"""Verified parent bank and independent weighted-state arithmetic, no forward."""
import numpy as np,torch
import caltech20_rich_weight_average as m
import audit_caltech20_rich_weight_average as audit
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'weight_average_CPU');checks={}
    for name in ('fit','test','bank','state','initialize','ready','score','identity'):
        f=getattr(m,name);f=getattr(f,'__wrapped__',f);assert not unresolved(f.__code__,f.__globals__)
    for name in ('run','verify'):
        f=getattr(audit,name);assert not unresolved(f.__code__,f.__globals__)
    rec=m.s.read(m.previous.OUT/'test_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(m.previous.OUT/'test_science/audit.json');m.identity()
    for d in (2,4,6):
        states,parents=m.bank(d);assert len(states)==3;assert all(torch.equal(v,m.state(d,0)[n]) for n,v in states[0].items());maximum=0.;rms={}
        for fraction in m.config()['mixture_fractions']:
            result=m.state(d,fraction)
            for n,v in result.items():
                if v.is_floating_point():
                    expected=(1-fraction)*states[0][n].numpy().astype(float)+fraction*(states[1][n].numpy().astype(float)+states[2][n].numpy().astype(float))/2;maximum=max(maximum,float(np.max(np.abs(v.numpy().astype(float)-expected))));assert np.allclose(v.numpy(),expected,atol=2e-7,rtol=1e-7)
                else:assert all(torch.equal(v,z[n]) for z in states)
        for j in (1,2):rms[str(j)]=float(np.sqrt(sum(float((states[j][n].double()-v.double()).square().sum()) for n,v in states[0].items() if n=='raw_router' or n.startswith('phase_sets.'))/sum(v.numel() for n,v in states[0].items() if n=='raw_router' or n.startswith('phase_sets.'))))
        assert sum(v.numel() for n,v in states[0].items() if n.startswith('linear.'))==5140;checks[str(d)]=dict(parents=parents,independent_average_maxabs=maximum,phase_endpoint_RMS_displacement=rms,parameter_shapes_unchanged=True)
    m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,real_model_constructions=0,optical_forward_calls=0,optimizer_steps=0,TEST_images_read=0,VALID_images_read=0,TRAIN_images_read=0,five_gpu_guard=m.resources.proof()),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
