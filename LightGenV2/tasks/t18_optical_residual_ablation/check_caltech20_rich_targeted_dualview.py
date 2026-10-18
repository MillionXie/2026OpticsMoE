"""Actual namespaces, physical-alpha projection, JS derivative and warm states."""
import numpy as np
import torch
import caltech20_rich_targeted_dualview as m
import audit_caltech20_rich_targeted_dualview as audit
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'dualview_RMS_CPU')
    for name in ('initialize','gate','numeric_gate_kernel','fit','campaign','ready','identity','mixed','gradients','incumbent','rates'):
        f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    assert not unresolved(audit.run.__code__,audit.run.__globals__) and not unresolved(audit.kernel.__code__,audit.kernel.__globals__)
    rng=np.random.default_rng(17);a=rng.uniform(.01,1,(4,20));a/=a.sum(1,keepdims=True);b=rng.uniform(.01,1,(4,20));b/=b.sum(1,keepdims=True)
    p=torch.tensor(a,requires_grad=True);q=torch.tensor(b,requires_grad=True);js=m.js_loss(p,q);gp,gq=torch.autograd.grad(js,(p,q));mid=(a+b)/2;expected=.5*np.mean(np.sum(a*np.log(a/mid)+b*np.log(b/mid),axis=1));npgrad=.5*np.log(a/mid)/len(a);nqgrad=.5*np.log(b/mid)/len(a)
    assert abs(float(js)-expected)<1e-12 and np.max(np.abs(gp.numpy()-npgrad))<1e-12 and np.max(np.abs(gq.numpy()-nqgrad))<1e-12
    checks={}
    for key,case in m.config()['cases'].items():
        m.select(key);model=m.initialize(case['depth'],False);parent,pt=m.diagnosis.parents.parent(case['depth']);expected=m.remap(pt['model'],parent['alpha_low']);assert m.s.digest(model.state_dict())==m.s.digest(expected)
        assert sum(p.numel() for p in model.linear.parameters())==5140 and len(list(model.parameters()))=={2:12,4:21,6:30}[case['depth']]
        with torch.no_grad():
            for p in model.fusion_logits:p.fill_(.40001)
        model.project();assert all(a>.4001 for a in model.alphas());model.load_state_dict(expected)
        checks[key]=dict(initial_digest=m.s.digest(expected),initial_alpha=model.alphas(),gradient_policy=model.full_rms_gradient,parent_sha256=parent['sha256'])
    m.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,five_gpu_guard=m.resources.proof(),all_actual_kernel_namespaces_resolved=True,JS_independent_derivative=True,parents=checks,real_model_forward_calls=0,real_optimizer_steps=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
