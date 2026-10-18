"""Fresh determinism, full RMS derivative, fixed TRAIN centers, no real forward."""
import numpy as np,torch
import caltech20_rich_fresh_fullRMS as m
import audit_caltech20_rich_fresh_fullRMS as audit
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();torch.set_num_threads(4);m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'fresh_fullRMS_CPU');cfg=m.config();assert cfg['alpha_restart']==.45 and cfg['phase_anchor_weight']==1e-5 and all(c['fresh'] and c['epochs']==50 for c in cfg['cases'].values())
    for name in ('fit','gradients','mixed','numeric_gate_kernel','gate','initialize','ready','identity','incumbent','prototype_data','target_data','campaign'):
        f=getattr(m,name);f=getattr(f,'__wrapped__',f);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    for name in ('run','kernel'):
        f=getattr(audit,name);assert not unresolved(f.__code__,f.__globals__)
    a=torch.tensor([[.2,.5,.8],[.7,.1,.4]],dtype=torch.float64,requires_grad=True);target=torch.tensor([[.3,.6,.1],[.1,.2,.5]],dtype=torch.float64);r=(a.square().mean(1,keepdim=True)+1e-8).sqrt();value=((a/r)*target).sum();g=torch.autograd.grad(value,a)[0];rg=a.detach().numpy();tt=target.numpy();rr=np.sqrt(np.mean(rg*rg,axis=1,keepdims=True)+1e-8);expected=tt/rr-rg*np.mean(rg*tt,axis=1,keepdims=True)/rr**3;assert np.max(np.abs(g.numpy()-expected))<1e-12
    p=torch.tensor([[.2,.3,.5],[.4,.4,.2]],dtype=torch.float64,requires_grad=True);q=torch.tensor([[.3,.3,.4],[.2,.3,.5]],dtype=torch.float64);j=m.js_loss(p,q);gj=torch.autograd.grad(j,p)[0];assert torch.max(torch.abs(gj-.5*torch.log(p.detach()/((p.detach()+q)/2))/2))<1e-12
    x=torch.tensor([[.2,.3,.4],[.8,.1,.5]],dtype=torch.float64,requires_grad=True);t=torch.zeros_like(x);mass=torch.tensor([1.,.7],dtype=torch.float64);v=m.feature_loss(x,t,mass);gx=torch.autograd.grad(v,x)[0];assert torch.max(torch.abs(gx-mass[:,None]*x.detach()/2))<1e-12;checks={}
    for d in (2,4,6):
        m.select('L'+str(d)+'_fresh_fullRMS');model=m.initialize(d,True);first=m.s.digest(model.state_dict());other=m.initialize(d,True);assert first==m.s.digest(other.state_dict()) and all(abs(v-.45)<1e-6 for v in model.alphas()) and len(list(model.parameters()))=={2:12,4:21,6:30}[d] and sum(p.numel() for p in model.linear.parameters())==5140 and model.full_rms_gradient;parent,pt,_=m.incumbent(d);assert first!=m.s.digest(m.remap(pt['model'],parent['alpha_low']));f,scale,sha,info=m.prototype_data(d);assert f.shape==(3141,256) and np.isfinite(f).all()
        with np.load(m.alignment.OUT/f'L{d}_train_targets.npz',allow_pickle=False) as z:teacher,labels=z['features'],z['labels']
        centers=[]
        for c in range(20):v=teacher[labels==c].astype(float).mean(0);centers.append((v-v.mean())/np.sqrt(v.var()+1e-5))
        assert np.max(np.abs(f-np.asarray(centers)[labels]))<3e-6;checks[str(d)]=dict(initial_digest=first,initial_alpha=model.alphas(),canonical_parent_SHA256=parent['sha256'],prototype_target_SHA256=sha,prototype_scale=scale)
    m.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,fresh_states=checks,fresh_initial_states_reproduced=True,independent_RMS_JS_prototype_derivatives=True,five_gpu_guard=m.resources.proof(),real_model_constructions=6,real_model_forward_calls=0,optimizer_steps=0,TEST_images_read=0,TRAIN_images_read=0,VAL_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
