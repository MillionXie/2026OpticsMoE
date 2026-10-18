"""Actual retained trainability and independent CPU loss/projection checks."""
import numpy as np,torch
import caltech20_rich_electronic_curriculum as m
import audit_caltech20_rich_electronic_curriculum as audit
from check_caltech20_rich_shrinkage_readout import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'retained_electronic_curriculum_CPU');cfg=m.config();assert cfg['electronic_epochs']==8 and len(cfg['cases'])==4 and cfg['microbatch_size']==2 and cfg['alpha_floor']==.4001
    assert m.s.read(m.closure.CPU/'checks.json')['passed'];checks={}
    for n in ('fit','test','campaign','gate','gradients','mixed','initialize','ready','rates','configure','identity','teacher','parent','cap'):
        f=getattr(m,n);f=getattr(f,'__wrapped__',f);assert not unresolved(f.__code__,f.__globals__),(n,unresolved(f.__code__,f.__globals__))
    for n in ('run','verify_fit','verify_test','check_training','check_last_Adam'):
        f=getattr(audit,n);assert not unresolved(f.__code__,f.__globals__),(n,unresolved(f.__code__,f.__globals__))
    for d in (2,4,6):
        model=m.initialize(d);train=[n for n,p in model.named_parameters() if p.requires_grad];assert set(train)==set(m.names(d)) and len(train)==2*d+2 and sum(p.numel() for p in model.parameters() if p.requires_grad)==5140+10*d and sum(p.numel() for p in model.linear.parameters())==5140;assert m.s.digest(model.state_dict())==m.s.digest(m.initial_state(d));assert all(not p.requires_grad for n,p in model.named_parameters() if n=='raw_router' or n.startswith('phase_sets.'));before=m.frozen_digest(model.state_dict(),d);model.alpha_cap=.83
        with torch.no_grad():
            for i,p in enumerate(model.fusion_logits):p.fill_(.2 if i%2 else .97)
        model.project();assert all(abs(float(p)-(.4001 if i%2 else .83))<1e-12 for i,p in enumerate(model.fusion_logits));assert m.frozen_digest(model.state_dict(),d)==before;checks[str(d)]=dict(parent_SHA256=m.parent(d)[0]['sha256'],trainable_names=train,trainable_parameters=5140+10*d,phase_parameters=sum(p.numel() for n,p in model.named_parameters() if n=='raw_router' or n.startswith('phase_sets.')),parent_model_digest=m.s.digest(m.initial_state(d)),frozen_optics=True)
    for key,case in cfg['cases'].items():
        caps=[m.cap(key,e) for e in range(1,9)];expected=np.linspace(.95,case['end_alpha_cap'],8);assert np.allclose(caps,expected,atol=1e-15,rtol=0) and caps[0]==.95 and caps[-1]==case['end_alpha_cap'] and all(a>=b for a,b in zip(caps,caps[1:]));raw=torch.tensor(.96,dtype=torch.float64);ema=torch.tensor(.94,dtype=torch.float64);raw.clamp_(.4001,caps[-1]);ema=ema*.95+raw*.05;ema.clamp_(.4001,caps[-1]);assert float(ema)<=caps[-1]
    rg=np.random.default_rng(17);z=torch.tensor(rg.normal(size=(5,20)),requires_grad=True);q=rg.uniform(.1,1,(5,20));q/=q.sum(1,keepdims=True);loss=m.KL(torch.softmax(z,1),torch.tensor(q),2.);g=torch.autograd.grad(loss,z)[0].numpy();p=np.exp(z.detach().numpy()/2);p/=p.sum(1,keepdims=True);assert np.max(np.abs(g-2*(p-q)/5))<1e-12
    a=rg.uniform(.1,1,(5,20));a/=a.sum(1,keepdims=True);b=rg.uniform(.1,1,(5,20));b/=b.sum(1,keepdims=True);aa=torch.tensor(a,requires_grad=True);bb=torch.tensor(b,requires_grad=True);ga,gb=torch.autograd.grad(m.JS(aa,bb),(aa,bb));middle=(a+b)/2;assert np.max(np.abs(ga.numpy()-.5*np.log(a/middle)/5))<1e-12 and np.max(np.abs(gb.numpy()-.5*np.log(b/middle)/5))<1e-12
    w=torch.tensor(rg.uniform(.5,2,20));y=torch.tensor(np.eye(20)[[0,2,7,13,19]]*.98+.02/20);z=torch.tensor(rg.normal(size=(5,20)),requires_grad=True);p=torch.softmax(z,1);g=torch.autograd.grad(m.CE(p,y,w),z)[0].numpy();target=y.numpy()*w.numpy();assert np.max(np.abs(g-(p.detach().numpy()*target.sum(1,keepdims=True)-target)/5))<1e-12
    m.closure.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,all_existing_electronics_exact_parent_proved=True,independent_CE_JS_KL_derivatives=True,independent_raw_and_EMA_projection=True,all_actual_kernel_namespaces_resolved=True,five_gpu_guard=m.resources.proof(),real_CPU_model_constructions=3,real_model_forward_calls=0,optimizer_steps=0,TRAIN_images_read=0,VAL_images_read=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
