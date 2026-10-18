"""Independent source/parent/center arithmetic and consistency derivative."""
import inspect,types,numpy as np,torch
import caltech20_rich_prototype_consistency as m
import check_caltech20_rich_targeted_dualview as previous
import audit_caltech20_rich_prototype_consistency as audit
def run():
    text=inspect.getsource(previous.run);site="parent,pt=m.diagnosis.parents.parent(case['depth'])";assert text.count(site)==1;text=text.replace(site,"parent,pt,_=m.incumbent(case['depth'])");site='f=getattr(m,name);assert';assert text.count(site)==1;text=text.replace(site,"f=getattr(m,name);f=getattr(f,'__wrapped__',f);assert");ns=dict(vars(previous));ns.update(m=m,audit=audit);exec(compile(text,__file__+':same_math_actual_selected_parents','exec'),ns);ns['run']()
    x=torch.tensor([[.3,.2,.7],[.4,.8,.1]],dtype=torch.float64,requires_grad=True);t=torch.tensor([[.1,.6,.2],[.2,.7,.3]],dtype=torch.float64);mass=torch.tensor([1.,.7],dtype=torch.float64);loss=m.feature_loss(x,t,mass);gradient=torch.autograd.grad(loss,x)[0];assert torch.max(torch.abs(gradient-mass[:,None]*(x.detach()-t)/2))<1e-12
    checks={}
    for d in (2,4,6):
        features,scale,sha,info=m.prototype_data(d);assert features.shape==(3141,256) and np.isfinite(features).all()
        with np.load(m.alignment.OUT/f'L{d}_train_targets.npz',allow_pickle=False) as z:a,y=z['features'],z['labels']
        centers=[]
        for c in range(20):v=a[y==c].astype(np.float64).mean(0);centers.append((v-v.mean())/np.sqrt(v.var()+1e-5))
        assert np.max(np.abs(features-np.asarray(centers)[y]))<3e-6;checks[str(d)]=dict(info,target_SHA256=sha,parent_SHA256=m.incumbent(d)[0]['sha256'])
    q=m.s.read(m.CPU/'checks.json');q.update(prototype_derivative_passed=True,TRAIN_class_centers=checks,new_teacher_forwards=0,new_optical_forwards=0,new_TEST_calls=0);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
