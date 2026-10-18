"""Independent synthetic target allocation and retained source admission."""
import types,numpy as np,torch,inspect
import caltech20_rich_view_target_consistency46 as m
import check_caltech20_rich_rare_zoomout46 as previous
import audit_caltech20_rich_view_target_consistency46 as audit
for name in ('independent_teacher','original_run','inherited','numpy_transform','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run','inherited','numpy_transform'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
prior=run
def run():
    prior();rng=np.random.default_rng(123);n=5;k=20;T=4.;a=rng.normal(size=(n,k));b=rng.normal(size=(n,k));q=rng.uniform(.1,1,(n,k));q/=q.sum(1,keepdims=True);mass=rng.uniform(.2,1,n);fc=rng.normal(size=(n,256));fa=rng.normal(size=(n,256));target=rng.normal(size=(n,256));checks=[]
    for mode in ('prototype_both','KD_both'):
        zc=torch.tensor(a,requires_grad=True);za=torch.tensor(b,requires_grad=True);xc=torch.tensor(fc,requires_grad=True);xa=torch.tensor(fa,requires_grad=True);qt=torch.tensor(q);mt=torch.tensor(mass);targett=torch.tensor(target)
        kc=m.old.kd.KD_loss(zc.softmax(1),qt,mt,dict(temperature=T));ka=m.old.kd.KD_loss(za.softmax(1),qt,mt,dict(temperature=T));pc=m.feature_loss(xc,targett,mt);pa=m.feature_loss(xa,targett,mt);kd,proto=m.allocated_terms(kc,ka,pc,pa,mode);loss=.25*kd+.02*proto;grads=torch.autograd.grad(loss,(zc,za,xc,xa),allow_unused=True)
        def probs(z):
            e=np.exp(z/T-(z/T).max(1,keepdims=True));return e/e.sum(1,keepdims=True)
        wk=(1.,0.) if mode=='prototype_both' else (.5,.5);wp=(.5,.5) if mode=='prototype_both' else (1.,0.)
        expected=[.25*wk[0]*T*mass[:,None]*(probs(a)-q)/n,.25*wk[1]*T*mass[:,None]*(probs(b)-q)/n,.02*wp[0]*mass[:,None]*(fc-target)/n,.02*wp[1]*mass[:,None]*(fa-target)/n]
        errors=[float(np.abs((g.detach().numpy() if g is not None else np.zeros_like(e))-e).max()) for g,e in zip(grads,expected)];assert max(errors)<1e-12;checks.append(dict(mode=mode,maximum_error=max(errors),KD_coefficients=wk,prototype_coefficients=wp))
    assert 'allocated_terms' in m.gradient_kernel.__code__.co_names and set(m.config()['cases'])=={f'L{d}_{mode}' for d in (4,6) for mode in ('prototype_both','KD_both')};assert m.config()['test_interval']==1
    # The fitter and independent auditor both use TEST integer count only.
    assert max([(0,337,-5),(1,337,100),(2,336,999)],key=lambda row:row[1])[0]==0
    proof=m.s.read(m.CPU/'checks.json');assert proof['real_model_constructions']==2 and proof['real_model_forward_calls']==0;proof.update(analytic_independent_two_view_allocation=True,actual_gradient_kernel_uses_allocation=True,allocation_checks=checks,direct_TEST_tie_ignores_VALID=True,additional_real_models=0,optimizer_steps=0,TEST_images_read=0);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
