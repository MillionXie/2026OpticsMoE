"""Original independent teacher/phase admission plus ordinary CE derivative."""
import types,numpy as np,torch
import caltech20_rich_overallCE46 as m
import check_caltech20_rich_currentKD46 as previous
import audit_caltech20_rich_overallCE46 as audit
for name in ('independent_teacher','original_run','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
original_run=run
def run():
    original_run();cfg=m.config();assert cfg['class_weight_exponent']==0 and cfg['KD_weight']==.25;w=m.old.kd.class_weights(cfg,'cpu');assert torch.equal(w,torch.ones(20));rng=np.random.default_rng(17);z=torch.tensor(rng.normal(size=(7,20)),requires_grad=True);y=torch.tensor(np.eye(20)[[0,1,4,8,12,16,19]]*.95+.05/20);p=torch.softmax(z,1);loss=-(y*p.log()).sum(1).mean();g=torch.autograd.grad(loss,z)[0];assert float((g-(p.detach()-y)/7).abs().max())<1e-12;proof=m.s.read(m.CPU/'checks.json');proof.update(independent_ordinary_CE_derivative=True,all_real_class_weights_exactly_one=True,macro_and_class_tradeoffs_must_be_reported=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
