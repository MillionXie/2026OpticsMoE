import copy,json
import torch
from torch.nn import functional as F
from .core import *

def verify():
    authorize();split=prepare();loaded,s=setup(plan()['seed'],ROOT/'runs/preflight');r,h=build(loaded,s)
    pairs=[Images(split,d,'validation')[i*len(Images(split,d,'validation'))//10] for d in (0,1) for i in range(10)]
    images,y=zip(*pairs);y=torch.tensor(y,device='cuda');d=torch.arange(20,device='cuda')//10;inputs=E.base._prepare(loaded,list(images),s)
    initial=M.clone(r,h);records=[]
    for stage,epoch,mode in [('shared',1,'uniform'),('shared',5,'isolated'),('expert_A',1,'isolated'),('expert_B',1,'isolated'),('router',1,'automatic')]:
        M.restore(r,h,initial);opt=optimizer(r,h,stage,epoch);frozen=M.digest(r,h,True)
        domains=d if stage=='shared' else (0 if stage=='expert_A' else 1)
        with E.autocast(loaded,s):logits=forward(loaded,r,h,inputs,mode,domains if mode=='isolated' else None);ce=F.cross_entropy(logits,y)
        grads=E.phase_gradients(r,h,ce) if not(stage=='shared' and epoch==1) else {}
        for n,g in grads.items():assert g['finite'] and g['l2']>0,(n,g)
        q=M.routes(r)['vision'];torch.testing.assert_close(q['weights'].square().sum(1),torch.ones(20,device='cuda'),atol=1e-6,rtol=1e-6)
        if mode=='automatic':assert bool((q['weights']>0).all())
        if stage in ('expert_A','expert_B'):
            own=(0,1) if stage=='expert_A' else (2,3);other=(2,3) if stage=='expert_A' else (0,1)
            assert bool((q['weights'][:,list(other)]==0).all())
            torch.testing.assert_close(q['weights'][:,list(own)],torch.full((20,2),2**-.5,device='cuda'))
        ce.backward();opt.step();assert frozen==M.digest(r,h,True)
        records.append(dict(stage=stage,epoch=epoch,mode=mode,frozen_unchanged=True,gradients=grads))
        print('[preflight]',stage,epoch,'passed',flush=True)
    a=copy.deepcopy(initial);b=copy.deepcopy(initial)
    for n,t in a['vision_optical'].items():
        i=expert_number('vision_optical.'+n)
        if i is not None and i<2:t.add_(.07)
    for n,t in b['vision_optical'].items():
        i=expert_number('vision_optical.'+n)
        if i is not None and i>=2:t.sub_(.09)
    merged=merge_states(initial,a,b)
    for domain,state in ((0,a),(1,b)):
        M.restore(r,h,state)
        with torch.no_grad(),E.autocast(loaded,s):before=forward(loaded,r,h,inputs,'isolated',domain).float()
        M.restore(r,h,merged)
        with torch.no_grad(),E.autocast(loaded,s):after=forward(loaded,r,h,inputs,'isolated',domain).float()
        torch.testing.assert_close(before,after,atol=0,rtol=0)
    with torch.no_grad(),E.autocast(loaded,s):
        one=forward(loaded,r,h,inputs).float();two=forward(loaded,r,h,E.base._prepare(loaded,list(reversed(images)),s)).float().flip(0)
    torch.testing.assert_close(one,two,atol=.02,rtol=.002);assert torch.equal(one.argmax(1),two.argmax(1))
    assert r.vision_surrogate.core.optical_branch.core.router.merge_mode=='automatic'
    assert r.vision_surrogate.core.optical_branch.core.router.merge_domains is None
    report=dict(status='passed',checks=records,isolated_merge_bitwise_equal=True,automatic_order_invariant=True,plan_sha256=signature(plan()),split_sha256=split['split_sha256'],test_used=False)
    atomic_json(ROOT/'runs/preflight.json',report);r.close()

if __name__=='__main__':verify()
