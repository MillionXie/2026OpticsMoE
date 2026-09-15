import torch
from .core import *

def merge():
    authorize();split=prepare();out=ROOT/'runs/merge_seed42'
    shared=load_checkpoint(out/'shared/selected.pt',split);a=load_checkpoint(out/'expert_A/selected.pt',split);b=load_checkpoint(out/'expert_B/selected.pt',split)
    merged=merge_states(shared,a,b);loaded,s=setup(plan()['seed'],out);r,h=build(loaded,s)
    target=[Images(split,d,'validation')[i][0] for d in (0,1) for i in range(10)]
    inputs=E.base._prepare(loaded,target,s);checks=[]
    for d,state in ((0,a),(1,b)):
        M.restore(r,h,state)
        with torch.no_grad(),E.autocast(loaded,s):one=forward(loaded,r,h,inputs,'isolated',d).float()
        M.restore(r,h,merged)
        with torch.no_grad(),E.autocast(loaded,s):two=forward(loaded,r,h,inputs,'isolated',d).float()
        torch.testing.assert_close(one,two,atol=0,rtol=0)
        checks.append(dict(domain=d,isolated_logits_bitwise_equal=True))
    M.restore(r,h,merged);isolated=evaluate(loaded,r,h,s,split,'isolated');uniform=evaluate(loaded,r,h,s,split,'uniform')
    merged.update(isolated_validation=isolated,uniform_validation=uniform)
    E.save(out/'merged.pt',merged)
    atomic_json(out/'merge_checks.json',dict(status='passed',checks=checks,shared_sha256=merged['shared_sha256'],copied_experts=[0,1,2,3],isolated_validation=isolated,uniform_validation=uniform))
    r.close()

if __name__=='__main__':merge()
