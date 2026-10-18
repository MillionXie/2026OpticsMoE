"""Actual unchanged teacher/phase proof plus independent synthetic patch semantics."""
import types,numpy as np,torch
import caltech20_rich_sameclass_patch46 as m
import check_caltech20_rich_currentKD46 as previous
import audit_caltech20_rich_sameclass_patch46 as audit
for name in ('independent_teacher','original_run','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited=run
def run():
    inherited();cfg=m.config();labels=np.repeat(np.arange(20),2);idx=np.arange(0,40,2);images=torch.arange(40*3*150*150,dtype=torch.int64).remainder(251).byte().reshape(40,3,150,150);before=images.clone();p=m.patch_plan(labels,idx,1,0,cfg);p['active']=True;proxy=m.AugmentedBatchOnly(images,idx,p);aug=proxy[idx];clean=proxy[idx];assert proxy.calls==2 and torch.equal(clean,images[idx]) and torch.equal(images,before) and not torch.equal(aug,clean)
    for k,(j,b) in enumerate(zip(p['donors'],p['boxes'])):
        y0,y1,x0,x1=b;assert labels[j]==labels[idx[k]] and j!=idx[k];mask=torch.ones((150,150),dtype=torch.bool);mask[y0:y1,x0:x1]=False;assert torch.equal(aug[k,:,y0:y1,x0:x1],images[j,:,y0:y1,x0:x1]) and torch.equal(aug[k,:,mask],images[idx[k],:,mask]) and .08<((y1-y0)*(x1-x0)/22500)<.28
    proof=m.s.read(m.CPU/'checks.json');proof.update(sameclass_patch_independent_synthetic_proof=True,source_images_not_mutated=True,clean_view_exactly_unchanged=True,TRAIN_only_same_class_donors=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
