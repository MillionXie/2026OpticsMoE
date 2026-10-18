"""Independent original scientific chain plus every same-class donor/box plan."""
import types,numpy as np,json,hashlib
import caltech20_rich_sameclass_patch46 as m
import audit_caltech20_rich_currentKD46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
original_run=run
def run(key):
    original_run(key);cfg=m.config();d=cfg['cases'][key]['depth'];labels=m.incumbent(d)[2]['train']['labels'];trace=m.s.read(m.OUT/key/'method_trace.json')
    for row in trace:
        order,_=m.engine.epoch_plan(row['epoch'],3141,cfg);idx=order[row['batch']*16:row['batch']*16+row['n']];rng=np.random.default_rng(np.random.SeedSequence([17,row['epoch'],row['batch'],20261018,468]));active=bool(rng.random()<cfg['sameclass_patch_probability']);donors=[];boxes=[]
        for i in idx:
            pool=np.flatnonzero(labels==labels[i]);pool=pool[pool!=i];j=int(rng.choice(pool));area=float(rng.uniform(cfg['sameclass_patch_area_min'],cfg['sameclass_patch_area_max']));ratio=float(rng.uniform(.9,1.1));w=min(150,max(1,int(round(150*np.sqrt(area*ratio)))));h=min(150,max(1,int(round(150*np.sqrt(area/ratio)))));x=int(rng.integers(0,151-w));y=int(rng.integers(0,151-h));assert labels[j]==labels[i] and j!=i;donors.append(j);boxes.append([y,y+h,x,x+w])
        plan=dict(active=active,donors=donors,boxes=boxes);sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest();v=row['view_info'];assert v['sameclass_patch_SHA256']==sha and v['sameclass_patch_active']==active and v['sameclass_patch_augmented_view_only']
    p=m.OUT/key/'science/audit.json';proof=m.s.read(p);proof.update(independent_all_TRAIN_sameclass_donor_box_plans=True,patch_augmented_view_only=True,original_clean_teacher_targets_unchanged=True);m.s.save(p,proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
