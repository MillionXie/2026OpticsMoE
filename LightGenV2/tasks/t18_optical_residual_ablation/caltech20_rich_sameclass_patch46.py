"""TRAIN-only same-class donor patches on augmented view; inference unchanged."""
import json,types,hashlib
import caltech20_rich_currentKD46 as baseline
import caltech20_rich_currentKD_strength46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original','old'):globals()[n]=v
original=baseline.original
FILES=('caltech20_rich_sameclass_patch46.py','caltech20_rich_sameclass_patch46_profile.json','check_caltech20_rich_sameclass_patch46.py','audit_caltech20_rich_sameclass_patch46.py','CALTECH20_RICH_SAMECLASS_PATCH46_PROTOCOL_20261018.md')
def config():
    c=baseline.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_sameclass_patch46_CPU_20261018';LAST_PATCH_PLAN=None
def sources():
    z=source_closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','ready','fit'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_teacher_targets_no_donor_label_change',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
def patch_plan(labels,idx,epoch,batch,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,468]));active=bool(rng.random()<cfg['sameclass_patch_probability']);donors=[];boxes=[]
    for i in idx:
        pool=np.flatnonzero(labels==labels[i]);pool=pool[pool!=i];assert len(pool)>0;j=int(rng.choice(pool));area=float(rng.uniform(cfg['sameclass_patch_area_min'],cfg['sameclass_patch_area_max']));ratio=float(rng.uniform(.9,1.1));w=min(150,max(1,int(round(150*np.sqrt(area*ratio)))));h=min(150,max(1,int(round(150*np.sqrt(area/ratio)))));x=int(rng.integers(0,151-w));y=int(rng.integers(0,151-h));donors.append(j);boxes.append([y,y+h,x,x+w])
    return dict(active=active,donors=donors,boxes=boxes)
def plan_hash(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
class AugmentedBatchOnly:
    def __init__(self,images,idx,plan):
        self.source=images;self.idx=np.asarray(idx);self.plan=plan;self.calls=0;self.shape=images.shape;self.augmented=images[idx].clone();assert images.shape[-2:]==(150,150)
        if plan['active']:
            for k,(j,box) in enumerate(zip(plan['donors'],plan['boxes'])):
                y0,y1,x0,x1=box;self.augmented[k,:,y0:y1,x0:x1]=images[j,:,y0:y1,x0:x1]
    def __getitem__(self,idx):
        assert np.array_equal(np.asarray(idx),self.idx);self.calls+=1;assert self.calls in (1,2);return self.augmented if self.calls==1 else self.source[idx]
f=baseline.mixed;mixed_kernel=types.FunctionType(f.__code__,globals(),'same_original_clean_and_augmented_views',f.__defaults__,f.__closure__)
def mixed(data,idx,theta,epoch,batch,cfg,device):
    global LAST_PATCH_PLAN
    plan=patch_plan(data[1].numpy(),idx,epoch,batch,cfg);proxy=AugmentedBatchOnly(data[0],idx,plan);modified=(proxy,*data[1:]);result=mixed_kernel(modified,idx,theta,epoch,batch,cfg,device);assert proxy.calls==2;LAST_PATCH_PLAN=plan;return result
f=baseline.gradients;record_kernel=types.FunctionType(f.__code__,globals(),'same_calibrated_gradient_recording',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=record_kernel(model,x,y,q,mass,mask,weights,cfg,micro);assert LAST_PATCH_PLAN is not None;model.view_info.update(sameclass_patch_SHA256=plan_hash(LAST_PATCH_PLAN),sameclass_patch_active=LAST_PATCH_PLAN['active'],sameclass_patch_augmented_view_only=True);return result
ready_kernel=ready
def ready():
    proof=s.read(CPU/'checks.json');assert proof['sameclass_patch_independent_synthetic_proof'] and proof['clean_view_exactly_unchanged'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
