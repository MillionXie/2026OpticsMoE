"""AugMix-inspired bounded appearance chains; optional small erasing, TRAIN only."""
import types,json,hashlib
import caltech20_rich_currentKD46 as baseline
import caltech20_rich_loss_gradient_diagnosis46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original','old'):globals()[n]=v
original=baseline.original
FILES=('caltech20_rich_appearance_mix46.py','caltech20_rich_appearance_mix46_profile.json','check_caltech20_rich_appearance_mix46.py','audit_caltech20_rich_appearance_mix46.py','CALTECH20_RICH_APPEARANCE_MIX46_PROTOCOL_20261018.md')
def config():
    c=baseline.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_appearance_mix46_CPU_20261018';ACTIVE_CASE=None;LAST_PLAN=None
def sources():
    z=source_closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel','select_kernel','teacher_targets','gradient_record_kernel','ready','fit'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
def select(key):
    global ACTIVE_DEPTH,ACTIVE_CASE
    ACTIVE_CASE=key;ACTIVE_DEPTH=config()['cases'][key]['depth'];select_kernel(key)
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_clean_teacher_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
def appearance_plan(n,epoch,batch,erasing):
    rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,479]));rows=[]
    for i in range(n):
        chains=[]
        for branch in range(2):
            chain=[]
            for j in range(int(rng.integers(1,3))):
                op=int(rng.integers(0,4));value=float(rng.uniform(.7,1.4) if op==0 else rng.uniform(.5,1.5) if op==1 else rng.uniform(.25,.75) if op==2 else rng.uniform(0,1.75));chain.append([op,value])
            chains.append(chain)
        weights=rng.dirichlet(np.ones(2)).tolist();mix=float(rng.uniform(.15,.5));erase=bool(rng.random()<.5);area=float(rng.uniform(.02,.1));ratio=float(rng.uniform(.5,2));w=min(150,max(1,int(round(150*np.sqrt(area*ratio)))));h=min(150,max(1,int(round(150*np.sqrt(area/ratio)))));x=int(rng.integers(0,151-w));y=int(rng.integers(0,151-h));rows.append(dict(chains=chains,weights=weights,mix=mix,erase=bool(erasing and erase),box=[y,y+h,x,x+w]))
    return rows
def plan_hash(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def transform(image,plan):
    assert image.dtype==torch.uint8 and image.shape==(len(plan),3,150,150);clean=image.float()/255;result=[]
    for x,row in zip(clean,plan):
        branches=[]
        for chain in row['chains']:
            z=x.clone()
            for op,value in chain:
                if op==0:z=z.clamp(0,1).pow(value)
                elif op==1:
                    gray=z.mean(0,keepdim=True);z=gray+value*(z-gray)
                elif op==2:
                    lo=z.amin((-2,-1),keepdim=True);span=z.amax((-2,-1),keepdim=True)-lo;stretch=torch.where(span>1e-6,(z-lo)/span.clamp_min(1e-6),z);z=(1-value)*z+value*stretch
                else:
                    blur=torch.nn.functional.avg_pool2d(torch.nn.functional.pad(z[None],(1,1,1,1),mode='reflect'),3,stride=1)[0];z=blur+value*(z-blur)
                z=z.clamp(0,1)
            branches.append(z)
        z=(1-row['mix'])*x+row['mix']*sum(w*v for w,v in zip(row['weights'],branches))
        if row['erase']:
            y0,y1,x0,x1=row['box'];z=z.clone();z[:,y0:y1,x0:x1]=x.mean((-2,-1),keepdim=True)
        result.append((z.clamp(0,1)*255).round().byte())
    return torch.stack(result)
class AugmentedBatchOnly:
    def __init__(self,images,idx,plan):self.source=images;self.idx=np.asarray(idx);self.calls=0;self.shape=images.shape;self.augmented=transform(images[idx],plan)
    def __getitem__(self,idx):
        assert np.array_equal(np.asarray(idx),self.idx);self.calls+=1;assert self.calls in (1,2);return self.augmented if self.calls==1 else self.source[idx]
f=baseline.mixed;mixed_kernel=types.FunctionType(f.__code__,globals(),'existing_geometry_clean_teacher_MixUp',f.__defaults__,f.__closure__)
def mixed(data,idx,theta,epoch,batch,cfg,device):
    global LAST_PLAN
    assert ACTIVE_CASE is not None;plan=appearance_plan(len(idx),epoch,batch,cfg['cases'][ACTIVE_CASE]['erasing']);proxy=AugmentedBatchOnly(data[0],idx,plan);result=mixed_kernel((proxy,*data[1:]),idx,theta,epoch,batch,cfg,device);assert proxy.calls==2;LAST_PLAN=plan;return result
f=baseline.gradients;record_kernel=types.FunctionType(f.__code__,globals(),'same_calibrated_gradients',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=record_kernel(model,x,y,q,mass,mask,weights,cfg,micro);assert LAST_PLAN is not None;model.view_info.update(appearance_plan_SHA256=plan_hash(LAST_PLAN),appearance_erasing=cfg['cases'][ACTIVE_CASE]['erasing'],appearance_augmented_view_only=True);return result
ready_kernel=ready
def ready():
    p=s.read(CPU/'checks.json');assert p['independent_numpy_appearance_pixels'] and p['clean_view_exactly_unchanged'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
