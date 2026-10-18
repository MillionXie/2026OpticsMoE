"""TRAIN-count-aware resize-and-pad rather than stronger object cropping."""
import types,json,hashlib
import caltech20_rich_currentKD46 as baseline
import caltech20_rich_appearance_mix46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_rare_zoomout46.py','check_caltech20_rich_rare_zoomout46.py','audit_caltech20_rich_rare_zoomout46.py','CALTECH20_RICH_RARE_ZOOMOUT46_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();case=dict(fresh=False,epochs=5,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True);c.update(run='caltech20_rich20_TRAIN_count_zoomout46_local5_20261018',cases={'L4_rare_zoomout':dict(case,depth=4),'L6_rare_zoomout':dict(case,depth=6)},gpu_order=[3,0,2,1,4],rare_TRAIN_count_threshold=102,zoomout_probability_rare=.75,zoomout_probability_common=.25,zoomout_scale=[.72,.92],augmentation_count_adaptation_uses_TRAIN_only=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_rare_zoomout46_CPU_20261018';LAST_PLAN=None
def sources():
    z=source_closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','ready','fit'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_clean_teacher_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
def zoom_plan(labels,idx,epoch,batch,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,483]));counts=np.bincount(labels,minlength=20);rows=[]
    for i in idx:
        count=int(counts[labels[i]]);p=cfg['zoomout_probability_rare'] if count<=cfg['rare_TRAIN_count_threshold'] else cfg['zoomout_probability_common'];active=bool(rng.random()<p);side=int(round(150*rng.uniform(*cfg['zoomout_scale'])));margin=150-side;lo=int(np.ceil(margin*.25));hi=int(np.floor(margin*.75));y=int(rng.integers(lo,hi+1));x=int(rng.integers(lo,hi+1));rows.append(dict(active=active,TRAIN_class_support=count,side=side,y=y,x=x))
    return rows
def plan_hash(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def transform(images,plan):
    assert images.dtype==torch.uint8 and images.shape==(len(plan),3,150,150);result=[]
    for image,row in zip(images,plan):
        if not row['active']:result.append(image.clone());continue
        z=image.float();border=torch.cat((z[:,0,:],z[:,-1,:],z[:,:,0],z[:,:,-1]),1).mean(1)[:,None,None];out=border.expand_as(z).clone();small=torch.nn.functional.interpolate(z[None],size=(row['side'],row['side']),mode='bilinear',align_corners=False)[0];y,x=row['y'],row['x'];out[:,y:y+row['side'],x:x+row['side']]=small;result.append(out.clamp(0,255).round().byte())
    return torch.stack(result)
class AugmentedBatchOnly:
    def __init__(self,images,idx,plan):self.source=images;self.idx=np.asarray(idx);self.calls=0;self.shape=images.shape;self.augmented=transform(images[idx],plan)
    def __getitem__(self,idx):
        assert np.array_equal(np.asarray(idx),self.idx);self.calls+=1;assert self.calls in (1,2);return self.augmented if self.calls==1 else self.source[idx]
f=baseline.mixed;mixed_kernel=types.FunctionType(f.__code__,globals(),'existing_geometry_and_exact_clean_teacher',f.__defaults__,f.__closure__)
def mixed(data,idx,theta,epoch,batch,cfg,device):
    global LAST_PLAN
    plan=zoom_plan(data[1].numpy(),idx,epoch,batch,cfg);proxy=AugmentedBatchOnly(data[0],idx,plan);result=mixed_kernel((proxy,*data[1:]),idx,theta,epoch,batch,cfg,device);assert proxy.calls==2;LAST_PLAN=plan;return result
f=baseline.gradients;record_kernel=types.FunctionType(f.__code__,globals(),'same_calibrated_gradients',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=record_kernel(model,x,y,q,mass,mask,weights,cfg,micro);model.view_info.update(zoomout_plan_SHA256=plan_hash(LAST_PLAN),zoomout_augmented_view_only=True);return result
ready_kernel=ready
def ready():
    q=s.read(CPU/'checks.json');assert q['independent_numpy_zoomout_pixels'] and q['clean_view_exactly_unchanged'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
