"""Independent saved science plus every TRAIN-count-aware resize/pad plan."""
import types,numpy as np,json,hashlib
import caltech20_rich_rare_zoomout46 as m
import audit_caltech20_rich_currentKD46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
original_run=run
def run(key):
    original_run(key);cfg=m.config();labels=m.incumbent(cfg['cases'][key]['depth'])[2]['train']['labels'];counts=np.bincount(labels,minlength=20);trace=m.s.read(m.OUT/key/'method_trace.json')
    for row in trace:
        order,_=m.engine.epoch_plan(row['epoch'],3141,cfg);idx=order[row['batch']*16:row['batch']*16+row['n']];rng=np.random.default_rng(np.random.SeedSequence([17,row['epoch'],row['batch'],20261018,483]));plan=[]
        for i in idx:
            count=int(counts[labels[i]]);prob=cfg['zoomout_probability_rare'] if count<=cfg['rare_TRAIN_count_threshold'] else cfg['zoomout_probability_common'];active=bool(rng.random()<prob);side=int(round(150*rng.uniform(*cfg['zoomout_scale'])));margin=150-side;lo=int(np.ceil(margin*.25));hi=int(np.floor(margin*.75));y=int(rng.integers(lo,hi+1));x=int(rng.integers(lo,hi+1));plan.append(dict(active=active,TRAIN_class_support=count,side=side,y=y,x=x))
        sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert row['view_info']['zoomout_plan_SHA256']==sha and row['view_info']['zoomout_augmented_view_only']
    p=m.OUT/key/'science/audit.json';proof=m.s.read(p);proof.update(independent_all_TRAIN_zoomout_plans=True,TRAIN_support_only_adaptation=True,clean_teacher_unchanged=True);m.s.save(p,proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
