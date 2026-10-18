"""Independent original science and deterministic per-macro appearance plans."""
import types,numpy as np,json,hashlib
import caltech20_rich_appearance_mix46 as m
import audit_caltech20_rich_currentKD46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
original_run=run
def independent_plan(n,epoch,batch,erasing):
    rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261018,479]));rows=[]
    for i in range(n):
        chains=[]
        for branch in range(2):
            chain=[]
            for j in range(int(rng.integers(1,3))):
                op=int(rng.integers(0,4));value=float(rng.uniform(.7,1.4) if op==0 else rng.uniform(.5,1.5) if op==1 else rng.uniform(.25,.75) if op==2 else rng.uniform(0,1.75));chain.append([op,value])
            chains.append(chain)
        weights=rng.dirichlet(np.ones(2)).tolist();mix=float(rng.uniform(.15,.5));active=bool(rng.random()<.5);area=float(rng.uniform(.02,.1));ratio=float(rng.uniform(.5,2));w=min(150,max(1,int(round(150*np.sqrt(area*ratio)))));h=min(150,max(1,int(round(150*np.sqrt(area/ratio)))));x=int(rng.integers(0,151-w));y=int(rng.integers(0,151-h));rows.append(dict(chains=chains,weights=weights,mix=mix,erase=bool(erasing and active),box=[y,y+h,x,x+w]))
    return rows
def run(key):
    original_run(key);cfg=m.config();trace=m.s.read(m.OUT/key/'method_trace.json')
    for row in trace:
        plan=independent_plan(row['n'],row['epoch'],row['batch'],cfg['cases'][key]['erasing']);sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest();v=row['view_info'];assert v['appearance_plan_SHA256']==sha and v['appearance_erasing']==cfg['cases'][key]['erasing'] and v['appearance_augmented_view_only']
    p=m.OUT/key/'science/audit.json';proof=m.s.read(p);proof.update(independent_all_appearance_plans=True,clean_teacher_unmodified=True,not_full_original_AugMix_reproduction=True);m.s.save(p,proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
