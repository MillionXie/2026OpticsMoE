"""Reuse independently reconstructed masks and weighted-view saved science."""
import types,numpy as np,hashlib
import caltech20_rich_input_dropout46 as m
import audit_caltech20_rich_augmented_supervision46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','original_run','zoom_audit','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited_science=run
def run(key):
    inherited_science(key);meta=m.s.read(m.OUT/key/'metadata.json');cfg=meta['config'];assert cfg['input_dropout']==.03 and cfg['feature_dropout']==.05
    for row in m.s.read(m.OUT/key/'method_trace.json'):
        n=row['n'];rg=np.random.default_rng(np.random.SeedSequence([17,row['epoch'],row['batch'],20261018,40,1]));rg.uniform(cfg['crop_area_min'],1,n);rg.uniform(.9,1.1,n);rg.uniform(-1,1,n);rg.uniform(-1,1,n);rg.random(n);rg.uniform(.8,1.2,(n,3,1,1));rg.uniform(.8,1.2,(n,1,1,1));rg.random(n);keep=(rg.random((n,1,16,16))>=cfg['input_dropout']).astype('uint8');feature=(rg.random((n,256))>=cfg['feature_dropout']).astype('uint8');v=row['view_info'];assert v['actual_input_keep']==dict(sha256=hashlib.sha256(keep.tobytes()).hexdigest(),kept=int(keep.sum()),n=n) and v['actual_feature_keep_SHA256']==hashlib.sha256(feature.tobytes()).hexdigest()
    path=m.OUT/key/'science/audit.json';proof=m.s.read(path);proof.update(independent_actual_seeded_input_and_feature_masks=True,input_dropout_probability=.03,clean_eval_input_dropout_disabled=True);m.s.save(path,proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
