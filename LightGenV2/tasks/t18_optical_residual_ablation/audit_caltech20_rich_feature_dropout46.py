"""Reuse independently reconstructed masks and weighted-view saved science."""
import types
import caltech20_rich_feature_dropout46 as m
import audit_caltech20_rich_augmented_supervision46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','original_run','zoom_audit','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited_science=run
def run(key):
    inherited_science(key);meta=m.s.read(m.OUT/key/'metadata.json');assert meta['config']['feature_dropout']==.2;path=m.OUT/key/'science/audit.json';proof=m.s.read(path);proof.update(independent_seeded_feature_dropout_masks=True,feature_dropout_probability=.2,clean_eval_dropout_disabled=True);m.s.save(path,proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
