"""Complete independent saved science with only the declared KD weight changed."""
import types
import caltech20_rich_currentKD_strength46 as m
import audit_caltech20_rich_currentKD46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
