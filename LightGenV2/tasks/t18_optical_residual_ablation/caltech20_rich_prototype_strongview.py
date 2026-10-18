"""Matched stronger TRAIN views, same prototype objective and inference."""
import types
import caltech20_rich_prototype_consistency as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('original','baseline'):globals()[n]=v
original=baseline.original
FILES=('caltech20_rich_prototype_strongview.py','caltech20_rich_prototype_strongview_profile.json','check_caltech20_rich_prototype_strongview.py','audit_caltech20_rich_prototype_strongview.py','CALTECH20_RICH_PROTOTYPE_STRONGVIEW_PROTOCOL_20261018.md')
def config():
    c=baseline.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_prototype_strongview_cpu_20261018'
def sources():
    z=baseline.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','gradients','campaign','feature_loss','initialize','gate','ready'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
