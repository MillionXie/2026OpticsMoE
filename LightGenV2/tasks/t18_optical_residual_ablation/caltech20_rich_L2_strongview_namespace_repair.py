"""Explicit audit NumPy namespace repair; preserve source382 CPU failure."""
import types
import caltech20_rich_L2_prototype_strongview as prior
for n,v in vars(prior).items():
    if not n.startswith('__') and n not in ('original','prior','parent'):globals()[n]=v
parent=prior.parent;original=prior.original
EXTRA=('caltech20_rich_L2_strongview_namespace_repair.py','check_caltech20_rich_L2_strongview_namespace_repair.py','audit_caltech20_rich_L2_strongview_namespace_repair.py','CALTECH20_RICH_L2_STRONGVIEW_NAMESPACE_REPAIR_PROTOCOL_20261018.md')
FILES=(EXTRA[0],prior.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=prior.config();c.update(run=c['run']+'_auditor_namespace_checked');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L2_strongview_namespace_cpu_20261018'
def sources():
    z=prior.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','gradients','campaign','feature_loss','initialize','gate','ready','admit_completed_mild_L2'):
    f=getattr(prior,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:admit_completed_mild_L2();campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
