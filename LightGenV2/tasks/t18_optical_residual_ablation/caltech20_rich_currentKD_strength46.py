"""One bounded stronger-KD comparison, same calibrated teacher and warm parents."""
import types
import caltech20_rich_currentKD46 as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_currentKD_strength46.py','check_caltech20_rich_currentKD_strength46.py','audit_caltech20_rich_currentKD_strength46.py','CALTECH20_RICH_CURRENTKD_STRENGTH46_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();c.update(run='caltech20_rich20_currentTRAIN_calibrated_teacherKD_strength05_46_local5_20261018',KD_weight=.5,gpu_order=[0,1,2,3,4],cases={'L4_strongerKD':dict(c['cases']['L4_currentKD']),'L6_strongerKD':dict(c['cases']['L6_currentKD'])});return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_currentKD_strength46_CPU_20261018'
def sources():
    z=baseline.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','mixed','campaign','feature_loss','target_data','initialize','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','gradients','ready','fit'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_target_MixUp_for_one_KD_strength_comparison',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
