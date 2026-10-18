"""One ordinary per-image CE comparison for requested overall accuracy."""
import types
import caltech20_rich_currentKD46 as baseline
import caltech20_rich_sameclass_patch46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_overallCE46.py','check_caltech20_rich_overallCE46.py','audit_caltech20_rich_overallCE46.py','CALTECH20_RICH_OVERALLCE46_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();c.update(run='caltech20_rich20_overallCE_teacher46_local5_20261018',class_weight_exponent=0.0,gpu_order=[1,4,0,3],cases={'L4_overallCE':dict(c['cases']['L4_currentKD']),'L6_overallCE':dict(c['cases']['L6_currentKD'])},report_macro_and_all_class_tradeoffs=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_overallCE46_CPU_20261018'
def sources():
    z=source_closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','mixed','campaign','feature_loss','target_data','initialize','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','gradients','ready','fit'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_calibrated_teacher_targets_only_CE_weights_change',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
ready_kernel=ready
def ready():
    q=s.read(CPU/'checks.json');assert q['independent_ordinary_CE_derivative'] and q['all_real_class_weights_exactly_one'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
