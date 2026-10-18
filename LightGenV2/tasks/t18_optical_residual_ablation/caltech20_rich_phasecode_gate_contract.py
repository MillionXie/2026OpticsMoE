"""Prelaunch producer/auditor capture-key contract fix; source421 unexecuted."""
import types
import caltech20_rich_phasecode_consistency as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_phasecode_gate_contract.py','check_caltech20_rich_phasecode_gate_contract.py','audit_caltech20_rich_phasecode_gate_contract.py','CALTECH20_RICH_PHASECODE_GATE_CONTRACT_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();c.update(run=c['run']+'_gate_contract_checked');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_phasecode_gate_contract_CPU_20261018'
def sources():
    z=baseline.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','campaign','feature_loss','target_data','initialize','incumbent','gradients','gradient_kernel','ready'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=baseline.prototype_data.__wrapped__;prototype_data=lru_cache(None)(types.FunctionType(f.__code__,globals(),'prototype_data',f.__defaults__,f.__closure__))
f=baseline.gate;gate_kernel=types.FunctionType(f.__code__,globals(),'same_actual_phasecode_gate',f.__defaults__,f.__closure__)
def complete_gate_contract(proof):
    result=dict(proof);assert result['parent_capture_maxabs']<=3e-6 and result['parent_clean_probability_maxabs']<=3e-6;result['parent_clean_capture_maxabs']=result['parent_capture_maxabs'];return result
def gate(depth,fresh,data,case):return complete_gate_contract(gate_kernel(depth,fresh,data,case))
ready_kernel=ready
def ready():
    proof=s.read(CPU/'checks.json');assert proof['gate_producer_auditor_capture_key_contract_proved'] and proof['numerical_thresholds_unchanged'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
