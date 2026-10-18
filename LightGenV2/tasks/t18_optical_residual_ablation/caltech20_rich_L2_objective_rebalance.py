"""Three local epochs testing a weaker TRAIN prototype constraint, retained L2."""
import types
import caltech20_rich_L2_retained_soup_finetune as baseline
import caltech20_rich_L2_local_weight_average as closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','closure','original'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_L2_objective_rebalance.py','check_caltech20_rich_L2_objective_rebalance.py','audit_caltech20_rich_L2_objective_rebalance.py','CALTECH20_RICH_L2_OBJECTIVE_REBALANCE_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])

def config():
    c=baseline.config();c.update(run='caltech20_rich20_L2_retained_objective_rebalance_local3_20261018',prototype_scale_multiplier=.25,test_interval=3,maximum_seconds=900,cases={'L2_weaker_prototype':dict(depth=2,fresh=False,epochs=3,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True)});return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L2_objective_rebalance_CPU_20261018'

def sources():
    z=closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','mixed','campaign','feature_loss','target_data','initialize','incumbent','gradients','gradient_kernel','gate_kernel','complete_gate_contract','gate','fit_kernel'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=baseline.prototype_data.__wrapped__;raw_prototype=lru_cache(None)(types.FunctionType(f.__code__,globals(),'same_original_prototype_calibration',f.__defaults__,f.__closure__))

@lru_cache(None)
def prototype_data(depth):
    target,scale,sha,info=raw_prototype(depth);info=dict(info,original_scale=scale,scale=scale*.25,scale_multiplier=.25,relative_clean_CE_feature_gradient=.05);return target,scale*.25,sha,info

def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['quarter_scale_independently_verified'] and p['loaded_actual_8005_model_not_fresh'];identity();old.controls();prototype_data(2)

def fit(key,gpu):
    assert not config()['cases'][key]['fresh'];fit_kernel(key,gpu);path=OUT/key/'fit_result.json';q=s.read(path);assert q['previous_budget_epochs'] is None;q.update(local_added_epochs=3,previous_training_has_no_single_epoch=True,not_fresh_initialization=True,loaded_canonical_PT_SHA256=incumbent(2)[0]['sha256'],historical_contributing_optical_epochs=[75,90,90],prototype_scale_multiplier=.25);s.save(path,q)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
