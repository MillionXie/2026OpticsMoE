"""Fine-tune the actual80.05% averaged L2 state; no fresh initialization."""
import ast,inspect,types
import caltech20_rich_phasecode_gate_contract as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('original','baseline'):globals()[n]=v
original=baseline.original
import caltech20_rich_alpha40_rebalance as static_fit
EXTRA=('caltech20_rich_L2_retained_soup_finetune.py','check_caltech20_rich_L2_retained_soup_finetune.py','audit_caltech20_rich_L2_retained_soup_finetune.py','CALTECH20_RICH_L2_RETAINED_SOUP_FINETUNE_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();c.update(run='caltech20_rich20_L2_from_canonical_soup_finetune_local10_20261018',cases={'L2_soup_finetune':dict(depth=2,fresh=False,epochs=10,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True)},gpu_order=[3,4,2,0,1],maximum_seconds=1800,local_adaptation_epoch_labels=True,no_single_historical_epoch_for_averaged_parent=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L2_retained_soup_finetune_CPU_20261018'
def sources():
    z=baseline.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','mixed','campaign','feature_loss','target_data','initialize','incumbent','gradients','gradient_kernel','gate_kernel','complete_gate_contract','gate','ready_kernel'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=baseline.prototype_data.__wrapped__;prototype_data=lru_cache(None)(types.FunctionType(f.__code__,globals(),'prototype_data',f.__defaults__,f.__closure__))
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['loaded_actual_8005_model_not_fresh'] and p['local_counter_not_historical_epoch'] and not CPU.with_name(CPU.name+'.failure.json').exists();identity();old.controls();prototype_data(2)
# Existing exact fit, with projected alpha/micro2/view accounting and an honest
# LOCAL1..10 counter because a parameter average has no single historical epoch.
text=inspect.getsource(static_fit.fit);site='opt.step();steps+=1';assert text.count(site)==1;text=text.replace(site,'opt.step();model.project();steps+=1');site="gradients(model,x,y,q,mass,mask,weights,cfg,4)";assert text.count(site)==1;text=text.replace(site,"gradients(model,x,y,q,mass,mask,weights,cfg,cfg['microbatch_size'])");site='crop_sha256=crop,loss=loss,clip_norm=norm';assert text.count(site)==1;text=text.replace(site,site+',view_info=model.view_info');site="epoch=tune if case['fresh'] else parent['joint_budget_epochs']+tune";assert text.count(site)==1;text=text.replace(site,'epoch=tune');exec(compile(text,__file__+':same_retained_fit_with_local_adaptation_counter','exec'),globals());fit_kernel=fit
def fit(key,gpu):
    assert not config()['cases'][key]['fresh'];fit_kernel(key,gpu);p=OUT/key/'fit_result.json';q=s.read(p);assert q['previous_budget_epochs'] is None;q.update(local_added_epochs=10,previous_training_has_no_single_epoch=True,not_fresh_initialization=True,loaded_canonical_PT_SHA256=incumbent(2)[0]['sha256'],historical_contributing_optical_epochs=[75,90,90]);s.save(p,q)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
