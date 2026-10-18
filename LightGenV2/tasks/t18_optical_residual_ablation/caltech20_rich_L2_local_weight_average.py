"""VALID-select one deployable L2 average of canonical and completed short tune."""
import types
import caltech20_rich_weight_average as original
import caltech20_rich_L2_retained_soup_finetune as retained
import caltech20_rich_phasecode_weight_average46 as closure
for n,v in vars(original).items():
    if not n.startswith('__') and n not in ('original','retained','closure'):globals()[n]=v
FILES=('caltech20_rich_L2_local_weight_average.py','check_caltech20_rich_L2_local_weight_average.py','audit_caltech20_rich_L2_local_weight_average.py','CALTECH20_RICH_L2_LOCAL_WEIGHT_AVERAGE_PROTOCOL_20261018.md')

def config():
    c=original.config();c.update(run='caltech20_rich20_L2_local_VALID_weight_average_20261018',depths=[2],one_hour_priority=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L2_local_weight_average_CPU_20261018'

def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def parent(depth):assert depth==2;return retained.incumbent(depth)
def initial_state(depth):
    q,pt,_=parent(depth);return retained.remap(pt['model'],q['alpha_low'])
# Explicit private context; never mutate an imported frozen module's namespace.
previous=types.SimpleNamespace(**vars(original.previous));previous.initial_state=initial_state
for name in ('identity','claim','state','initialize','ready','configure','score','fit','test'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)

@lru_cache(None)
def bank(depth):
    assert depth==2;q,_,_=parent(depth);states=[initial_state(depth)];rows=[dict(path=q['path'],sha256=q['sha256'],role='canonical_329_TEST_correct',parameter_average=True,historical_contributing_epochs=[75,90,90],single_historical_epoch=None)]
    folder=retained.OUT/'L2_soup_finetune';fit=s.read(folder/'fit_result.json');rec=s.read(folder/'audit_completed.json');ex=s.read(retained.OUT/'L2_soup_finetune.producer_exit.json');meta=s.read(folder/'metadata.json')
    assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed']
    assert meta['sources']==retained.sources();engine.verify_sources(meta['sources'],meta['git_commit']);assert fit['actual_new_epochs']==10 and fit['previous_budget_epochs'] is None
    for file,state_key,label in [('best.pt','model','verified_best_local_tune'),('last.pt','ema','verified_final_local_EMA')]:
        path=folder/file;assert fit[('best' if file=='best.pt' else 'last')+'_sha256']==s.sha(path);pt=torch.load(path,map_location='cpu',weights_only=False);state={n:v.detach().cpu().clone() for n,v in pt[state_key].items()};assert state.keys()==states[0].keys() and all(state[n].shape==v.shape and state[n].dtype==v.dtype and torch.isfinite(state[n]).all() for n,v in states[0].items());states.append(state);rows.append(dict(path=str(path),sha256=s.sha(path),state_key=state_key,role=label,source_git=meta['git_commit'],science_SHA256=rec['audit_sha256'],local_added_epoch=pt['tune_epoch'],single_historical_epoch=None))
    return states,rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',choices=[2],type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
