"""VALID-selected single-network interpolation after electronic co-adaptation."""
import types
import caltech20_rich_weight_average as original
import caltech20_rich_electronic_curriculum as recent
for n,v in vars(original).items():
    if not n.startswith('__') and n not in ('original','recent','previous'):globals()[n]=v
previous=types.SimpleNamespace(**vars(original.previous))
FILES=('caltech20_rich_electronic_weight_average46.py','check_caltech20_rich_electronic_weight_average46.py','audit_caltech20_rich_electronic_weight_average46.py','CALTECH20_RICH_ELECTRONIC_WEIGHT_AVERAGE46_PROTOCOL_20261018.md')
def config():
    c=original.config();c.update(run='caltech20_rich20_electronic_curriculum_VALID_weight_average46_20261018',depths=[4,6],L2_final_no_further_training_or_TEST=True,parent_local_electronic_epochs=8,interpolation_bank='canonical_VALIDbest_electronicLast',optical_phases_identical=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_electronic_weight_average46_CPU_20261018'
def sources():
    z=recent.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def parent(depth):
    assert depth in (4,6);return recent.parent(depth)
def initial_state(depth):
    assert depth in (4,6);return recent.initial_state(depth)
previous.parent=parent;previous.initial_state=initial_state
for name in ('identity','claim','state','initialize','configure','score','fit','test'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
@lru_cache(None)
def bank(depth):
    q,pt,_=parent(depth);initial=initial_state(depth);states=[initial];rows=[dict(path=q['path'],sha256=q['sha256'],role='canonical_unchanged',selected_epoch=q['selected_optical_epoch'],joint_budget=q['joint_budget_epochs'])];key=f'L{depth}_gentle';f=recent.OUT/key;fit=s.read(f/'fit_result.json');meta=s.read(f/'metadata.json');ex=s.read(recent.OUT/(key+'.producer_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==recent.sources() and fit['actual_electronic_epochs']==8 and fit['optical_phase_updates']==0;engine.verify_sources(meta['sources'],meta['git_commit'])
    for phase in ('fit','test'):
        rec=s.read(f/(phase+'_audit_completed.json'));assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(f/(phase+'_science/audit.json')) and s.read(f/(phase+'_science/audit.json'))['passed']
    for name in ('best','last'):
        path=f/(name+'.pt');assert fit['checkpoint_SHA256'][name]==s.sha(path);p=torch.load(path,map_location='cpu',weights_only=False);state={n:v.detach().cpu().clone() for n,v in (p['model'] if name=='best' else p['ema']).items()};assert state.keys()==initial.keys() and all(state[n].shape==v.shape and state[n].dtype==v.dtype and torch.isfinite(state[n]).all() for n,v in initial.items());assert all(torch.equal(state[n],v) for n,v in initial.items() if n=='raw_router' or n.startswith('phase_sets.'));states.append(state);rows.append(dict(path=str(path),sha256=s.sha(path),role='verified_electronic_'+name,state_key='model' if name=='best' else 'ema',source_git=meta['git_commit'],science_SHA256=s.read(f/'fit_audit_completed.json')['audit_sha256'],actual_added_electronic_epochs=8,local_electronic_epoch=p['local_electronic_epoch'],new_joint_epochs=0))
    return states,rows
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['optical_phase_bytes_identical_for_all_fractions'];identity();recent.closure.old.controls()
    for depth in (4,6):bank(depth)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',type=int,choices=[4,6],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
