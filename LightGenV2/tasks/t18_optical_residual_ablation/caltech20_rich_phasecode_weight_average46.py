"""Finite VALID weight averaging of verified retained depth4/6 endpoints."""
import types
import caltech20_rich_weight_average as original
import caltech20_rich_L2_retained_soup_finetune as closure
import caltech20_rich_phasecode_gate_contract as recent
for n,v in vars(original).items():
    if not n.startswith('__') and n not in ('original','closure','recent'):globals()[n]=v
FILES=('caltech20_rich_phasecode_weight_average46.py','check_caltech20_rich_phasecode_weight_average46.py','audit_caltech20_rich_phasecode_weight_average46.py','CALTECH20_RICH_PHASECODE_WEIGHT_AVERAGE46_PROTOCOL_20261018.md')

def config():
    c=original.config();c.update(run='caltech20_rich20_retained_phasecode_VALID_weight_average46_20261018',depths=[4,6],L2_frozen_for_hardware=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_phasecode_weight_average46_CPU_20261018'

def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','parent','state','initialize','ready','configure','score','fit','test'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)

@lru_cache(None)
def bank(depth):
    assert depth in (4,6);old_states,old_rows=original.bank(depth);initial=old_states[0];j=1 if depth==4 else 2
    states=[initial,old_states[j]];rows=[old_rows[0],old_rows[j]]
    key=f'L{depth}_phasecode';folder=recent.OUT/key;fit=s.read(folder/'fit_result.json');receipt=s.read(folder/'audit_completed.json');ex=s.read(recent.OUT/(key+'.producer_exit.json'));meta=s.read(folder/'metadata.json')
    assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed']
    assert meta['sources']==recent.sources();engine.verify_sources(meta['sources'],meta['git_commit']);path=folder/'last.pt';assert fit['last_sha256']==s.sha(path)
    last=torch.load(path,map_location='cpu',weights_only=False);state={n:v.detach().cpu().clone() for n,v in last['ema'].items()}
    assert state.keys()==initial.keys() and all(state[n].shape==v.shape and state[n].dtype==v.dtype and torch.isfinite(state[n]).all() for n,v in initial.items())
    assert all(.4001<float(v.float())<.95 for n,v in state.items() if n.startswith('fusion_logits.'))
    states.append(state);rows.append(dict(path=str(path),sha256=s.sha(path),role='verified_phasecode_endpoint_EMA',source_git=meta['git_commit'],science_SHA256=receipt['audit_sha256'],actual_joint_budget=fit['previous_budget_epochs']+fit['actual_new_epochs'],epoch=last['epoch']))
    return states,rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',choices=[4,6],type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
