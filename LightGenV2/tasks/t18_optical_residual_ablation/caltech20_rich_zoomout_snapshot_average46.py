"""VALID-selected single parameter interpolation of verified preserved zoom states."""
import types
import caltech20_rich_weight_average as original
import caltech20_rich_currentKD46 as retained
import caltech20_rich_rare_zoomout46 as train
for n,v in vars(original).items():
    if not n.startswith('__') and n not in ('original','previous'):globals()[n]=v
previous=types.SimpleNamespace(**vars(original.previous))
FILES=('caltech20_rich_zoomout_snapshot_average46.py','check_caltech20_rich_zoomout_snapshot_average46.py','audit_caltech20_rich_zoomout_snapshot_average46.py','CALTECH20_RICH_ZOOMOUT_SNAPSHOT_AVERAGE46_PROTOCOL_20261018.md')
def config():
    c=original.config();c.update(run='caltech20_rich20_VALID_preserved_zoomout_average46_20261018',depths=[4,6],L2_FINAL_untouched=True,snapshot_selection='available_preserved_and_final_VALID_correct_then_balancedNLL_then_earliest_local',not_an_exhaustive_epoch_search=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_zoomout_snapshot_average46_CPU_20261018'
def sources():
    z=train.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def parent(depth):assert depth in (4,6);return retained.incumbent(depth)
def initial_state(depth):return {n:v.detach().cpu().clone() for n,v in parent(depth)[1]['model'].items()}
previous.parent=parent;previous.initial_state=initial_state
for name in ('identity','claim','state','initialize','configure','score','fit','test'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
@lru_cache(None)
def bank(depth):
    q,pt,pred=parent(depth);initial=initial_state(depth);key=f'L{depth}_rare_zoomout';folder=train.OUT/key;ex=s.read(train.OUT/(key+'.producer_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();rec=s.read(folder/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed'];meta=s.read(folder/'metadata.json');assert meta['sources']==train.sources();engine.verify_sources(meta['sources'],meta['git_commit']);hist=s.read(folder/'history.json');fit=s.read(folder/'fit_result.json');assert fit['actual_new_epochs']==5 and fit['last_sha256']==s.sha(folder/'last.pt');root=train.OUT.with_name(train.OUT.name+'_preserved_existing_checkpoints');paths=sorted(root.glob(key+'_local*.pt'))+[folder/'last.pt'];candidates=[]
    for path in paths:
        p=torch.load(path,map_location='cpu',weights_only=False);row=next(h for h in hist if h['tune_epoch']==p['tune_epoch']);assert p['sources']==train.sources() and p['config']==train.config() and p['steps']==197*p['tune_epoch'];assert s.digest(p['ema'])==row['ema_digest'] and s.digest(p['model'])==row['raw_digest']
        if path.parent==root:
            receipt=s.read(path.with_suffix('.receipt.json'));assert receipt['copied_SHA256']==s.sha(path) and receipt['history_row']==row
        item=dict(path=str(path),sha256=s.sha(path),state_key='ema',role='already_produced_verified_zoomout_state',local_epoch=p['tune_epoch'],train=row['train'],val=row['val'],source_git=meta['git_commit'],science_SHA256=rec['audit_sha256']);candidates.append(item)
    canonical=dict(path=q['path'],sha256=q['sha256'],state_key='model',role='retained_canonical',local_epoch=0,train=pt['metrics']['train'] if 'metrics' in pt else engine.metrics(pred['train']['labels'],pred['train']['probabilities']),val=engine.metrics(pred['val']['labels'],pred['val']['probabilities']))
    rank=lambda a:(int(np.trace(np.asarray(a['val']['confusion_matrix']))),-a['val']['balanced_nll'],-a['local_epoch']);chosen=max([canonical,*candidates],key=rank);loaded=torch.load(chosen['path'],map_location='cpu',weights_only=False);state={n:v.detach().cpu().clone() for n,v in loaded[chosen['state_key']].items()};assert state.keys()==initial.keys() and all(v.shape==initial[n].shape and v.dtype==initial[n].dtype and torch.isfinite(v).all() for n,v in state.items());row=dict(chosen,all_available_candidates=candidates,selection=list(rank(chosen)),not_new_optimizer_steps=True);return [initial,state,state],[canonical,row,dict(row)]
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['independent_raw_snapshot_interpolation'];identity();retained.old.controls()
    for d in (4,6):assert q['snapshot_banks'][str(d)]==bank(d)[1]
def test(depth,gpu):
    ready();rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');folder=OUT/f'L{depth}';claim(folder/'once_test_claim.json','single_VALID_selected_weight_average_TEST');fit=s.read(folder/'fit_result.json');assert fit['candidate_SHA256']==s.sha(folder/'candidate.pt');configure(gpu);torch.set_num_threads(4);same=fit['model_digest']==s.digest(initial_state(depth));source=train.OUT/f'L{depth}_rare_zoomout';last=torch.load(source/'last.pt',map_location='cpu',weights_only=False);known=fit['model_digest']==s.digest(last['ema'])
    if same:p=parent(depth)[2]['test'];metric=metrics(p['labels'],p['probabilities'])
    elif known:p=retained.load_predictions(source/'test_epoch05.npz');metric=metrics(p['labels'],p['probabilities'])
    else:model=initialize(depth,fit['selected_fraction']).cuda();metric,p=base.evaluate(model,base.load('test'))
    base.save_np(folder/'candidate_test.npz',p);s.save(folder/'test_result.json',dict(passed=True,metrics=metric,model_digest=fit['model_digest'],new_TEST_models=0 if same or known else 1,parent_TEST_reused=same,known_endpoint_TEST_reused=known and not same,TEST_used_to_select_fraction=False,TEST_informed_development=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',choices=[4,6],type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
