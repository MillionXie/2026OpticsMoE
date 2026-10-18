"""Continue the scientifically improved L4 prototype policy from selected E100."""
import types
import caltech20_rich_prototype_consistency as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('original','baseline'):globals()[n]=v
original=baseline.original
import caltech20_rich_L6_shared_supervision as closure
EXTRA=('caltech20_rich_L4_prototype_continue.py','check_caltech20_rich_L4_prototype_continue.py','audit_caltech20_rich_L4_prototype_continue.py','CALTECH20_RICH_L4_PROTOTYPE_CONTINUE_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();c.update(run='caltech20_rich20_L4_successful_prototype_continue_s17_20261018',cases={'L4_prototype_continue':dict(depth=4,fresh=False,epochs=10,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True)},gpu_order=[2,4,1,3,0]);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L4_prototype_continue_cpu_20261018'
def sources():
    z=closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def completed_L4():
    folder=baseline.OUT/'L4_prototype';ex=s.read(baseline.OUT/'L4_prototype.producer_exit.json');rec=s.read(folder/'audit_completed.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed'];f=s.read(folder/'fit_result.json');assert f['best_sha256']==s.sha(folder/'best.pt') and f['metrics']['test']['accuracy']>baseline.incumbent(4)[1]['metrics']['test']['accuracy'];return f
@lru_cache(None)
def incumbent(depth):
    if depth!=4:return baseline.incumbent(depth)
    f=completed_L4();folder=baseline.OUT/'L4_prototype';meta=s.read(folder/'metadata.json');assert meta['sources']==baseline.sources();pt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);q=dict(path=str(folder/'best.pt'),sha256=f['best_sha256'],source_git=meta['git_commit'],module=baseline.__name__,class_name='Alpha40MoE',alpha_low=None,joint_budget_epochs=f['previous_budget_epochs']+f['actual_new_epochs'],selected_optical_epoch=pt['epoch']);pred={r:load_predictions(folder/('selected_'+r+'.npz')) for r in ('train','val','test')};return q,pt,pred
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','gradients','campaign','feature_loss','initialize','gate','ready'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=baseline.prototype_data.__wrapped__;prototype_data=lru_cache(None)(types.FunctionType(f.__code__,globals(),'prototype_data',f.__defaults__,f.__closure__));f=baseline.target_data;target_data=types.FunctionType(f.__code__,globals(),'target_data',f.__defaults__,f.__closure__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:completed_L4();campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
