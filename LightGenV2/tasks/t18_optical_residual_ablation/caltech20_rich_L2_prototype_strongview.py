"""L2 matched strong-view followup after completed mild prototype review."""
import types
import caltech20_rich_prototype_strongview as parent
for n,v in vars(parent).items():
    if not n.startswith('__') and n not in ('original','parent'):globals()[n]=v
original=parent.original
EXTRA=('caltech20_rich_L2_prototype_strongview.py','check_caltech20_rich_L2_prototype_strongview.py','audit_caltech20_rich_L2_prototype_strongview.py','CALTECH20_RICH_L2_PROTOTYPE_STRONGVIEW_PROTOCOL_20261018.md')
FILES=(EXTRA[0],parent.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=parent.config();c.update(run='caltech20_rich20_TRAIN_class_prototype_strongview_L2_s17_20261018',cases={'L2_prototype_strong':dict(depth=2,fresh=False,epochs=10,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True)},gpu_order=[3,1,4,2,0]);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L2_prototype_strongview_cpu_20261018'
def sources():
    z=parent.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','gradients','campaign','feature_loss','initialize','gate','ready'):
    f=getattr(parent,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def admit_completed_mild_L2():
    previous=parent.baseline;folder=previous.OUT/'L2_prototype';rec=s.read(folder/'audit_completed.json');ex=s.read(previous.OUT/'L2_prototype.producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed'];fit=s.read(folder/'fit_result.json');pred=incumbent(2)[2]['test'];assert fit['actual_new_epochs']==10 and fit['best_sha256']==s.sha(folder/'best.pt') and fit['metrics']['test']['accuracy']<=float((pred['probabilities'].argmax(1)==pred['labels']).mean());return rec
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:admit_completed_mild_L2();campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
