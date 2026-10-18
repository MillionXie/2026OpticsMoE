"""Class-common logit gauge centering before float32; same5140 classifier."""
import inspect,types
import caltech20_rich_shrinkage_readout as original
for n,v in vars(original).items():
    if not n.startswith('__'):globals()[n]=v
EXTRA=('caltech20_rich_centered_shrinkage_readout.py','check_caltech20_rich_centered_shrinkage_readout.py','audit_caltech20_rich_centered_shrinkage_readout.py','CALTECH20_RICH_CENTERED_SHRINKAGE_PROTOCOL_20261018.md')
FILES=(EXTRA[0],original.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=original.config();c.update(run=c['run']+'_classcentered_float32',float32_stability='Subtract same class-mean W/b from all logits before float32 cast; exact real-valued softmax invariant');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_centered_shrinkage_cpu_20261018'
def sources():
    z=original.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('parents','core_digest','train_statistics','discriminant','probabilities','test','identity','claim'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['real_role_centering_proof_passed'] and not CPU.with_name(CPU.name+'.failure.json').exists();identity();stage.old.controls();return parents()
text=inspect.getsource(original.fit);site='w=(w*scale).astype(np.float32);b=(b*scale).astype(np.float32)';assert text.count(site)==1
text=text.replace(site,'w=w*scale;b=b*scale;w=(w-w.mean(0,keepdims=True)).astype(np.float32);b=(b-b.mean()).astype(np.float32)');exec(compile(text,__file__+':same_fit_with_exact_classcommon_gauge_removed','exec'),globals())
text=inspect.getsource(original.audit)
for a,b in [('w*scale,atol=1e-6','(w-w.mean(0,keepdims=True))*scale,atol=1e-6'),('b*scale,atol=1e-6','(b-b.mean())*scale,atol=1e-6'),('wi=(wi*expected_scale).astype(np.float32).astype(float);bi=(bi*expected_scale).astype(np.float32).astype(float)','wi=wi*expected_scale;bi=bi*expected_scale;wi=(wi-wi.mean(0,keepdims=True)).astype(np.float32).astype(float);bi=(bi-bi.mean()).astype(np.float32).astype(float)')]:
    assert text.count(a)==1;text=text.replace(a,b)
exec(compile(text,__file__+':independent_covariance_and_same_gauge_math','exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','audit_fit','audit_test'],required=True);a=p.parse_args()
    try:audit(a.phase.split('_')[1]) if a.phase.startswith('audit_') else globals()[a.phase]()
    except BaseException as e:
        mark=OUT/(a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
