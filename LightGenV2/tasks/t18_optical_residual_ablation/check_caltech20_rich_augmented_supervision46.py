"""Independent CE view-weight gradient plus actual TRAIN micro4 admission."""
import types,numpy as np,torch
import caltech20_rich_augmented_supervision46 as m
import check_caltech20_rich_rare_zoomout46 as previous
import audit_caltech20_rich_augmented_supervision46 as audit
for name in ('independent_teacher','original_run','inherited','numpy_transform','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run','inherited','numpy_transform'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
prior=run
def run():
    prior();r=np.random.default_rng(76);a=torch.tensor(r.normal(size=(5,20)),requires_grad=True);b=torch.tensor(r.normal(size=(5,20)),requires_grad=True);y=torch.tensor([0,3,7,12,19]);loss=.2*torch.nn.functional.cross_entropy(a,y)+.8*torch.nn.functional.cross_entropy(b,y);ga,gb=torch.autograd.grad(loss,(a,b));target=np.eye(20)[y.numpy()]
    for z,g,weight in ((a,ga,.2),(b,gb,.8)):
        raw=z.detach().numpy();p=np.exp(raw-raw.max(1,keepdims=True));p/=p.sum(1,keepdims=True);assert np.max(abs(g.numpy()-weight*(p-target)/5))<1e-12
    bench=m.source_closure;path=bench.OUT/'benchmark.json';assert m.s.sha(path)=='f2d4149faea8aae24e947232942344ca747cece0aef4b64d38ca5ca0502ae535';q=m.s.read(path);ex=m.s.read(bench.OUT.with_name(bench.OUT.name+'.producer_exit.json'));assert q['passed'] and q['sources']==bench.sources() and q['state_before']==q['state_after'] and ex['returncode']==0 and ex['actual_exit'];row=next(x for x in q['rows'] if x['micro']==4);assert row['passed'] and row['peak_allocated_bytes']<2*1024**3 and row['gradient_max_relative']<=2e-4 and row['probability_maxabs']<=3e-6
    assert m.config()['microbatch_size']==4 and m.config()['clean_CE_coefficient']+m.config()['augmented_CE_coefficient']==1 and m.config()['test_interval']==1 and m.allocated_terms(2,3,5,7,'clean_targets')==(2,5)
    proof=m.s.read(m.CPU/'checks.json');proof.update(independent_weighted_view_gradient=True,micro4_TRAIN_throughput_admission_SHA256=m.s.sha(path),actual_full_all20_tail5_GPU_gate_still_required=True,direct_TEST_only_selection=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
