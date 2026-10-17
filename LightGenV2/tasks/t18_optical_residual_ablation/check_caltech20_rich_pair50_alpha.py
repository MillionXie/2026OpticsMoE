"""Own asymmetric-alpha/graph/data proof; zero TEST images or optimizer updates."""
import argparse,ast,builtins,inspect,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_pair50_alpha as m
from caltech20_rich_pair50_alpha_model import fuse,alpha_value,BOUNDS,INITIAL
from check_caltech20_alt246_richsplit import abo_reference

def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and m.slots()['total']==0
    m.claim(out.with_name(out.name+'.claim.json'),'own_pair50_asymmetric_alpha_CPU');torch.set_num_threads(4);cfg=m.config();manifest=m.manifest();data=m.load('train')
    assert len(data[1])==3141 and [sum(manifest['supports'][r]) for r in ('train','val','test')]==[3141,383,411]
    assert len({i for role in ('train','val','test') for i in manifest['split_ids'][role]})==3935
    old=m.s.read(m.HERE/'runs/smoke/caltech20_richsplit_cpu_20261016/checks.json');assert old['passed'] and old['sources']==m.dataset.sources();m.engine.verify_sources(old['sources'],old['git_commit'])
    ref=abo_reference();reference_checks=[]
    for rho in (.3,0.):
        torch.manual_seed(17);e=torch.rand(2,13,17,requires_grad=True);o=torch.rand_like(e,requires_grad=True);a=torch.tensor(-.4,requires_grad=True);ee=e.detach().clone().requires_grad_();oo=o.detach().clone().requires_grad_();aa=a.detach().clone().requires_grad_();w=torch.randn_like(e)
        x=fuse(e,o,a,rho);y=ref(ee,oo,aa,bounds=BOUNDS[rho]);gx=torch.autograd.grad((x*w).sum(),(e,o,a));gy=torch.autograd.grad((y*w).sum(),(ee,oo,aa))
        assert torch.equal(x,y) and all(torch.equal(v,u) for v,u in zip(gx,gy));reference_checks.append(dict(rho=rho,bounds=list(BOUNDS[rho]),forward_gradient_exact=True))
        endpoints=alpha_value(torch.tensor([-50.,0.,50.]),rho);assert (endpoints>=BOUNDS[rho][0]).all() and (endpoints<=BOUNDS[rho][1]).all() and BOUNDS[rho][0]>(.85 if rho==0 else .7)
    checks=[]
    for depth in (2,4,6):
        physical_initial=[];raw_initial=[]
        for rho in (.3,0.):
            m.engine.seed(17);model=m.PairModel(depth,rho);raw_initial.append(m.s.digest(model.state_dict()));nonalpha=m.s.digest({n:v for n,v in model.state_dict().items() if not n.startswith('fusion_logits.')});physical_initial.append(nonalpha)
            params=dict(model.named_parameters());assert model.stages==['expert','global']*(depth//2) and len(params)=={2:12,4:21,6:30}[depth] and np.allclose(model.alphas(),[INITIAL]*depth,atol=1e-7)
            assert sum(p.numel() for n,p in params.items() if n=='raw_router' or n.startswith('phase_sets.'))==50176+429188*(depth//2) and sum(p.numel() for n,p in params.items() if n!='raw_router' and not n.startswith('phase_sets.'))==5140+10*depth
            assert model.contract()['alpha_bounds']==list(BOUNDS[rho]) and model.contract()['terminal_linear_parameters']==5140
            with torch.no_grad():
                black,_=model(m.encode(torch.zeros(1,3,150,150,dtype=torch.uint8),None,100));assert torch.allclose(black,torch.full_like(black,.05),atol=1e-8)
                prob,_=model(m.encode(data[0][:1],None,100));features=model.readout['features'].numpy().astype(np.float64);W=model.linear.weight.detach().numpy().astype(np.float64);bias=model.linear.bias.detach().numpy().astype(np.float64);z=features@W.T+bias;q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True);error=float(np.abs(q-prob.numpy()).max());assert error<1e-6 and model.trace==['router']+model.stages
            checks.append(dict(depth=depth,rho=rho,initial_digest=raw_initial[-1],initial_nonalpha_digest=nonalpha,initial_effective_alpha=model.alphas(),contract=model.contract(),numpy_maxabs=error));del model
        assert physical_initial[0]==physical_initial[1] and raw_initial[0]!=raw_initial[1]
    for routine in ('fit','campaign','supervise'):
        node=ast.parse(inspect.getsource(getattr(m.frozen,routine))).body[0];reads={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)};local={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Param))}|{a.arg for a in node.args.args};missing=reads-local-set(m.__dict__)-set(dir(builtins));assert not missing,(routine,missing)
    out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,checks=checks,ABO_fusion_forward_gradient_checks=reference_checks,forward_calls=12,optimizer_steps=0,TEST_images_read=0,TEST_model_evaluations=0,old_split_proof_sha256=m.s.sha(m.HERE/'runs/smoke/caltech20_richsplit_cpu_20261016/checks.json'),maximum_epochs=50,steps_per_epoch=197,tail_samples=5,raw_alpha_initial_logits_intentionally_different=True,nonalpha_initial_and_effective_alpha_equal=True,comparison_scope=cfg['comparison_scope']),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
