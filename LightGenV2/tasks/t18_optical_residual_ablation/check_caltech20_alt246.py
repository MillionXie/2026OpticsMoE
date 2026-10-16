"""Independent structural and ABO-fusion parity admission, without TEST arrays."""
import argparse,ast,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246 as study
from caltech20_alt246_model import HybridMoE,alpha_value,fuse,features

def abo_reference():
    source=(study.ROOT/study.ABO).read_text();tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('rms','alpha_value','fuse')]
    assert len(functions)==3
    ns={'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(study.ROOT/study.ABO),'exec'),ns)
    return ns['fuse']

def check(out):
    s=study.s;assert out==study.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    study.slots();study.claim(out.with_name(out.name+'.claim.json'),'own_cpu_structural_and_ABO_math')
    torch.set_num_threads(4);cfg=study.config();m=study.old.manifest(Path(cfg['data']));data=study.load('train');val=study.load('val')
    assert len(data[1])==1372 and len(val[1])==283 and len(data[1])%16==12
    reference=abo_reference();parity=[]
    for shape in ((2,13,17),(3,8,8)):
        torch.manual_seed(17);e=torch.rand(shape,requires_grad=True);o=torch.rand(shape,requires_grad=True);a=torch.tensor(-.4,requires_grad=True)
        ee=e.detach().clone().requires_grad_();oo=o.detach().clone().requires_grad_();aa=a.detach().clone().requires_grad_();weight=torch.randn(shape)
        v=fuse(e,o,a);w=reference(ee,oo,aa,bounds=(.7001,.95));gv=torch.autograd.grad((v*weight).sum(),(e,o,a));gw=torch.autograd.grad((w*weight).sum(),(ee,oo,aa))
        error=float((v-w).abs().max());grad=max(float((x-y).abs().max()) for x,y in zip(gv,gw));assert error==0 and grad==0
        parity.append(dict(shape=shape,maxabs=error,gradient_maxabs=grad))
    for raw in (-100.,0.,100.):assert .7<float(alpha_value(torch.tensor(raw)))<=.95
    assert torch.isfinite(fuse(torch.zeros(1,8,8),torch.zeros(1,8,8),torch.tensor(0.))).all()
    checks=[];forward_calls=0
    for depth in (2,4,6):
        digests=[]
        for rho in (.3,0.):
            study.engine.seed(17);model=HybridMoE(depth,rho);digests.append(s.digest(model.state_dict()))
            assert model.contract()['router_count']==1 and model.stages==['expert','global']*(depth//2)
            names=dict(model.named_parameters());assert len([n for n in names if n=='raw_router' or n.startswith('phase_sets.')])==1+5*(depth//2)
            assert sum(p.numel() for n,p in names.items() if n=='raw_router' or n.startswith('phase_sets.'))==50176+429188*(depth//2)
            assert sum(p.numel() for n,p in names.items() if n!='raw_router' and not n.startswith('phase_sets.'))==5140+11*depth
            assert all(abs(a-.8)<1e-6 for a in model.alphas())
            with torch.no_grad():
                p,_=model(study.encode(torch.zeros(1,3,150,150,dtype=torch.uint8),None,100));forward_calls+=1
                assert torch.allclose(p,torch.full_like(p,.05),atol=1e-8) and model.trace==['router']+model.stages
                x=study.encode(data[0][:1],None,100);p,_=model(x);forward_calls+=1
                f=model.readout['features'].numpy().astype('float64');W=model.linear.weight.detach().numpy().astype('float64');b=model.linear.bias.detach().numpy().astype('float64')
                logits=f@W.T+b;q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
                assert np.allclose(q,p.numpy(),atol=1e-6) and model.trace==['router']+model.stages
            checks.append(dict(depth=depth,rho=rho,contract=model.contract(),initial_digest=digests[-1],black20_uniform=True,trace=model.trace,linear_numpy_maxabs=float(np.abs(q-p.numpy()).max())))
            del model
        assert digests[0]==digests[1]
    out.mkdir();s.save(out/'checks.json',dict(study.identity(),passed=True,checks=checks,abo_parity=parity,abo_reference_sha256=s.sha(study.ROOT/study.ABO),alpha_bounds_proven=True,train_support=m['supports']['train'],val_support=m['supports']['val'],forward_calls=forward_calls,optimizer_steps=0,test_arrays_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        dest=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not dest.exists():study.s.save(dest,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
