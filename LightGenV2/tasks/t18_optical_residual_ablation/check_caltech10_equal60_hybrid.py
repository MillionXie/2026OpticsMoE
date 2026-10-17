"""New CPU contract: immutable existing split, identical hybrid graph, no TEST pixels."""
import os,argparse
from pathlib import Path
import numpy as np
import torch
import caltech10_equal60_hybrid as m
from check_caltech20_alt246_richsplit import abo_reference
from caltech20_alt246_biasfree_model import fuse,alpha_value
def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and m.slots()['total']==0
    m.claim(out.with_name(out.name+'.claim.json'),'own_equal60_CPU');torch.set_num_threads(4);cfg=m.config();manifest=m.dataset.manifest();data,val=m.load('train'),m.load('val');old=m.s.read(m.frozen.CPU/'checks.json');assert old['passed'] and old['sources']==m.frozen.sources();m.engine.verify_sources(old['sources'],old['git_commit'])
    assert len(data[1])==761 and len(val[1])==157 and sum(manifest['supports']['test'])==1833
    assert len({i for role in ('train','val','test') for i in manifest['split_ids'][role]})==2751
    ref=abo_reference();torch.manual_seed(17);e=torch.rand(2,13,17,requires_grad=True);o=torch.rand_like(e,requires_grad=True);a=torch.tensor(-.4,requires_grad=True);ee=e.detach().clone().requires_grad_();oo=o.detach().clone().requires_grad_();aa=a.detach().clone().requires_grad_();w=torch.randn_like(e);x=fuse(e,o,a);y=ref(ee,oo,aa,bounds=(.7001,.95));gx=torch.autograd.grad((x*w).sum(),(e,o,a));gy=torch.autograd.grad((y*w).sum(),(ee,oo,aa));assert torch.equal(x,y) and all(torch.equal(v,u) for v,u in zip(gx,gy))
    checks=[]
    for depth in (2,4,6):
        initial=[]
        for rho in (.3,0.):
            m.engine.seed(17);model=m.RegularizedMoE(depth,rho);initial.append(m.s.digest(model.state_dict()));params=dict(model.named_parameters());assert model.stages==['expert','global']*(depth//2) and len(params)=={2:12,4:21,6:30}[depth]
            assert sum(p.numel() for n,p in params.items() if n=='raw_router' or n.startswith('phase_sets.'))==50176+429188*(depth//2) and sum(p.numel() for n,p in params.items() if n!='raw_router' and not n.startswith('phase_sets.'))==2570+10*depth
            with torch.no_grad():
                black,_=model(m.b.encode(torch.zeros(1,3,150,150,dtype=torch.uint8),None,100));assert torch.allclose(black,torch.full_like(black,.1),atol=1e-8)
                prob,_=model(m.b.encode(data[0][:1],None,100));features=model.readout['features'].numpy().astype(np.float64);W=model.linear.weight.detach().numpy().astype(np.float64);bias=model.linear.bias.detach().numpy().astype(np.float64);z=features@W.T+bias;q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True);error=float(np.abs(q-prob.numpy()).max());assert error<1e-6 and model.trace==['router']+model.stages
            checks.append(dict(depth=depth,rho=rho,initial_digest=initial[-1],contract=model.contract(),numpy_maxabs=error));del model
        assert initial[0]==initial[1]
    out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,checks=checks,ABO_fusion_forward_gradient_exact=True,forward_calls=12,optimizer_steps=0,TEST_images_read=0,TEST_model_evaluations=0,old_split_proof_sha256=m.s.sha(m.frozen.CPU/'checks.json'),maximum_epochs=60),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
