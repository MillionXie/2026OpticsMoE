"""Independent structural and ABO-fusion parity admission, with integrity-only TEST pixels and no TEST model evaluation."""
import argparse,ast,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit as study
from caltech20_alt246_biasfree_model import alpha_value,fuse,features
from caltech20_alt246_genreg_model import RegularizedMoE as HybridMoE

def abo_reference():
    source=(study.ROOT/study.ABO).read_text();tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('rms','alpha_value','fuse')]
    assert len(functions)==3
    ns={'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(study.ROOT/study.ABO),'exec'),ns)
    return ns['fuse']

def independent_data_audit(cfg,m):
    # Independent adjacency-components implementation (producer uses disjoint sets).
    import hashlib,collections
    directory=Path(cfg['data_directory']);rows=study.s.read(directory/'group_identity.json');assert len(rows)==3935
    assert study.s.sha(directory/'group_identity.json')==m['group_identity_sha256']
    adjacency=[set() for _ in rows];index={}
    for i,r in enumerate(rows):
        for kind in ('original_hash','cache_hash'):
            token=(kind,r[kind]);j=index.get(token)
            if j is not None:assert r['label']==rows[j]['label'];adjacency[i].add(j);adjacency[j].add(i)
            else:index[token]=i
    remaining=set(range(len(rows)));groups={}
    while remaining:
        start=min(remaining);stack=[start];component=set()
        while stack:
            i=stack.pop()
            if i in component:continue
            component.add(i);stack.extend(adjacency[i]-component)
        remaining-=component;groups[start]=sorted(component)
    assigned={r:[] for r in ('train','val','test')}
    for label,name in enumerate(cfg['classes']):
        keys=sorted(k for k,v in groups.items() if rows[v[0]]['label']==label)
        seed=cfg['split_seed']+int(hashlib.sha256(name.encode()).hexdigest()[:6],16);np.random.default_rng(seed).shuffle(keys);n1=int(.8*len(keys));n2=int(.1*len(keys))
        for role,part in zip(assigned,(keys[:n1],keys[n1:n1+n2],keys[n1+n2:])):assigned[role].extend(i for k in part for i in groups[k])
    sets=[]
    with np.load(cfg['data'],allow_pickle=False) as z:
        for role,indices in assigned.items():
            indices.sort();names=[rows[i]['name'] for i in indices];assert names==m['split_ids'][role]==z[role+'_ids'].tolist();assert z[role+'_images'].shape==(len(indices),150,150,3)
            assert z[role+'_labels'].tolist()==[rows[i]['label'] for i in indices] and np.bincount(z[role+'_labels'],minlength=20).tolist()==m['supports'][role]
            for i,image in zip(indices,z[role+'_images']):assert hashlib.sha256(image.tobytes()).hexdigest()==rows[i]['cache_hash']
            sets.append({(kind,rows[i][kind]) for i in indices for kind in ('original_hash','cache_hash')})
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i)) and study.s.sha(cfg['data'])==cfg['cache_sha256']
    # Decode original selected photos again for integrity, never model evaluation.
    decoded,counts=study.dataset.archive_rows();assert [{k:v for k,v in r.items() if k!='image'} for r in decoded]==rows
    assert sorted(sorted((c for c in counts if c not in cfg['excluded_categories']),key=lambda c:(-counts[c],c))[:20])==cfg['classes']
    return dict(passed=True,adjacency_components=len(groups),records=len(rows),cross_role_exact_pixel_intersections=0,subject_independence_proven=False,all_role_pixels_read_for_integrity=True,model_test_evaluations=0)

def check(out):
    s=study.s;assert out==study.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    study.slots();study.claim(out.with_name(out.name+'.claim.json'),'own_cpu_structural_and_ABO_math')
    torch.set_num_threads(4);cfg=study.config();m=study.dataset.manifest();data=study.load('train');val=study.load('val')
    assert len(data[1])==cfg['training_samples'] and len(val[1])==len(m['split_ids']['val']) and len(data[1])>3000
    data_audit=independent_data_audit(cfg,m)
    # The failed v1 gate exposed a null surrogate bias direction: detached
    # branch RMS followed by final LayerNorm removes a constant bias gradient.
    # Remove that redundant parameter, rather than weakening the gate.
    from caltech20_alt246_model import SmallElectronicResidual as OldResidual
    torch.manual_seed(17);old_block=OldResidual();x=torch.rand(2,32,32)+.2;o=torch.rand(2,32,32)+.1
    outfield=fuse(old_block(x,32),o,torch.tensor(-.4));f=features(outfield);weight=torch.randn_like(f);(f*weight).sum().backward()
    bias_norm=float(old_block.local.bias.grad.norm());weight_norm=float(old_block.local.weight.grad.norm())
    assert bias_norm<1e-4*weight_norm and bias_norm<1e-5,(bias_norm,weight_norm)
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
            assert sum(p.numel() for n,p in names.items() if n!='raw_router' and not n.startswith('phase_sets.'))==5140+10*depth
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
    out.mkdir();s.save(out/'checks.json',dict(study.identity(),passed=True,checks=checks,abo_parity=parity,old_null_surrogate_bias_norm=bias_norm,old_nonzero_weight_norm=weight_norm,engineering_fix='remove bias parameter; preserve RMS math and numeric thresholds',abo_reference_sha256=s.sha(study.ROOT/study.ABO),alpha_bounds_proven=True,data_audit=data_audit,train_support=m['supports']['train'],val_support=m['supports']['val'],forward_calls=forward_calls,optimizer_steps=0,test_model_evaluations=0,test_pixels_read_for_integrity=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        dest=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not dest.exists():study.s.save(dest,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
