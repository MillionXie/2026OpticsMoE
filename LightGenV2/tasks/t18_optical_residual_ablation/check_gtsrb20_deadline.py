"""New grouped-data and actual hybrid contract admission; no TEST forward."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import gtsrb20_deadline as run
from check_caltech20_alt246_richsplit import abo_reference
from caltech20_alt246_biasfree_model import fuse,alpha_value
def check(out):
    b=run.b;s=run.s;assert out==run.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and b.slots()['total']==0;run.claim(out.with_name(out.name+'.claim.json'),'GTSRB20_own_CPU_contract');torch.set_num_threads(4);cfg=run.config();m=run.dataset.manifest();data,val=b.load('train'),b.load('val');directory=Path(cfg['data_directory'])
    records=s.read(directory/'selected_records.json');groups=s.read(directory/'training_group_records.json');assert s.sha(directory/'selected_records.json')==m['selected_records_sha256'] and s.sha(directory/'training_group_records.json')==m['training_groups_sha256']
    counts={int(k):v for k,v in m['training_raw_class_counts'].items()};assert sorted(sorted(counts,key=lambda k:(-counts[k],k))[:20])==m['original_class_ids'];assert len(data[1])==4000 and m['supports']['train']==[200]*20
    sets=[]
    with np.load(cfg['data'],allow_pickle=False) as z:
        for role in ('train','val','test'):
            assert z[role+'_ids'].tolist()==m['split_ids'][role] and np.bincount(z[role+'_labels'],minlength=20).tolist()==m['supports'][role]
            assert [r['label'] for r in records[role]]==z[role+'_labels'].tolist();sets.append({r['cache_sha256'] for r in records[role]})
            if role!='test':
                import hashlib
                assert [hashlib.sha256(v.tobytes()).hexdigest() for v in z[role+'_images']]==[r['cache_sha256'] for r in records[role]]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    for kind in ('track','component'):assert not {r[kind] for r in records['train']}&{r[kind] for r in records['val']}
    # Rebuild fixed component assignments independently from saved all-TRAIN
    # records, verify role ownership and sampling for all20 classes.
    for label,original in enumerate(m['original_class_ids']):
        components=sorted({r['component'] for r in groups if r['label']==label});np.random.default_rng(cfg['split_seed']+original).shuffle(components);cut=int(.8*len(components))
        for role,keys,cap in [('train',components[:cut],200),('val',components[cut:],100)]:
            candidates=[r for r in groups if r['label']==label and r['component'] in keys];assert all(r['assigned_role']==role for r in candidates)
            expected=run.dataset.sample(candidates,cap,cfg['split_seed']+original+(1 if role=='train' else 2));actual=[r for r in records[role] if r['label']==label];assert [r['path'] for r in expected]==[r['path'] for r in actual]
    reference=abo_reference();torch.manual_seed(17);e=torch.rand(2,13,17,requires_grad=True);o=torch.rand_like(e,requires_grad=True);a=torch.tensor(-.4,requires_grad=True);ee=e.detach().clone().requires_grad_();oo=o.detach().clone().requires_grad_();aa=a.detach().clone().requires_grad_();w=torch.randn_like(e);x=fuse(e,o,a);y=reference(ee,oo,aa,bounds=(.7001,.95));gx=torch.autograd.grad((x*w).sum(),(e,o,a));gy=torch.autograd.grad((y*w).sum(),(ee,oo,aa));assert torch.equal(x,y) and all(torch.equal(v,u) for v,u in zip(gx,gy))
    for raw in (-100.,0.,100.):assert .7<float(alpha_value(torch.tensor(raw)))<=.95
    checks=[];forward=0
    for depth in (2,4,6):
        initial=[]
        for rho in (.3,0.):
            run.engine.seed(17);model=b.RegularizedMoE(depth,rho);initial.append(s.digest(model.state_dict()));params=dict(model.named_parameters());assert model.stages==['expert','global']*(depth//2) and len(params)=={2:12,4:21,6:30}[depth]
            assert sum(p.numel() for n,p in params.items() if n=='raw_router' or n.startswith('phase_sets.'))==50176+429188*(depth//2) and sum(p.numel() for n,p in params.items() if n!='raw_router' and not n.startswith('phase_sets.'))==5140+10*depth
            with torch.no_grad():
                black,_=model(b.encode(torch.zeros(1,3,150,150,dtype=torch.uint8),None,100));forward+=1;assert torch.allclose(black,torch.full_like(black,.05),atol=1e-8) and model.trace==['router']+model.stages
                p,_=model(b.encode(data[0][:1],None,100));forward+=1;f=model.readout['features'].numpy().astype(np.float64);W=model.linear.weight.detach().numpy().astype(np.float64);bias=model.linear.bias.detach().numpy().astype(np.float64);logits=f@W.T+bias;q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True);err=float(np.abs(q-p.numpy()).max());assert err<=1e-6 and model.trace==['router']+model.stages
            checks.append(dict(depth=depth,rho=rho,initial_digest=initial[-1],contract=model.contract(),black20_uniform=True,linear_numpy_maxabs=err,trace=model.trace));del model
        assert initial[0]==initial[1]
    out.mkdir();s.save(out/'checks.json',dict(run.identity(),passed=True,checks=checks,ABO_forward_gradient_exact=True,train_support=m['supports']['train'],val_support=m['supports']['val'],test_support_metadata=m['supports']['test'],TRAIN_VAL_track_intersections=0,declared_cross_role_cached_pixel_intersections=0,TEST_images_read=0,TEST_model_evaluations=0,forward_calls=forward,optimizer_steps=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():run.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
