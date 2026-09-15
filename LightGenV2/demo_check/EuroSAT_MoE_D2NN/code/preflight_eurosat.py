"""Meaningful GPU and split checks; all updates are discarded before training."""
import gc,json,time
import torch
from torch.nn import functional as F
from eurosat_data import *
from eurosat_runtime import *

def main(architecture_only=False):
    if architecture_only:
        raise ValueError('EuroSAT requires the full verified paired dataset preflight')
    (ROOT/'runs').mkdir(parents=True,exist_ok=True)
    source=verify_source()
    if not architecture_only:
        checks=json.loads((ROOT/'DATA_CHECKS.json').read_text());assert checks['passed'] and checks['images']==len(RECORDS)
        assert checks['split_sha256']==PLAN['split_records_sha256']
    if not architecture_only and (ROOT/'runs/preflight.json').exists():
        previous=json.loads((ROOT/'runs/preflight.json').read_text())
        if previous['passed'] and previous['source_sha256']==source:return
    started=time.time();reports={};electronic=None;initial_d2=None
    for domain in ('A','B'):
        for partition in ('train','validation','test'):
            rs=[r for r in RECORDS if r['domain']==domain and r['split']==partition]
            assert len(rs)==PLAN['counts'][domain][partition]
            assert {r['label'] for r in rs}==set(range(10))
    assert checks['paired_split_consistent'] and checks['spatial_groups_disjoint'] and checks['cross_split_minimum_center_distance_m']>=3000
    for partition in ('train','validation','test'):
        a={r['pair_id'] for r in RECORDS if r['domain']=='A' and r['split']==partition}
        b={r['pair_id'] for r in RECORDS if r['domain']=='B' and r['split']==partition}
        assert a==b
    # Same index and augmented-pixel output independent of global model RNG consumption.
    for stage in STAGES:
        one=next(index_batches(stage,1));torch.rand(123)
        two=next(index_batches(stage,1));assert batch_hash(one)==batch_hash(two)
        if not architecture_only:
            for (d,i,m),(e,j,n) in zip(one[:3],two[:3]):
                a=d.image(i,m[3]).tobytes();torch.rand(27);b=e.image(j,n[3]).tobytes();assert a==b
    images=[];domain=[];labels=[]
    for d in ('A','B'):
        ds=Images(d,'train')
        for c in (0,5):
            idx=ds.labels.index(c)
            if architecture_only:
                import numpy as np
                from PIL import Image
                im=Image.fromarray(np.random.default_rng(c+ds.domain*7).integers(0,256,(224,224,3),dtype=np.uint8))
            else:im=ds.image(idx)
            images.append(im);labels.append(c);domain.append(ds.domain)
    for architecture in ('moe','d2nn'):
        loaded,s,r,h=build(architecture,ROOT/'runs/preflight')
        try:
            report=M.resource_report(r,h);report['checks']=[];reports[architecture]=report
            counts=report['parameter_counts'];assert sum(counts.values())==PLAN['training_parameters'][architecture]
            assert counts['electronic']==702822 and counts['head']==4618
            assert report['backbone']['frozen_parameters']==3933184
            if electronic is None:electronic=M.electronic_digest(r,h)
            else:assert electronic==M.electronic_digest(r,h),'Electronic/head initialization differs'
            initial=stamped(r,h,stage='initial',epoch=0)
            E.save(ROOT/'runs'/f'initial_{architecture}.pt',initial)
            report['initial_state_sha256']=C.state_digest(initial)
            inputs=E.base._prepare(loaded,images,s);ys=torch.tensor(labels,device=loaded.device);dom=torch.tensor(domain,device=loaded.device)
            conditions=[('shared',1),('shared',5),('expert_A',1),('expert_B',1),('router',1)] if architecture=='moe' else [('A_only',1),('A_only',5)]
            for stage,epoch in conditions:
                M.restore(r,h,initial);opt=optimizer(r,h,architecture,stage,epoch,STAGES.get(stage,{'epochs':70})['epochs'])
                frozen=M.digest(r,h,True);before={n:p.detach().cpu().clone() for n,p in M.named(r,h).items() if p.requires_grad}
                mode='automatic' if architecture=='d2nn' or stage=='router' else 'uniform' if stage=='shared' else 'isolated'
                # For expert isolation train only the matching domain, as in real training.
                use=0 if stage=='expert_A' else 1 if stage=='expert_B' else None
                ims=images if use is None else [im for im,d in zip(images,domain) if d==use]
                inp=inputs if use is None else E.base._prepare(loaded,ims,s)
                y=ys if use is None else ys[dom==use];d=dom if use is None else dom[dom==use]
                with E.autocast(loaded,s):
                    logits=forward(loaded,r,h,inp,mode,d if mode=='isolated' else None);loss=F.cross_entropy(logits,y,label_smoothing=.05)
                assert logits.shape==(len(y),10) and bool(torch.isfinite(logits).all())
                grads=E.phase_gradients(r,h,loss) if any(n.endswith(('raw_phase','raw_router_phase')) for n in before) else {}
                assert all(g['finite'] and g['l2']>0 for g in grads.values()),grads
                if architecture=='moe':
                    q=M.routes(r)['vision'];assert torch.allclose(q['weights'].square().sum(1),torch.ones(len(y),device=loaded.device),atol=1e-5)
                    if mode=='automatic':assert bool((q['weights']>0).all())
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                assert frozen==M.digest(r,h,True)
                changed=[n for n,p in M.named(r,h).items() if n in before and not torch.equal(before[n],p.detach().cpu())];assert changed
                report['checks'].append(dict(stage=stage,epoch=epoch,phase_gradients=grads,changed_parameters=len(changed),frozen_unchanged=True))
                opt.zero_grad(set_to_none=True)
            M.restore(r,h,initial)
            # Exercise the approved effective batch size, not only the tiny gradient probe.
            big_images=(images*15)[:60];big_y=torch.tensor((labels*15)[:60],device=loaded.device)
            big_inputs=E.base._prepare(loaded,big_images,s)
            opt=optimizer(r,h,architecture,'shared',5,40)
            torch.cuda.reset_peak_memory_stats();times=[]
            for iteration in range(3):
                opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
                with E.autocast(loaded,s):
                    z=forward(loaded,r,h,big_inputs,'uniform' if architecture=='moe' else 'automatic')
                    ce=F.cross_entropy(z,big_y,label_smoothing=.05)
                ce.backward();torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize();times.append(time.perf_counter()-t)
            report['batch60_probe']=dict(step_seconds=times,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),includes_preprocessing=False,discarded_updates=True)
            M.restore(r,h,initial)
        finally:r.close()
        del r,h,loaded,initial;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/'runs/architecture.json',reports)
    atomic(ROOT/('runs/architecture_probe.json' if architecture_only else 'runs/preflight.json'),dict(passed=True,architecture_only=architecture_only,source_sha256=source,split_sha256=PLAN['split_records_sha256'],
        matched_electronic_initial_sha256=electronic,sampling_checks=True,augmented_pixel_checks=not architecture_only,spatial_groups_and_pairs_disjoint=True,
        phase_gradient_and_frozen_state_checks=True,elapsed_sec=time.time()-started))
    print('PREFLIGHT_PASSED',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--architecture-only',action='store_true');main(p.parse_args().architecture_only)
