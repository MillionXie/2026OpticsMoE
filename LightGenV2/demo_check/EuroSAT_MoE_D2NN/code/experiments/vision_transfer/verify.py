"""Real GPU contract checks, before formal training; never reads test images."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .settings import load_settings
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import prepare_cifar10
from .backbone import load_backbone
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything
from . import model as M
from .engine import base,autocast,phase_gradients,routing_loss
from .data import atomic_json,transform_one,CORRUPTIONS

def main():
    from .protocol import require_authorization, output_root, policy_sha256, PROJECT_ROOT
    require_authorization()
    s=load_settings('experiments/vision_transfer/config.yaml');torch.set_num_threads(8)
    s.output_dir=output_root()
    bundle=prepare_cifar10(s,persist=False)
    loaded=load_backbone(s,torch.device('cuda'))
    images=[bundle.validation[i][0] for i in range(60)]
    labels=torch.tensor([bundle.validation[i][1] for i in range(60)],device='cuda')
    inputs=base._prepare(loaded,images,s)
    report={};electronic=None
    for arch in ('moe','d2nn'):
        seed_everything(42);r,h=M.build(loaded,s,arch)
        try:
            assert not hasattr(r,'language_surrogate')
            assert loaded.source_metadata['language_parameters']==0
            assert not any('attention' in n or 'language' in n for n,_ in loaded.model.named_parameters())
            assert set(inputs)=={'pixel_values','image_grid_thw'}
            resource=M.resource_report(r,h)
            if electronic is None:electronic=resource['electronics_initial_sha256']
            else:assert resource['electronics_initial_sha256']==electronic,'Electronic initialization mismatch'
            initial=M.clone(r,h);records=[]
            for task,epoch,variant in [('A',1,'reserved'),('A',6,'reserved'),('A',41,'reserved'),('B',1,'reserved'),('B',1,'all'),('B',11,'reserved')]:
                M.restore(r,h,initial)
                opt=M.optimizer_for(r,h,task,epoch,variant)
                M.set_mode(loaded,r,h,True,task,epoch)
                frozen=M.digest(r,h,True)
                domain=torch.arange(60,device='cuda')%2 if task=='B' else torch.zeros(60,device='cuda',dtype=torch.long)
                M.force_route(r,None)
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                with autocast(loaded,s):
                    logits=M.classification_logits(loaded.model,r,h,inputs)[0]
                    ce=F.cross_entropy(logits,labels);aux,_=routing_loss(r,domain)
                    loss=ce+(0 if aux is None else aux)
                grads=phase_gradients(r,h,ce)
                if epoch!=11:
                    for name,g in grads.items():
                        assert g['l2']>0,(arch,task,name,'zero CE gradient')
                for mod,q in M.routes(r).items():
                    torch.testing.assert_close(q['weights'].detach().square().sum(1),torch.ones(60,device='cuda'),atol=1e-6,rtol=1e-6)
                    assert bool(q['selected_mask'].all())
                    assert bool((q['weights']>0).all())
                    torch.testing.assert_close(q['weights'],M.dense_weights(q['probabilities']))
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                assert M.digest(r,h,True)==frozen,'Frozen state changed after AdamW'
                if arch=='d2nn':
                    for _,sur in M.surrogates(r):assert float(sur.core.optical_branch.core.last_power_relative_error.max())<1e-5
                records.append(dict(task=task,epoch=epoch,variant=variant,ce=float(ce),ce_gradients=grads,frozen_unchanged=True,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-start))
                print('[contract]',arch,task,epoch,variant,'passed',flush=True)
            # Same model, no routing mask, paired clean/corrupted images permuted.
            pair=images[:10]+[__import__('PIL').Image.fromarray(transform_one((np.asarray(im),0,2,777+i))) for i,im in enumerate(images[:10])]
            order=list(reversed(range(20)))
            inp1=base._prepare(loaded,pair,s);inp2=base._prepare(loaded,[pair[i] for i in order],s)
            with torch.no_grad(),autocast(loaded,s):
                one=M.predict(loaded,r,h,inp1).float();two=M.predict(loaded,r,h,inp2).float()[torch.tensor(order,device='cuda')]
            torch.testing.assert_close(one,two,rtol=2e-3,atol=2e-2)
            assert torch.equal(one.argmax(1),two.argmax(1)),'Batch order changes predictions'
            report[arch]=dict(resource=resource,checks=records,mixed_domain_order_invariant=True)
        finally:r.close()
    # Deterministic severity 1-3 API coverage, all 15 generators; test data unused.
    for k in range(15):
        for sev in (1,2,3):
            job=(np.asarray(images[0]),k,sev,8120+k*10+sev)
            assert np.array_equal(transform_one(job),transform_one(job)),(CORRUPTIONS[k],sev)
    report['corruption_reproducibility']='all 15 types x severities 1-3 passed'
    report['status']='passed'
    report['policy_sha256']=policy_sha256()
    report['review_manifest_sha256']=hashlib.sha256((PROJECT_ROOT/'REVIEW_MANIFEST.json').read_bytes()).hexdigest()
    atomic_json(Path(str(s.output_dir)+'_smoke')/'contract_tests.json',report)
    print('[contracts] ALL PASSED',flush=True)

if __name__=='__main__':main()
