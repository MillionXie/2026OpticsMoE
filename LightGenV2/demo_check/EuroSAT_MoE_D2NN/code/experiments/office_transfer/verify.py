"""Real-image GPU checks using only train/validation images."""
import json
import torch
from torch.nn import functional as F
from .data import prepare,Images,atomic_json,signature,plan
from .runtime import ROOT,setup,build,verify_source
from .train import optimizer,mode_for
from experiments.vision_transfer import model as M,engine as E
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything

def verify():
    verify_source();split=prepare();loaded,s=setup(42,ROOT/'runs/preflight')
    images=[Images(split,d,'validation')[i][0] for d in (0,1) for i in range(10)]
    y=torch.tensor([Images(split,d,'validation')[i][1] for d in (0,1) for i in range(10)],device='cuda')
    inputs=E.base._prepare(loaded,images,s);report={};electronics=None
    for arch in ('moe','d2nn'):
        seed_everything(42);r,h=build(loaded,s,arch);initial=M.clone(r,h)
        resource=M.resource_report(r,h)
        if electronics is None:electronics=resource['electronics_initial_sha256']
        else:assert electronics==resource['electronics_initial_sha256']
        records=[]
        for stage,domain,epoch,variant in [('source',0,1,'reserved'),('source',0,4,'reserved'),('source',0,10,'reserved'),('source',1,4,'reserved'),('transfer',1,1,'reserved'),('transfer',1,1,'all')]:
            M.restore(r,h,initial);opt=optimizer(r,h,stage,domain,epoch,'pilot',variant);mode_for(loaded,r,h);frozen=M.digest(r,h,True)
            with E.autocast(loaded,s):logits=M.predict(loaded,r,h,inputs);loss=F.cross_entropy(logits,y)
            grads=E.phase_gradients(r,h,loss) if epoch!=1 or stage=='transfer' else {}
            for n,g in grads.items():assert g['finite'] and g['l2']>0,(arch,n,g)
            for q in M.routes(r).values():
                assert bool((q['weights']>0).all()) and bool(q['selected_mask'].all())
                torch.testing.assert_close(q['weights'].square().sum(1),torch.ones(len(y),device='cuda'),atol=1e-6,rtol=1e-6)
            loss.backward();opt.step();assert M.digest(r,h,True)==frozen
            records.append(dict(stage=stage,domain=domain,epoch=epoch,variant=variant,frozen_unchanged=True,phase_gradients=grads))
        with torch.no_grad(),E.autocast(loaded,s):
            one=M.predict(loaded,r,h,inputs).float();two=M.predict(loaded,r,h,E.base._prepare(loaded,list(reversed(images)),s)).float().flip(0)
        torch.testing.assert_close(one,two,atol=.02,rtol=.002);assert torch.equal(one.argmax(1),two.argmax(1))
        assert not hasattr(r,'language_surrogate') and loaded.source_metadata['vision_transformer_parameters']==0
        report[arch]=dict(checks=records,resources=resource,mixed_domain_order_invariant=True)
        r.close();del r,h
        print('[preflight]',arch,'passed',flush=True)
    report.update(status='passed',plan_sha256=signature(plan()),split_sha256=split['split_sha256'],test_read=False)
    atomic_json(ROOT/'runs/preflight.json',report)
    return report

if __name__=='__main__':verify()
