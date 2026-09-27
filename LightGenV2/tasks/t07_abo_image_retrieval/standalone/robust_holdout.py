"""Continuation holdout and paired stability; no TEST optimization signal."""
import hashlib
import torch
from torch.nn import functional as F

def split_train(groups):
    by={}
    for row in groups['train']:
        assert row['split']=='train'
        by.setdefault(row['product_id'],[]).append(row)
    fitting,held=[],[]
    for key in sorted(by):
        rows=sorted(by[key],key=lambda r:hashlib.sha256(('robust-val-73:'+r['sample_id']).encode()).hexdigest())
        assert len(rows)==8
        fitting.extend(rows[:6]);held.extend(rows[6:])
    fit_ids={r['sample_id'] for r in fitting};held_ids={r['sample_id'] for r in held}
    assert len(fit_ids)==1200 and len(held_ids)==400 and not fit_ids&held_ids
    assert not (fit_ids|held_ids)&{r['sample_id'] for r in groups['query']}
    selection=dict(gallery=[dict(r,split='gallery',source_split='train') for r in fitting],query=[dict(r,split='query',source_split='train') for r in held],train=fitting)
    return dict(groups,train=fitting),selection,dict(fitting_ids=sorted(fit_ids),validation_ids=sorted(held_ids),scope='Continuation-only holdout: these photos were seen by the warm-start model in earlier runs; not pristine independent validation. No holdout gradients in this run; original800TEST excluded from epoch selection.')

def paired_loss(z,clean_z,bank,excluded,model,clean_routes,count):
    logits=F.normalize(z[:count].float(),dim=-1)@F.normalize(bank.detach().float(),dim=-1).T/.1
    target=F.normalize(clean_z[:count].detach().float(),dim=-1)@F.normalize(bank.detach().float(),dim=-1).T/.1
    if excluded is not None:
        logits=logits.masked_fill(excluded,-1e4);target=target.masked_fill(excluded,-1e4)
    relation=F.kl_div(logits.log_softmax(-1),target.softmax(-1),reduction='batchmean')
    embedding=(1-F.cosine_similarity(z.float(),clean_z.detach().float(),dim=-1)).mean()
    routes=[]
    for name in ('vision','language'):
        p=getattr(model,name).optics.router.last['probabilities'].float().clamp_min(1e-7)
        q=clean_routes[name].detach().float().clamp_min(1e-7)
        routes.append(F.kl_div(p.log(),q,reduction='batchmean'))
    return relation+.1*embedding,torch.stack(routes).mean()

def score(metrics,clean_floor,eligible):
    clean=metrics['test'];noisy=metrics['validation_noisy']
    return (int(eligible),int(clean['hit_at_1']>=clean_floor),noisy['hit_at_1'],clean['hit_at_1'],clean['map_at_10'])
