"""Training-only product-gallery objectives; no new inference module or teacher."""
import torch
from torch.nn import functional as F


def product_bank(features, samples):
    """Same view-normalize/mean/normalize rule as evaluation, TRAIN products only."""
    if len(samples)!=len(features) or any(s.split!='train' for s in samples):
        raise ValueError('Retrieval training bank requires aligned training-only samples')
    products=sorted({s.product_id for s in samples})
    positions={p:i for i,p in enumerate(products)}
    ids=torch.tensor([positions[s.product_id] for s in samples],device=features.device)
    views=F.normalize(features.detach().float(),dim=-1)
    centers=[];labels=[]
    for i,product in enumerate(products):
        categories={s.category_id for s in samples if s.product_id==product}
        if len(categories)!=1:raise ValueError('Product has conflicting categories')
        centers.append(F.normalize(views[ids==i].mean(0),dim=0));labels.append(categories.pop())
    return torch.stack(centers),torch.tensor(labels,device=features.device),ids


def gallery_loss(query, labels, own_product, bank, bank_labels, temperature=.10, margin=.08, class_balance=False):
    """Multi-positive retrieval NLL and top-negative margin, excluding own product.

    All other same-category products are relevant, never just the same SKU.
    The bank is detached and refreshed by the caller; test labels never enter it.
    """
    query=F.normalize(query.float(),dim=-1)
    score=query@bank.detach().float().T
    valid=torch.arange(len(bank),device=query.device)[None]!=own_product[:,None]
    positive=labels[:,None].eq(bank_labels[None]) & valid
    negative=~labels[:,None].eq(bank_labels[None]) & valid
    if not bool((positive.any(1)&negative.any(1)).all()):
        raise ValueError('Every query needs another positive product and a negative product')
    logits=score/temperature
    if class_balance:
        # Expanded TRAIN gallery has unequal products/class; evaluation has 12
        # each. Average exp-scores within classes instead of rewarding density.
        # Count after excluding the own product, and avoid autocast count rounding.
        columns=bank_labels[None].expand(len(query),-1)
        counts=torch.zeros(len(query),int(bank_labels.max())+1,device=query.device,dtype=torch.long)
        counts.scatter_add_(1,columns,valid.long())
        logits=logits-counts.gather(1,columns).clamp_min(1).float().log()
    nll=(logits.masked_fill(~valid,-torch.inf).logsumexp(1)
         -logits.masked_fill(~positive,-torch.inf).logsumexp(1)).mean()
    best_positive=score.masked_fill(~positive,-torch.inf).amax(1)
    best_negative=score.masked_fill(~negative,-torch.inf).amax(1)
    ranking=F.softplus((best_negative-best_positive+margin)/temperature).mean()*temperature
    hit=bank_labels[score.masked_fill(~valid,-torch.inf).argmax(1)].eq(labels).float().mean()
    return nll,ranking,hit.detach()


def readout_polish(epoch, epochs, cfg):
    # Tiny smoke runs retain joint training, rather than freezing everything.
    return epochs>cfg['readout_polish_epochs'] and epoch>epochs-cfg['readout_polish_epochs']
