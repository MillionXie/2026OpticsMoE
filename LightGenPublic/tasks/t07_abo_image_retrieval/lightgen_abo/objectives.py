"""Training-only helpers; never add a branch to the inference network."""
import math
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageOps,ImageEnhance,ImageFilter

def parameter_kind(name):
    if 'optical_fusion_logit' in name:return 'alpha'
    if 'raw_router_phase' in name:return 'router'
    if 'optics.experts.' in name or name.endswith('optics.global_phase'):return 'phase'
    if name.startswith('readout.'):return 'readout'
    if any(name.startswith(m+'.'+a+'.') for m in ('vision','language') for a in ('input_adapter','input_norm','output_adapter')):return 'adapter'
    return 'electronic'

def group_kind(name):
    kind=parameter_kind(name)
    return 'optical_electronic' if kind=='electronic' and '.optics.' in name else kind

class CategoryProxies(nn.Module):
    """Training labels only; never used to restrict retrieval candidates."""
    def __init__(self, classes, dimension=64):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(classes,dimension)*.02)

    def forward(self,features):
        return 16*F.normalize(features.float(),dim=-1)@F.normalize(self.weight,dim=-1).T

def make_groups(samples):
    groups=defaultdict(lambda:defaultdict(list))
    for i,s in enumerate(samples):groups[s.category_id][s.product_id].append(i)
    if sorted(groups)!=list(range(len(groups))):raise ValueError('Labels must be consecutive')
    return dict(groups)

def sampled_indices(groups,classes_per_batch,products_per_class,rng):
    result=[]
    for category in rng.sample(list(groups),classes_per_batch):
        products=groups[category]
        for product in rng.sample(list(products),products_per_class):result.append(rng.choice(products[product]))
    rng.shuffle(result)
    return result

def optical_heads(classes):
    return nn.ModuleDict({m:nn.Sequential(nn.LayerNorm(384),nn.Linear(384,classes)) for m in ('vision','language')})

def optical_classification_loss(model,heads,labels):
    losses=[]
    for m in ('vision','language'):
        x=getattr(model,m).last_optical
        if x is None:raise RuntimeError('Optical auxiliary supervision needs optical branch enabled')
        pooled=torch.cat((x.float().mean(1),x.float().amax(1)),dim=-1)
        losses.append(torch.nn.functional.cross_entropy(heads[m](pooled),labels,label_smoothing=.05))
    return torch.stack(losses).mean()

def augment(image,rng,cfg):
    side=round(224*rng.uniform(cfg['minimum_crop_side_fraction'],1.))
    left,top=[rng.randint(0,224-side) for _ in range(2)]
    image=image.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.BICUBIC)
    if rng.random()<cfg['horizontal_flip_probability']:image=ImageOps.mirror(image)
    if rng.random()<.5:
        image=image.rotate(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees']),Image.Resampling.BICUBIC,fillcolor=(255,255,255))
    image=ImageEnhance.Brightness(image).enhance(rng.uniform(cfg['brightness_min'],cfg['brightness_max']))
    image=ImageEnhance.Contrast(image).enhance(rng.uniform(cfg['contrast_min'],cfg['contrast_max']))
    if rng.random()<cfg['blur_probability']:image=image.filter(ImageFilter.GaussianBlur(cfg['blur_radius']))
    return image

def phase_change(model,initial):
    result={}
    for name,p in model.named_parameters():
        if parameter_kind(name) not in ('phase','router'):continue
        raw=p.detach().cpu().float();before=initial[name].float()
        delta=2*torch.pi*(raw.sigmoid()-before.sigmoid())
        wrapped=torch.atan2(delta.sin(),delta.cos())
        result[name]=dict(raw_rms=float((raw-before).square().mean().sqrt()),phase_circular_rms_radians=float(wrapped.square().mean().sqrt()))
    return result

def phase_shuffle(model,seed=42):
    """Preserve each phase histogram, disturb its spatial pattern. Caller restores."""
    generator=torch.Generator().manual_seed(seed);saved={}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if parameter_kind(name) not in ('phase','router'):continue
            saved[name]=p.detach().clone()
            order=torch.randperm(p.numel(),generator=generator).to(p.device)
            p.copy_(p.flatten()[order].reshape_as(p))
    return saved

def restore_phase(model,saved):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in saved:p.copy_(saved[name])

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

def gallery_loss(query, labels, own_product, bank, bank_labels, temperature=.10, margin=.08):
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

def supcon(z, labels):
    z = F.normalize(z.float(),dim=-1)
    scores = z@z.T/.1
    diagonal = torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive = labels[:,None].eq(labels[None,:]) & ~diagonal
    logp = scores-scores.masked_fill(diagonal,-torch.inf).logsumexp(1,keepdim=True)
    return -(logp.masked_fill(~positive,0).sum(1)/positive.sum(1)).mean()

def regularization(model):
    balances, importances, hard, dc, operating = [],[],[],[],[]
    for modality in (model.vision,model.language):
        optics = modality.optics
        r = optics.router.last
        balances.append(r['balance']);importances.append(r['importance']);hard.append(r['hard_balance'])
        for raw in [*optics.experts,optics.global_phase]:
            phase = 2*torch.pi*raw.sigmoid()
            dc.append(phase.cos().mean().square()+phase.sin().mean().square())
        operating.append(F.smooth_l1_loss(optics.operating_losses[-1],torch.full_like(optics.operating_losses[-1],np.log(.25))))
    return .08*torch.stack(balances).mean()+.02*torch.stack(importances).mean()+.5*torch.stack(hard).mean()+.005*torch.stack(dc).mean()+.02*torch.stack(operating).mean()
