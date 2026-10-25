"""Training-only relational distillation, without student projections or new layers.

Distance/angle relations follow Park et al., CVPR 2019 (RKD).
Teacher and student feature dimensions need not be equal. Relations are fitted
within each sensor domain so raw sensor offsets are not treated as semantics.
"""
import torch
from torch.nn import functional as F
from .augmented_teacher import tensor_images


@torch.no_grad()
def teacher_representations(model, data, device, batch=128):
    model.eval(); logits=[]; embeddings=[]
    for i in range(0,len(data['labels']),batch):
        x=tensor_images(data['images'][i:i+batch],device)
        x=(x.float()/255.-model.mean)/model.std
        x=model.stem(x);x=x+model.res1(x);x=model.down(x);x=x+model.res2(x)
        embedding=x.mean((2,3))
        embeddings.append(embedding.cpu());logits.append(model.head(embedding).cpu())
    return torch.cat(logits),torch.cat(embeddings)


def relations(x):
    # Gram distance avoids cdist's singular diagonal gradient at zero.
    delta=x[:,None,:]-x[None,:,:]
    n=len(x);off=~torch.eye(n,device=x.device,dtype=torch.bool)
    distance=delta.square().sum(-1).clamp_min(1e-12).sqrt()
    distance=distance/distance[off].mean().clamp_min(1e-6)
    unit=F.normalize(delta,dim=-1,eps=1e-6)
    angle=torch.einsum('ijd,ikd->ijk',unit,unit)
    triples=off[:,:,None]&off[:,None,:]&off[None,:,:]
    return distance[off],angle[triples]


def relational_losses(student,teacher,domains):
    """Equal sensor-domain means; gradients only flow to original CCD features."""
    ds=[];angles=[]
    for domain in domains.unique():
        mask=domains==domain
        if int(mask.sum())<3:continue
        s=student[mask];t=teacher[mask].detach()
        sd,sa=relations(s);td,ta=relations(t)
        ds.append(F.smooth_l1_loss(sd,td));angles.append(F.smooth_l1_loss(sa,ta))
    if not ds:return student.sum()*0,student.sum()*0
    return torch.stack(ds).mean(),torch.stack(angles).mean()
