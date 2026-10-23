"""TRAIN source-family pairs and bounded original-feature contrastive loss.

Training-only loss/batch order. No projection head or inference graph change.
"""
import numpy as np


def family_paired_order(labels,families,rng):
    labels=np.asarray(labels);families=np.asarray(families)
    assert labels.shape==families.shape==(1600,) and np.bincount(labels,minlength=20).tolist()==[80]*20
    pairs={}
    for k in range(20):
        members=np.flatnonzero(labels==k)
        for attempt in range(100):
            value=rng.permutation(members).reshape(40,2)
            if np.all(families[value[:,0]]!=families[value[:,1]]):pairs[k]=value;break
        else:raise AssertionError('Cannot form independent TRAIN family pairs')
    batches=[]
    for slot in range(40):
        classes=rng.permutation(20)
        for begin in range(0,20,4):batches.append(np.concatenate([pairs[k][slot] for k in classes[begin:begin+4]]))
    order=np.concatenate(batches);assert np.array_equal(np.sort(order),np.arange(1600))
    return order


def loss(features,labels):
    import torch
    from torch.nn import functional as F
    assert features.ndim==2 and len(features)==8 and labels.shape==(8,)
    value=F.normalize(features.float(),dim=1);similarity=value@value.T/.2
    diagonal=torch.eye(8,dtype=torch.bool,device=labels.device);positive=(labels[:,None]==labels[None,:])&~diagonal
    assert torch.all(positive.sum(1)==1)
    denominator=torch.logsumexp(similarity.masked_fill(diagonal,-torch.inf),dim=1)
    return (denominator-(similarity*positive).sum(1)).mean()


def selftest():
    import torch
    y=torch.arange(4).repeat_interleave(2);feature=torch.eye(4).repeat_interleave(2,0).requires_grad_()
    value=loss(feature,y);value.backward();assert value<.05 and torch.isfinite(feature.grad).all()
    constant=loss(torch.ones(8,4),y);assert abs(float(constant)-np.log(7))<1e-6
    return dict(passed=True,finite_original_feature_gradient=True,TRAIN_positive_pair_loss_check=True,new_inference_layers=0,TEST_used=False)
