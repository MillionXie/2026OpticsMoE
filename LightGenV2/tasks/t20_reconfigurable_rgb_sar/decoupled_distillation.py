"""TRAIN-only logit DKD (Zhao et al., CVPR 2022); no inference modules.

Separately weights target-vs-rest KL and KL conditional on non-target classes.
Teacher-wrong rows retain their TRAIN CE but do not contribute distillation.
"""
import torch
from torch.nn import functional as F


def components(logits,targets,labels,temperature):
    if logits.shape!=targets.shape or logits.ndim!=2 or logits.shape[1]<2:
        raise ValueError('Matching class logits with at least two classes required')
    targets=targets.detach();s=logits/temperature;t=targets/temperature
    mask=F.one_hot(labels,num_classes=s.shape[1]).bool()
    sn=s[~mask].reshape(len(labels),-1);tn=t[~mask].reshape(len(labels),-1)
    sb=torch.stack((s[mask],torch.logsumexp(sn,1)),1).log_softmax(1)
    tb=torch.stack((t[mask],torch.logsumexp(tn,1)),1).log_softmax(1)
    target=F.kl_div(sb,tb.exp(),reduction='none').sum(1)*temperature**2
    other=F.kl_div(sn.log_softmax(1),tn.softmax(1),reduction='none').sum(1)*temperature**2
    return target,other


def supervised_decoupled(logits,targets,labels,temperature,target_weight=1.,other_weight=4.):
    ce=F.cross_entropy(logits,labels,reduction='none',label_smoothing=.02)
    a,b=components(logits,targets,labels,temperature)
    gate=(targets.detach().argmax(1)==labels).to(logits.dtype)
    return ce+gate*(target_weight*a+other_weight*b)
