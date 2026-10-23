"""TRAIN-only soft-target family-deleted linear readout diagnostic.

An unchanged original 256-to-20 affine head can consume the returned W/b.
No TEST, hardware, Torch, model state, inference teacher or added layer.
"""
import math
import numpy as np
from lab_caltech20_TRAIN_group_ridge_20261024 import prepare,estimate,golden


def soft_targets(labels,teacher_logits):
    labels=np.asarray(labels,dtype=np.int64);t=np.asarray(teacher_logits,dtype=np.float64)
    assert t.shape==(len(labels),20) and np.isfinite(t).all()
    def probability(z):
        z=z-z.max(1,keepdims=True);v=np.exp(z);return v/v.sum(1,keepdims=True)
    raw=probability(t);gate=(raw.argmax(1)==labels)&(raw.max(1)>=.5)
    hard=np.eye(20)[labels];soft=probability(t/2)
    targets=hard.copy();targets[gate]=.5*hard[gate]+.5*soft[gate]
    assert np.allclose(targets.sum(1),1) and np.all(targets.argmax(1)==labels)
    return targets,gate


def fit(features,labels,prior_counts,family_ids,teacher_logits):
    x=np.asarray(features);labels=np.asarray(labels,dtype=np.int64);prior=np.asarray(prior_counts,dtype=float);families=np.asarray(family_ids)
    assert x.shape==(1600,256) and labels.shape==(1600,) and np.bincount(labels,minlength=20).tolist()==[80]*20
    assert len(prior)==20 and prior.sum()==3141 and len(np.unique(families))==1508
    assert max(np.unique(families,return_counts=True)[1])==2
    targets,gate=soft_targets(labels,teacher_logits);weights=.5+.5*np.sqrt(prior[labels]/prior.mean())
    system=prepare(x,targets,weights,families);scale=float(system['s'][0]**2);assert scale>0
    objective=lambda logratio:estimate(system,scale*math.exp(logratio))[3]
    logratio,mse,calls=golden(objective,math.log(1e-6),math.log(1e3));regularization=scale*math.exp(logratio)
    coef,bias,deleted,mse,dof=estimate(system,regularization)
    def nll(logscale):
        z=math.exp(logscale)*deleted;z=z-z.max(1,keepdims=True);values=np.log(np.exp(z).sum(1))-z[np.arange(1600),labels]
        return float(np.average(values,weights=weights))
    logs,nll_value,scale_calls=golden(nll,math.log(.1),math.log(1000.));temperature=math.exp(logs)
    weight=coef.T*temperature;bias=bias*temperature;weight-=weight.mean(0,keepdims=True);bias-=bias.mean()
    assert weight.shape==(20,256) and np.isfinite(weight).all() and np.isfinite(bias).all()
    return weight,bias,dict(TRAIN_samples=1600,TRAIN_source_families=1508,augmented_siblings=92,all_siblings_deleted_together=True,
        teacher_eligible_TRAIN=int(gate.sum()),teacher_temperature=2,hard_target_weight=.5,teacher_only_when_TRAIN_correct_confident=True,
        regularization=regularization,regularization_to_max_singular_squared=math.exp(logratio),analytic_TRAIN_deleted_MSE=mse,
        analytic_TRAIN_deleted_NLL=nll_value,analytic_TRAIN_deleted_correct=int((deleted.argmax(1)==labels).sum()),effective_degrees_of_freedom=dof,
        logit_scale=temperature,regularization_objective_evaluations=calls,scale_objective_evaluations=scale_calls,
        external_VALID_samples=0,TEST_used=False,final_fits_all1600=True,new_inference_layers=0,inference_teacher=False)


def selftest():
    y=np.arange(40)%20;t=np.eye(20)[y]*8;targets,gate=soft_targets(y,t)
    assert gate.all() and np.all(targets.max(1)<1) and np.all(targets.argmax(1)==y)
    bad=np.eye(20)[(y+1)%20]*8;q,g=soft_targets(y,bad)
    assert not g.any() and np.array_equal(q,np.eye(20)[y])
    return dict(passed=True,wrong_teacher_ignored=True,soft_targets_preserve_TRAIN_label=True,TEST_used=False,new_inference_layers=0)
