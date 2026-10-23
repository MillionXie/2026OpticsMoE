"""TRAIN-only normalized teacher-logit estimator, original affine readout only."""
import math
import numpy as np
from lab_caltech20_TRAIN_group_ridge_20261024 import prepare,estimate,golden


def targets(labels,logits):
    y=np.asarray(labels,dtype=np.int64);z=np.asarray(logits,dtype=np.float64).copy()
    assert z.shape==(len(y),20) and np.isfinite(z).all()
    z-=z.mean(1,keepdims=True);rms=np.sqrt(np.mean(z*z,axis=1,keepdims=True))
    gate=(z.argmax(1)==y)&(rms[:,0]>=.5);v=z/rms.clip(.5)
    hard=np.eye(20)[y];hard-=hard.mean(1,keepdims=True);hard/=np.sqrt(np.mean(hard*hard,axis=1,keepdims=True));v[~gate]=hard[~gate]
    assert np.all(v.argmax(1)==y) and np.isfinite(v).all()
    return v,gate


def fit(features,labels,prior_counts,family_ids,teacher_logits):
    x=np.asarray(features);y=np.asarray(labels,dtype=np.int64);prior=np.asarray(prior_counts,dtype=float);families=np.asarray(family_ids)
    assert x.shape==(1600,256) and np.bincount(y,minlength=20).tolist()==[80]*20
    assert prior.shape==(20,) and prior.sum()==3141 and len(np.unique(families))==1508 and max(np.unique(families,return_counts=True)[1])==2
    target,gate=targets(y,teacher_logits);weights=.5+.5*np.sqrt(prior[y]/prior.mean());q=prepare(x,target,weights,families);scale=float(q['s'][0]**2);assert scale>0
    lr,mse,calls=golden(lambda v:estimate(q,scale*math.exp(v))[3],math.log(1e-6),math.log(1e3));regularization=scale*math.exp(lr);w,b,deleted,mse,dof=estimate(q,regularization)
    def nll(logscale):
        z=math.exp(logscale)*deleted;z-=z.max(1,keepdims=True);return float(np.average(np.log(np.exp(z).sum(1))-z[np.arange(1600),y],weights=weights))
    ls,nll_value,ncalls=golden(nll,math.log(.1),math.log(1000));correct=int((deleted.argmax(1)==y).sum())
    # Predeclared support based only on TRAIN, before any native TEST candidate.
    assert correct>=873 and nll_value<1.519680122241567,'TRAIN family-deleted support gate failed'
    weight=w.T*math.exp(ls);bias=b*math.exp(ls);weight-=weight.mean(0,keepdims=True);bias-=bias.mean()
    return weight,bias,dict(TRAIN_samples=1600,TRAIN_source_families=1508,augmented_siblings=92,all_siblings_deleted_together=True,teacher_correct_TRAIN=int(gate.sum()),
        target='centered RMS-normalized teacher logits; wrong/nearzero TRAIN teacher replaced with normalized true TRAIN onehot',regularization=regularization,regularization_to_max_singular_squared=math.exp(lr),
        analytic_TRAIN_deleted_correct=correct,analytic_TRAIN_deleted_NLL=nll_value,analytic_TRAIN_deleted_MSE=mse,effective_degrees_of_freedom=dof,logit_scale=math.exp(ls),
        predeclared_TRAIN_support_correct=873,regularization_objective_evaluations=calls,scale_objective_evaluations=ncalls,external_VALID_samples=0,TEST_used=False,final_fits_all1600=True,new_inference_layers=0,inference_teacher=False)


def selftest():
    y=np.arange(20);z=np.eye(20)*8;v,g=targets(y,z);assert g.all() and np.all(v.argmax(1)==y) and np.allclose(v.mean(1),0)
    wrong=np.eye(20)[(y+1)%20]*8;a,b=targets(y,wrong);zero,c=targets(y,np.zeros((20,20)));assert not b.any() and not c.any() and np.array_equal(a,zero)
    assert np.array_equal(z,np.eye(20)*8)
    return dict(passed=True,teacher_inputs_unchanged=True,wrong_teacher_ignored=True,TRAIN_targets_preserve_true_labels=True,TEST_used=False,new_inference_layers=0)
