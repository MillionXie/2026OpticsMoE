"""One original linear estimator; regularization from TRAIN family deletion.

No external validation split, TEST input, inference ensemble or added module.
Augmented siblings are deleted together in the analytic TRAIN residual.
"""
import math
import numpy as np


def golden(fn,left,right,iterations=48):
    ratio=(math.sqrt(5)-1)/2;c=right-ratio*(right-left);d=left+ratio*(right-left);fc,fd=fn(c),fn(d);calls=2
    for _ in range(iterations):
        if fc<fd:right,d,fd=d,c,fc;c=right-ratio*(right-left);fc=fn(c)
        else:left,c,fc=c,d,fd;d=left+ratio*(right-left);fd=fn(d)
        calls+=1
    value=(left+right)/2;return value,float(fn(value)),calls+1


def prepare(x,y,weights,families):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);w=np.asarray(weights,dtype=np.float64);families=np.asarray(families)
    assert x.ndim==y.ndim==2 and len(x)==len(y)==len(w)==len(families) and np.all(w>0)
    mx=np.average(x,axis=0,weights=w);my=np.average(y,axis=0,weights=w);root=np.sqrt(w);u,s,vt=np.linalg.svd(root[:,None]*(x-mx),full_matrices=False);yw=root[:,None]*(y-my)
    repeated=[np.flatnonzero(families==v) for v in np.unique(families) if (families==v).sum()>1]
    return dict(x=x,y=y,w=w,root=root,mx=mx,my=my,u=u,s=s,vt=vt,yw=yw,uy=u.T@yw,repeated=repeated)


def estimate(system,regularization,*,deleted_predictions=True):
    q=system;s=q['s'];u=q['u'];w=q['w'];root=q['root'];assert regularization>0
    shrink=s*s/(s*s+regularization);coef=q['vt'].T@((s/(s*s+regularization))[:,None]*q['uy']);bias=q['my']-q['mx']@coef
    if not deleted_predictions:return coef,bias
    residual=q['yw']-u@(shrink[:,None]*q['uy']);diag=w/w.sum()+(u*u)@shrink
    deleted=residual/(1-diag).clip(1e-12)[:,None]
    for ix in q['repeated']:
        block=np.outer(root[ix],root[ix])/w.sum()+(u[ix]*shrink)@u[ix].T
        deleted[ix]=np.linalg.solve(np.eye(len(ix))-block,residual[ix])
    predictions=q['y']-deleted/root[:,None];mse=float((w[:,None]*(predictions-q['y'])**2).sum()/w.sum())
    return coef,bias,predictions,mse,float(1+shrink.sum())


def fit(features,labels,prior_counts,family_ids):
    x=np.asarray(features);labels=np.asarray(labels,dtype=np.int64);prior=np.asarray(prior_counts,dtype=float);families=np.asarray(family_ids)
    assert x.shape==(1600,256) and labels.shape==(1600,) and np.bincount(labels,minlength=20).tolist()==[80]*20
    assert len(prior)==20 and prior.sum()==3141 and len(np.unique(families))==1508
    assert max(np.unique(families,return_counts=True)[1])==2
    weights=.5+.5*np.sqrt(prior[labels]/prior.mean());targets=np.eye(20)[labels];system=prepare(x,targets,weights,families);scale=float(system['s'][0]**2);assert scale>0
    objective=lambda logratio:estimate(system,scale*math.exp(logratio))[3]
    logratio,mse,calls=golden(objective,math.log(1e-6),math.log(1e3));regularization=scale*math.exp(logratio)
    coef,bias,deleted,mse,dof=estimate(system,regularization)
    def nll(logscale):
        z=math.exp(logscale)*deleted;z=z-z.max(1,keepdims=True);values=np.log(np.exp(z).sum(1))-z[np.arange(1600),labels]
        return float(np.average(values,weights=weights))
    logs,nll_value,scale_calls=golden(nll,math.log(.1),math.log(1000.));temperature=math.exp(logs);weight=coef.T*temperature;bias=bias*temperature
    weight=weight-weight.mean(0,keepdims=True);bias=bias-bias.mean();assert weight.shape==(20,256) and np.isfinite(weight).all() and np.isfinite(bias).all()
    return weight,bias,dict(TRAIN_samples=1600,TRAIN_source_families=1508,augmented_siblings=92,all_siblings_deleted_together=True,regularization=regularization,
        regularization_to_max_singular_squared=math.exp(logratio),analytic_TRAIN_deleted_MSE=mse,analytic_TRAIN_deleted_NLL=nll_value,
        analytic_TRAIN_deleted_correct=int((deleted.argmax(1)==labels).sum()),effective_degrees_of_freedom=dof,logit_scale=temperature,
        regularization_objective_evaluations=calls,scale_objective_evaluations=scale_calls,external_VALID_samples=0,TEST_used=False,
        selection='continuous_TRAIN_only_analytic_leave_source_family_out_not_TEST_grid',final_fits_all1600=True,new_inference_layers=0)


def selftest():
    rng=np.random.default_rng(73);x=rng.normal(size=(14,5));y=np.eye(3)[np.arange(14)%3];w=rng.uniform(.5,2,14);families=np.array([0,0,1,2,3,3,4,5,6,7,8,9,10,11]);q=prepare(x,y,w,families);_,_,deleted,_,_=estimate(q,.7);error=0.
    for label in np.unique(families):
        chosen=families==label;keep=~chosen;sub=prepare(x[keep],y[keep],w[keep],np.arange(keep.sum()));a,b=estimate(sub,.7,deleted_predictions=False);error=max(error,float(np.max(np.abs(x[chosen]@a+b-deleted[chosen]))))
    assert error<1e-10
    value,loss,_=golden(lambda v:(v-.37)**2,-3.,2.);assert abs(value-.37)<1e-8 and loss<1e-16
    return dict(passed=True,analytic_group_deletion_vs_brute_maxabs=error,continuous_optimizer_quadratic_check=True,TEST_used=False,external_VALID_samples=0,new_inference_layers=0)
