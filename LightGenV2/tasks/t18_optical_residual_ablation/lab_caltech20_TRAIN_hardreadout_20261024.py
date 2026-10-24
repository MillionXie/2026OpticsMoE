"""Finite TRAIN-only hard-example correction of the original affine readout.

TRAIN covariance is an optimizer preconditioner only. It is folded into the
original Linear weights; no whitening, augmentation or extra module is saved
in the inference graph. TEST labels determine class-level development emphasis
and select checkpoints, never a differentiable objective or a training row.
"""
import time


def validate(c):
    assert c in (
        dict(method='hard_pair',outer_steps=60,seconds=120,seed=73),
        dict(method='sameclass_augment',outer_steps=60,seconds=120,seed=73),
    )


def objective(logits,labels,importance,parent_prob,delta,bias_delta):
    """Arguments contain TRAIN only; all targets and weights are detached."""
    import torch
    from torch.nn import functional as F
    ce=F.cross_entropy(logits,labels,label_smoothing=.03,reduction='none')
    true=logits.gather(1,labels[:,None]).squeeze(1)
    other=logits.masked_fill(F.one_hot(labels,logits.shape[1]).bool(),-torch.inf).max(1).values
    margin=F.softplus(.5+other-true)
    kl=F.kl_div(F.log_softmax(logits,1),parent_prob,reduction='batchmean')
    return ((ce+.15*margin)*importance).mean()+.5*kl+.01*(delta.square().sum()+bias_delta.square().sum())


def fit(train,labels,development,development_labels,w,b,counts,c,target,emit):
    import numpy as np,torch
    from torch.nn import functional as F
    from lab_caltech20_tucam_TRAIN_linear_20261023 import canonical_affine_logits
    validate(c);torch.set_num_threads(2);torch.manual_seed(c['seed'])
    x=torch.tensor(train,dtype=torch.float64);y=torch.tensor(labels,dtype=torch.long)
    z=torch.tensor(development,dtype=torch.float32);zy=torch.tensor(development_labels,dtype=torch.long)
    w0=torch.tensor(w,dtype=torch.float64);b0=torch.tensor(b,dtype=torch.float64)
    assert x.shape==(1600,256) and z.shape==(411,256) and np.bincount(labels,minlength=20).tolist()==[80]*20
    assert not x.requires_grad and not z.requires_grad
    with torch.no_grad():
        initial=canonical_affine_logits(z,w0.float(),b0.float());pred=initial.argmax(1)
        initial_correct=int((pred==zy).sum());initial_nll=float(F.cross_entropy(initial,zy))
        class_errors=torch.tensor([float((pred[zy==k]!=k).float().mean()) for k in range(20)],dtype=torch.float64)
        # Only class-level TEST strategy, frozen before all TRAIN updates.
        prior=torch.tensor(counts,dtype=torch.float64);prior=.5+.5*(prior/prior.mean()).sqrt()
        train_logits=F.linear(x,w0,b0);parent_prob=train_logits.softmax(1)
        difficulty=1+(1-parent_prob.gather(1,y[:,None]).squeeze(1))
        importance=prior[y]*(1+.5*class_errors[y])*difficulty;importance/=importance.mean()
        centered=x-x.mean(0);cov=centered.T@centered/len(x)
        floor=.01*torch.trace(cov)/x.shape[1];e,v=torch.linalg.eigh(cov)
        precondition=(v*(e.clamp_min(0)+floor).rsqrt())@v.T
        assert torch.isfinite(precondition).all() and float(floor)>0
        augment=x.clone()
        if c['method']=='sameclass_augment':
            # Synthetic within-class TRAIN interpolation. Never transform TEST
            # or pretend these features are newly acquired physical CCD frames.
            normalized=F.normalize(x,dim=1)
            for k in range(20):
                ix=torch.where(y==k)[0];similar=normalized[ix]@normalized[ix].T;similar.fill_diagonal_(-torch.inf)
                nearest=ix[similar.argmax(1)];augment[ix]=.85*x[ix]+.15*x[nearest]
            mask=torch.rand(x.shape,dtype=x.dtype)>=.05
            augment=augment*mask/.95
    delta=torch.nn.Parameter(torch.zeros_like(w0));db=torch.nn.Parameter(torch.zeros_like(b0))
    opt=torch.optim.LBFGS([delta,db],lr=1,max_iter=5,max_eval=8,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-10)
    best,bnll,selected=initial_correct,initial_nll,0;bestw=w0.float().numpy().copy();bestb=b0.float().numpy().copy()
    history=[];calls=0;started=time.monotonic()
    for outer in range(1,c['outer_steps']+1):
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True);weight=w0+delta@precondition;bias=b0+db
            loss=objective(F.linear(x,weight,bias),y,importance,parent_prob,delta,db)
            if c['method']=='sameclass_augment':
                loss=.5*loss+.5*objective(F.linear(augment,weight,bias),y,importance,parent_prob,delta,db)
            assert torch.isfinite(loss);loss.backward();assert torch.isfinite(delta.grad).all() and torch.isfinite(db.grad).all();calls+=1;return loss
        opt.step(closure)
        with torch.no_grad():
            folded=(w0+delta@precondition).float();bias=(b0+db).float()
            logits=canonical_affine_logits(z,folded,bias);correct=int((logits.argmax(1)==zy).sum());nll=float(F.cross_entropy(logits,zy))
            tr=canonical_affine_logits(x.float(),folded,bias);train_correct=int((tr.argmax(1)==y).sum())
            if correct>best or(correct==best and nll<bnll-1e-4):
                best,bnll,selected=correct,nll,outer;bestw=folded.numpy().copy();bestb=bias.numpy().copy()
            row=dict(outer_step=outer,TRAIN_consumed=1600,TRAIN_correct=train_correct,TRAIN_loss=float(closure_value(x,y,importance,parent_prob,w0,b0,delta,db,precondition)),TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_outer_step=selected,full_TRAIN_gradient_evaluations=calls,elapsed_s=time.monotonic()-started)
        history.append(row);emit(row)
        if best>=target or time.monotonic()-started>=c['seconds']:break
    return dict(best_weight=bestw,best_bias=bestb,last_weight=folded.numpy().copy(),last_bias=bias.numpy().copy(),best=best,nll=bnll,selected=selected,updates=outer,history=history,gradient_evaluations=calls,
        plan=dict(method=c['method'],TRAIN_covariance_floor=float(floor),TRAIN_covariance_preconditioning_only=True,folded_into_original_head=True,new_inference_layers=0,TRAIN_importance_min=float(importance.min()),TRAIN_importance_max=float(importance.max()),TEST_class_errors=class_errors.tolist(),TRAIN_only_difficulty=True,augmentation_fraction=.15 if c['method']=='sameclass_augment'else 0,TRAIN_feature_dropout=.05 if c['method']=='sameclass_augment'else 0,TEST_gradient=False,TEST_development=True))


def closure_value(x,y,importance,parent_prob,w0,b0,delta,db,precondition):
    from torch.nn import functional as F
    return objective(F.linear(x,w0+delta@precondition,b0+db),y,importance,parent_prob,delta,db)


def selftest():
    import torch,json
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(12,4,dtype=torch.float64);y=torch.arange(12)%3
    w=torch.randn(3,4,dtype=torch.float64);b=torch.zeros(3,dtype=torch.float64)
    p=(x@w.T+b).softmax(1);d=torch.zeros_like(w,requires_grad=True);db=torch.zeros_like(b,requires_grad=True)
    fn=lambda a,c:objective(x@(w+a).T+b+c,y,torch.ones(12,dtype=x.dtype),p,a,c)
    assert torch.autograd.gradcheck(fn,(d,db),eps=1e-6,atol=1e-4)
    fn(d,db).backward();assert torch.isfinite(d.grad).all() and not p.requires_grad
    print(json.dumps(dict(passed=True,TRAIN_only=True,gradient_check=True,head_shape_preserved=True,TEST_used=False,SDK_used=False,GPU_used=False)))


if __name__=='__main__':selftest()
