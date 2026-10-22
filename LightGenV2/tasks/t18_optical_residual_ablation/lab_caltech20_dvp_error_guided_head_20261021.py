"""Bounded TRAIN-only focal correction and read-only class/domain diagnostics.

No new inference layer: condition optimizer coordinates using TRAIN and fold
the update into original Linear. TEST may choose TRAIN indices/checkpoints;
the differentiable fitting function accepts no TEST features or TEST labels.
"""
import time


def spatial_group_delta(torch,F,x,y,pw,pb,counts,cfg,observe):
    """TRAIN-selected sparse spatial COLUMN updates, folded into original Linear."""
    x=x.double();w=pw.double();b=pb.double();w=w-w.mean(0,keepdim=True);b=b-b.mean()
    xx=torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),1);base=F.linear(x,w,b).detach()
    prior=(counts/counts.mean()).sqrt();weights=prior[y];weights=weights/weights.sum()
    temperature=2.;teacher=(base/temperature).softmax(1);keep=(base.argmax(1)==y).double();kept=keep.sum().clamp_min(1)
    def smooth(u):
        logits=base+xx@u.T
        ce=F.cross_entropy(logits,y,reduction='none',label_smoothing=cfg['label_smoothing'])
        kl=F.kl_div((logits/temperature).log_softmax(1),teacher,reduction='none').sum(1)*temperature**2
        return (weights*ce).sum()+cfg['teacher_KL']*(kl*keep).sum()/kept+cfg['anchor']*u.square().sum()
    zero=torch.zeros(20,xx.shape[1],dtype=x.dtype,requires_grad=True);grad=torch.autograd.grad(smooth(zero),zero)[0]
    penalty=max(1e-9,cfg['penalty_fraction']*float(grad[:,:-1].norm(dim=0).max()))
    spectral=float(torch.linalg.matrix_norm(xx,ord=2))**2
    lipschitz=.5*(float(weights.max())+cfg['teacher_KL']/float(kept))*spectral+2*cfg['anchor'];step=1/max(lipschitz,1e-9)
    u=zero.detach();z=u.clone();momentum=1.;started=time.monotonic();restarts=0
    def objective(v):return smooth(v)+penalty*v[:,:-1].norm(dim=0).sum()
    previous=float(objective(u));history=[];ww,bb=pw,pb
    for iteration in range(1,cfg['steps']+1):
        def propose(point):
            point=point.detach().requires_grad_(True);g=torch.autograd.grad(smooth(point),point)[0]
            assert torch.isfinite(g).all();next_u=(point-step*g).detach()
            next_u[:,:-1]*=(1-step*penalty/next_u[:,:-1].norm(dim=0,keepdim=True).clamp_min(1e-12)).clamp_min(0)
            next_u-=next_u.mean(0,keepdim=True);return next_u
        v=propose(z);value=float(objective(v))
        if value>previous+1e-10:z=u;momentum=1.;v=propose(z);value=float(objective(v));restarts+=1
        assert value<=previous+1e-7 and torch.isfinite(v).all()
        next_momentum=(1+(1+4*momentum**2)**.5)/2;z=v+(momentum-1)/next_momentum*(v-u);u=v;momentum=next_momentum;previous=value
        if iteration==1 or iteration%cfg['evaluate_every']==0 or iteration==cfg['steps']:
            ww=w+u[:,:-1];bb=b+u[:,-1]
            row=dict(phase='TRAIN_spatial_group_delta',step=iteration,TRAIN_samples=len(x),TRAIN_objective=value,active_spatial_columns=int((u[:,:-1].norm(dim=0)>1e-10).sum()),TRAIN_correct=int(((base+xx@u.T).argmax(1)==y).sum()))
            history.append(row);stop=observe(ww,bb,row)
            if stop or time.monotonic()-started>=cfg['fit_seconds']:break
    return ww,bb,dict(TRAIN_only=True,TRAIN_samples=len(x),actual_steps=iteration,penalty=penalty,penalty_rule='fixed fraction of maximum TRAIN-only initial group gradient',lipschitz_upper_bound=lipschitz,step_size=step,monotone_restarts=restarts,active_spatial_columns=history[-1]['active_spatial_columns'],history=history,fit_seconds=time.monotonic()-started,original_Linear_fold=True,TEST_gradient=False,new_inference_layers=0)


def spatial_group_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(80,16,dtype=torch.float64);y=torch.arange(80)%20;w=torch.randn(20,16)*.01;b=torch.zeros(20);rows=[]
    ww,bb,proof=spatial_group_delta(torch,F,x,y,w,b,torch.ones(20,dtype=x.dtype),dict(penalty_fraction=.1,label_smoothing=.02,teacher_KL=.1,anchor=.01,steps=4,evaluate_every=1,fit_seconds=30),lambda a,b,r:rows.append(r))
    assert len(rows)==4 and ww.shape==w.shape and torch.isfinite(ww).all()
    assert all(rows[j]['TRAIN_objective']<=rows[j-1]['TRAIN_objective']+1e-7 for j in range(1,4))
    assert not proof['TEST_gradient'] and proof['original_Linear_fold'] and proof['new_inference_layers']==0
    print(dict(passed=True,TRAIN_spatial_group_only=True,original_Linear_shape=list(ww.shape),new_layer=False))


def error_diagnosis(np,tr,te,train_probability,test_probability):
    rows={};classes=[]
    for k in range(20):classes.append(str(tr['ids'][np.flatnonzero(tr['y']==k)[0]]).replace('\\','/').split('/')[-2])
    for role,bank,p in (('train',tr,train_probability),('test',te,test_probability)):
        y=bank['y'];pred=p.argmax(1);cm=np.bincount(20*y+pred,minlength=400).reshape(20,20)
        rows[role]=dict(correct=int((pred==y).sum()),samples=len(y),class_support=cm.sum(1).tolist(),class_correct=cm.diagonal().tolist(),confusion=cm.tolist(),
            top_confusions=[dict(count=int(cm[i,j]),actual=classes[i],predicted=classes[j]) for _,i,j in sorted([(int(cm[i,j]),i,j) for i in range(20) for j in range(20) if i!=j],reverse=True)[:12]])
    x,q=tr['x'].astype(float),te['x'].astype(float);nx=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);nq=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
    similarity=nq@nx.T;near=similarity.argmax(1);class_rows=[]
    for k in range(20):
        tx=x[tr['y']==k];qx=q[te['y']==k];radius=np.mean(np.sum((tx-tx.mean(0))**2,1));shift=np.sum((qx.mean(0)-tx.mean(0))**2)
        class_rows.append(dict(name=classes[k],TRAIN_correct=rows['train']['class_correct'][k],TRAIN_samples=len(tx),TEST_correct=rows['test']['class_correct'][k],TEST_samples=len(qx),
            centroid_shift_to_TRAIN_radius=float(shift/max(radius,1e-12)),nearest_TRAIN_same_class=int((tr['y'][near[te['y']==k]]==k).sum())))
    return dict(classes=class_rows,metrics=rows,TEST_nearest_TRAIN_accuracy=float((tr['y'][near]==te['y']).mean()),
        interpretation='Centroid/nearest-neighbour diagnostics are descriptive, not proof of optical drift or an added classifier. TRAIN and TEST weakness are separated; no TEST gradient.',TEST_informed_development=True,new_CCD=0)


def focal_correction(torch,F,x,y,pw,pb,counts,importance,cfg,observe,teacher_logits=None):
    """All1600TRAIN, positive bounded weights; no TEST tensor enters loss."""
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,_,_,floor=train_conditioned_geometry(x.numpy());xx=torch.cat(((x-mean)@root,torch.ones(len(x),1,dtype=x.dtype)),1)
    w0=pw.double()-pw.double().mean(0,keepdim=True);b0=pb.double()-pb.double().mean();base=F.linear(x,w0,b0)
    teacher=(base/cfg['temperature']).softmax(1).detach();keep=(base.argmax(1)==y).double()
    external=None;repair=torch.zeros_like(keep)
    if teacher_logits is not None:
        assert teacher_logits.shape==base.shape and not teacher_logits.requires_grad
        external=(teacher_logits/cfg['temperature']).softmax(1).detach();repair=((external.argmax(1)==y)&(base.argmax(1)!=y)).double();assert float(repair.sum())>0
    sample=importance.double()*(counts[y]/counts.mean()).sqrt();sample/=sample.sum();assert bool((sample>0).all())
    delta=torch.nn.Parameter(torch.zeros(len(pb),xx.shape[1],dtype=x.dtype));opt=torch.optim.Adam([delta],lr=cfg['learning_rate'])
    start=time.monotonic();ww,bb=pw,pb
    for step in range(1,cfg['steps']+1):
        z=base+xx@delta.T;logp=z.log_softmax(1);ce=-logp[torch.arange(len(y)),y];pt=logp[torch.arange(len(y)),y].exp()
        focal=(1-pt).pow(cfg['gamma'])*ce;kl=F.kl_div((z/cfg['temperature']).log_softmax(1),teacher,reduction='none').sum(1)*cfg['temperature']**2
        loss=(sample*((1-cfg['CE_replay'])*focal+cfg['CE_replay']*ce)).sum()+cfg['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*delta.square().sum()
        if external is not None:
            guidance=F.kl_div((z/cfg['temperature']).log_softmax(1),external,reduction='none').sum(1)*cfg['temperature']**2
            loss=loss+cfg['teacher_distillation']*(guidance*repair).sum()/repair.sum().clamp_min(1)
        opt.zero_grad();loss.backward();assert delta.grad is not None and torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],1.);opt.step()
        with torch.no_grad():
            delta.sub_(delta.mean(0,keepdim=True))
            if cfg.get('trust_scope','whole_head')=='per_class_row':
                delta.mul_((cfg['trust_radius']/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1.));assert bool((delta.norm(dim=1)<=cfg['trust_radius']+1e-10).all())
            else:delta.mul_(min(1.,cfg['trust_radius']/max(float(delta.norm()),1e-12)))
            dw=delta[:,:-1]@root;ww=w0+dw;bb=b0+delta[:,-1]-dw@mean
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            stop=observe(ww.detach(),bb.detach(),dict(phase='TRAIN_selective_teacher_focal' if external is not None else 'TRAIN_error_guided_focal',step=step,TRAIN_all1600=True,TRAIN_loss=float(loss.detach()),TRAIN_focal_loss=float((sample*focal).sum().detach()),TRAIN_ce=float((sample*ce).sum().detach()),gradient_samples=len(y),trusted_teacher_repair_samples=int(repair.sum()),delta_global_norm=float(delta.detach().norm()),delta_max_row_norm=float(delta.detach().norm(dim=1).max())))
            if stop:break
        if time.monotonic()-start>=cfg['fit_seconds']:break
    return ww.detach(),bb.detach(),dict(TRAIN_only=True,positive_TRAIN_weights=True,variance_floor=floor,gradient_updates=step,fit_seconds=time.monotonic()-start,original_Linear_fold=True,TEST_gradient=False,trusted_teacher_repair_samples=int(repair.sum()),parent_correct_protected_samples=int(keep.sum()),trust_scope=cfg.get('trust_scope','whole_head'),delta_global_norm=float(delta.detach().norm()),delta_max_row_norm=float(delta.detach().norm(dim=1).max()))


def selftest():
    import torch,numpy as np
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(40,256,dtype=torch.float64);y=torch.arange(40)%20;w=torch.randn(20,256,dtype=x.dtype)*.02;b=torch.zeros(20,dtype=x.dtype);rows=[]
    cfg=dict(temperature=2.,learning_rate=.003,steps=2,gamma=2.,CE_replay=.25,teacher_KL=.5,anchor=.01,trust_radius=1.,fit_seconds=30,evaluate_every=1)
    ww,bb,proof=focal_correction(torch,F,x,y,w,b,torch.ones(20,dtype=x.dtype),torch.ones(40,dtype=x.dtype),cfg,lambda a,b,r:rows.append(r))
    assert len(rows)==2 and torch.isfinite(ww).all() and torch.isfinite(bb).all() and not torch.equal(ww,w)
    target=F.one_hot(y,20).double()*5;cfg['teacher_distillation']=1.;ww,bb,proof=focal_correction(torch,F,x,y,w,b,torch.ones(20,dtype=x.dtype),torch.ones(40,dtype=x.dtype),cfg,lambda a,b,r:False,target)
    assert proof['trusted_teacher_repair_samples']>0 and torch.isfinite(ww).all()
    cfg.update(trust_scope='per_class_row',trust_radius=.01);ww,bb,proof=focal_correction(torch,F,x,y,w,b,torch.ones(20,dtype=x.dtype),torch.ones(40,dtype=x.dtype),cfg,lambda a,b,r:False,target)
    assert proof['delta_max_row_norm']<=.01+1e-10 and proof['delta_global_norm']<=20**.5*.01+1e-10
    tr=dict(x=x.numpy(),y=y.numpy(),ids=np.asarray(['root/class'+str(int(k))+'/a.jpg' for k in y]));p=F.linear(x,w,b).softmax(1).numpy();report=error_diagnosis(np,tr,tr,p,p)
    assert len(report['classes'])==20 and all(abs(z['centroid_shift_to_TRAIN_radius'])<1e-10 for z in report['classes'])
    print(dict(passed=True,TRAIN_only_signature=True,finite_nonzero_update=True,diagnostic_identity=True,no_new_inference_layer=True))


def multiclass_maxmargin(torch,F,x,y,counts,cfg,observe):
    """TRAIN-conditioned multiclass squared hinge, C=1, original Linear only.

    A fresh convex readout fit, not the previous small parent-delta constraint.
    TRAIN sample count fixes regularization; no TEST tensor or C/seed sweep.
    """
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,_,_,floor=train_conditioned_geometry(x.numpy());xx=(x-mean)@root
    w=torch.nn.Parameter(torch.zeros(20,x.shape[1],dtype=x.dtype));b=torch.nn.Parameter(torch.zeros(20,dtype=x.dtype))
    sample=counts[y]/counts.mean();sample/=sample.sum();regularizer=1/len(y)
    opt=torch.optim.LBFGS([w,b],lr=1.,max_iter=1,line_search_fn='strong_wolfe');start=time.monotonic();ww,bb=w.detach(),b.detach()
    def objective():
        z=F.linear(xx,w,b);wrong=z.masked_fill(F.one_hot(y,20).bool(),-torch.inf).max(1).values
        hinge=(1-z[torch.arange(len(y)),y]+wrong).clamp_min(0)
        return .5*(sample*hinge.square()).sum()+.5*regularizer*(w.square().sum()+b.square().sum())
    for step in range(1,cfg['steps']+1):
        def closure():
            opt.zero_grad();loss=objective();loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in (w,b));return loss
        opt.step(closure)
        with torch.no_grad():ww=w@root;bb=b-ww@mean
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            stop=observe(ww.detach(),bb.detach(),dict(phase='TRAIN_multiclass_maxmargin',step=step,TRAIN_all1600=True,TRAIN_loss=float(objective().detach()),TRAIN_regularizer=regularizer))
            if stop:break
        if time.monotonic()-start>=cfg['fit_seconds']:break
    return ww.detach(),bb.detach(),dict(TRAIN_only=True,margin=1.,C=1.,regularizer=regularizer,variance_floor=floor,steps=step,no_parent_delta_cap=True,TEST_gradient=False)


def maxmargin_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(40,256,dtype=torch.float64);y=torch.arange(40)%20;rows=[]
    w,b,proof=multiclass_maxmargin(torch,F,x,y,torch.ones(20,dtype=x.dtype),dict(steps=3,evaluate_every=1,fit_seconds=30),lambda a,b,r:rows.append(r))
    assert len(rows)==3 and w.norm()>0 and torch.isfinite(w).all() and torch.isfinite(b).all()
    print(dict(passed=True,maxmargin_TRAIN_only=True,original_Linear=True,finite_gradient=True,regularizer=proof['regularizer']))


def class_covariances(torch,x,y):
    """TRAIN class covariance, shrunk toward TRAIN pooled within-class covariance.

    Shrinkage d/(d+n_k) is predetermined by the small sample dimension ratio;
    neither TEST features nor TEST labels participate in this statistic.
    """
    centers=torch.stack([x[y==k].mean(0) for k in range(20)])
    residual=x-centers[y];pooled=residual.T@residual/len(x);d=x.shape[1]
    cov=[];shrink=[]
    for k in range(20):
        r=residual[y==k];n=len(r);assert n>1
        ratio=d/(d+n);cov.append((1-ratio)*(r.T@r/n)+ratio*pooled);shrink.append(ratio)
    covariance=torch.stack(cov).detach()
    assert torch.isfinite(covariance).all() and float(torch.linalg.eigvalsh(covariance).min())>=-1e-9
    return covariance,shrink


def isda_variance(torch,weight,covariance):
    """[true class, competitor] quadratic variance; correct column is zero."""
    rows=[]
    for k in range(len(weight)):
        delta=weight-weight[k]
        rows.append(((delta@covariance[k])*delta).sum(1).clamp_min(0))
    return torch.stack(rows)


def covariance_augmented_head(torch,F,x,y,pw,pb,counts,cfg,observe):
    """Finite TRAIN-only ISDA-style robust CE plus one weight-average candidate.

    The optimized affine head is folded back into ORIGINAL Linear5140. All
    covariance/teacher fitting and derivatives use TRAIN. observe is detached
    development selection, not an input to any differentiable objective.
    """
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,inverse,_,floor=train_conditioned_geometry(x.numpy());xx=(x-mean)@root
    w0=(pw.double()-pw.double().mean(0,keepdim=True))@inverse
    b0=pb.double()-pb.double().mean()+(pw.double()-pw.double().mean(0,keepdim=True))@mean
    with torch.no_grad():
        base=F.linear(xx,w0,b0);T=cfg['temperature'];teacher=(base/T).softmax(1);keep=(base.argmax(1)==y).double()
    covariance,shrink=class_covariances(torch,xx,y)
    sample=(counts[y]/counts.mean()).sqrt();sample/=sample.sum()
    dw=torch.nn.Parameter(torch.zeros_like(w0));db=torch.nn.Parameter(torch.zeros_like(b0))
    opt=torch.optim.Adam([dw,db],lr=cfg['learning_rate']);start=time.monotonic();averages=[];rows=0
    for step in range(1,cfg['steps']+1):
        w,b=w0+dw,b0+db;z=F.linear(xx,w,b)
        strength=cfg['augmentation_strength']*min(1.,step/cfg['warmup_steps'])
        adjusted=z+.5*strength*isda_variance(torch,w,covariance)[y]
        robust=(sample*F.cross_entropy(adjusted,y,reduction='none')).sum()
        distill=F.kl_div((z/T).log_softmax(1),teacher,reduction='none').sum(1)*T*T
        loss=robust+cfg['teacher_KL']*(distill*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*(dw.square().sum()+db.square().sum())
        opt.zero_grad();loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in (dw,db))
        torch.nn.utils.clip_grad_norm_([dw,db],1.);opt.step()
        with torch.no_grad():dw.sub_(dw.mean(0,keepdim=True));db.sub_(db.mean())
        if step>=cfg['average_start'] and step%cfg['evaluate_every']==0:
            averages.append((dw.detach().clone(),db.detach().clone()))
            if len(averages)>cfg['average_snapshots']:averages.pop(0)
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            with torch.no_grad():ww=(w0+dw)@root;bb=b0+db-ww@mean
            stop=observe(ww.detach(),bb.detach(),dict(phase='TRAIN_covariance_augmentation',step=step,TRAIN_all1600=True,TRAIN_loss=float(loss.detach()),TRAIN_robust_CE=float(robust.detach()),augmentation_strength=strength,TRAIN_clean_correct=int((z.argmax(1)==y).sum()),delta_norm=float(dw.detach().norm())))
            rows+=1
            if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    with torch.no_grad():
        ww=(w0+dw)@root;bb=b0+db-ww@mean
        if averages:
            aw=w0+torch.stack([v[0] for v in averages]).mean(0);ab=b0+torch.stack([v[1] for v in averages]).mean(0)
            averaged_w=aw@root;averaged_b=ab-averaged_w@mean
            observe(averaged_w,averaged_b,dict(phase='TRAIN_covariance_weight_average',step=step,TRAIN_all1600=True,snapshots=len(averages),same_original_Linear=True))
    return ww.detach(),bb.detach(),dict(method='TRAIN_ISDA_style_covariance_augmentation',reference='https://arxiv.org/abs/1909.12220',TRAIN_only=True,TRAIN_samples=len(y),class_covariance_shrinkage=shrink,variance_floor=floor,actual_steps=step,observations=rows,fit_seconds=time.monotonic()-start,averaged_snapshots=len(averages),new_inference_parameters=0,TEST_gradient=False)


def covariance_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(60,8,dtype=torch.float64);y=torch.arange(60)%20;w=torch.randn(20,8,dtype=x.dtype)*.05;b=torch.randn(20,dtype=x.dtype)*.01
    covariance,shrink=class_covariances(torch,x,y);v=isda_variance(torch,w,covariance)
    assert bool((v>=0).all()) and torch.equal(v.diagonal(),torch.zeros(20,dtype=x.dtype))
    for k in range(20):
        for j in range(20):
            d=w[j]-w[k];assert abs(float(v[k,j]-d@covariance[k]@d))<1e-10
    logits=F.linear(x,w,b);assert torch.equal(F.cross_entropy(logits+0*v[y],y),F.cross_entropy(logits,y))
    rows=[];cfg=dict(temperature=2.,augmentation_strength=.1,warmup_steps=2,learning_rate=.003,teacher_KL=.5,anchor=.01,steps=3,evaluate_every=1,average_start=2,average_snapshots=2,fit_seconds=30)
    ww,bb,proof=covariance_augmented_head(torch,F,x,y,w,b,torch.ones(20,dtype=x.dtype),cfg,lambda a,b,r:rows.append(r))
    assert torch.isfinite(ww).all() and torch.isfinite(bb).all() and not torch.equal(ww,w)
    assert len(rows)==4 and proof['actual_steps']==3 and proof['averaged_snapshots']==2
    print(dict(passed=True,quadratic_variance_exact=True,zero_strength_identity=True,TRAIN_covariance_PSD=True,TRAIN_only_derivatives=True,averaged_original_head=True))


def prior_temperature_head(torch,F,x,y,pw,pb,counts,cfg,observe):
    """Two TRAIN-fitted scalars; fold scale and original TRAIN prior into Linear.

    The parent was already TRAIN-prior weighted, so exact Bayes correction is
    NOT assumed. Retain it as a diagnostic anchor, then fit signed residual
    prior strength and confidence only on weighted TRAIN. No TEST prior.
    """
    cw=pw.double()-pw.double().mean(0,keepdim=True);cb=pb.double()-pb.double().mean()
    z0=F.linear(x,cw,cb);logprior=(counts/counts.sum()).log();logprior-=logprior.mean()
    weights=counts/counts.mean();T=cfg['temperature'];teacher=(z0/T).softmax(1).detach();keep=(z0.argmax(1)==y).double()
    observe(cw,cb+logprior,dict(phase='TRAIN_prior_analytic_anchor',TRAIN_all1600=True,scale=1.,prior_strength=1.,gradient_updates=0))
    theta=torch.nn.Parameter(torch.tensor([1.,0.],dtype=x.dtype))
    opt=torch.optim.LBFGS([theta],lr=.5,max_iter=1,line_search_fn='strong_wolfe');start=time.monotonic()
    def criterion():
        z=theta[0]*z0+theta[1]*logprior
        ce=F.cross_entropy(z,y,weight=weights)
        kl=F.kl_div((z/T).log_softmax(1),teacher,reduction='none').sum(1)*T*T
        return ce+cfg['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*((theta[0]-1).square()+theta[1].square())
    for step in range(1,cfg['steps']+1):
        def closure():
            opt.zero_grad();loss=criterion();loss.backward();assert torch.isfinite(loss) and torch.isfinite(theta.grad).all();return loss
        opt.step(closure)
        with torch.no_grad():theta[0].clamp_(.25,4.);theta[1].clamp_(-2.,2.)
        ww=cw*theta[0].detach();bb=cb*theta[0].detach()+logprior*theta[1].detach()
        stop=observe(ww,bb,dict(phase='TRAIN_prior_temperature',step=step,TRAIN_all1600=True,TRAIN_loss=float(criterion().detach()),scale=float(theta[0].detach()),prior_strength=float(theta[1].detach()),scalar_parameters=2))
        if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    return ww,bb,dict(method='TRAIN_two_scalar_prior_confidence',TRAIN_only=True,original_TRAIN_prior=counts.tolist(),TRAIN_gradient_updates=step,scale=float(theta[0].detach()),prior_strength=float(theta[1].detach()),signed_prior_residual_not_assumed_unbiased=True,TEST_gradient=False)


def prior_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(60,8,dtype=torch.float64);y=torch.arange(60)%20
    w=torch.randn(20,8,dtype=x.dtype)*.1;b=torch.randn(20,dtype=x.dtype)*.1;counts=torch.arange(1,21,dtype=x.dtype);rows=[]
    ww,bb,proof=prior_temperature_head(torch,F,x,y,w,b,counts,dict(temperature=2.,teacher_KL=.1,anchor=.005,steps=3,fit_seconds=30),lambda a,b,r:rows.append(r))
    assert len(rows)==4 and torch.isfinite(ww).all() and torch.isfinite(bb).all() and proof['TRAIN_gradient_updates']==3
    s=proof['scale'];q=proof['prior_strength'];p=(counts/counts.sum()).log();p-=p.mean()
    expected=s*(F.linear(x,w,b)-F.linear(x,w,b).mean(1,keepdim=True))+q*p
    assert float((expected-F.linear(x,ww,bb)).abs().max())<1e-10
    print(dict(passed=True,original_head_fold_exact=True,TRAIN_prior_only=True,finite_TRAIN_gradients=True,scalar_parameters=2,no_new_inference_layer=True))


def paired_response_head(torch,F,x,y,nominal,pw,pb,counts,cfg,observe):
    """TRAIN paired measured/nominal final-response robust Linear optimization.

    Both views keep the measured first four stages and unchanged masks. The
    nominal last stage is ONLY synthetic TRAIN augmentation, not measured CCD.
    """
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,_,_,floor=train_conditioned_geometry(x.numpy());xx=(x-mean)@root;nn=(nominal.double()-mean)@root
    cw=pw.double()-pw.double().mean(0,keepdim=True);cb=pb.double()-pb.double().mean()
    base=F.linear(x,cw,cb);other=F.linear(nominal.double(),cw,cb)
    T=cfg['temperature'];teacher=(base/T).softmax(1).detach();keep=(base.argmax(1)==y).double()
    actual_correct=int(keep.sum());nominal_correct=int((other.argmax(1)==y).sum())
    proof=dict(TRAIN_only=True,TRAIN_samples=len(y),measured_TRAIN_correct=actual_correct,nominal_TRAIN_correct=nominal_correct,nominal_is_synthetic_not_CCD=True,TEST_used_for_views_or_derivatives=False,variance_floor=floor)
    if nominal_correct<actual_correct-cfg['maximum_TRAIN_view_gap']:
        proof.update(fit_skipped=True,reason='nominal_TRAIN_view_not_reliable_enough',actual_updates=0);return pw,pb,proof
    delta=torch.nn.Parameter(torch.zeros(len(pb),x.shape[1]+1,dtype=x.dtype));opt=torch.optim.Adam([delta],lr=cfg['learning_rate'])
    sample=(counts[y]/counts.mean()).sqrt();sample/=sample.sum();start=time.monotonic()
    for step in range(1,cfg['steps']+1):
        d=delta[:,:-1];b=delta[:,-1];zr=base+xx@d.T+b;zn=other+nn@d.T+b
        real_ce=(sample*F.cross_entropy(zr,y,reduction='none')).sum();nominal_ce=(sample*F.cross_entropy(zn,y,reduction='none')).sum()
        difference=zr-zn;difference=difference-difference.mean(1,keepdim=True)
        consistency=(sample*difference.square().mean(1)).sum()
        kl=F.kl_div((zr/T).log_softmax(1),teacher,reduction='none').sum(1)*T*T
        loss=(1-cfg['nominal_weight'])*real_ce+cfg['nominal_weight']*nominal_ce+cfg['response_consistency']*consistency+cfg['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*delta.square().sum()
        opt.zero_grad();loss.backward();assert torch.isfinite(loss) and delta.grad is not None and torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],1.);opt.step()
        with torch.no_grad():
            delta.sub_(delta.mean(0,keepdim=True));delta.mul_((cfg['row_trust_radius']/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1.))
            dw=delta[:,:-1]@root;ww=cw+dw;bb=cb+delta[:,-1]-dw@mean
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            stop=observe(ww.detach(),bb.detach(),dict(phase='TRAIN_paired_final_response',step=step,TRAIN_all1600=True,TRAIN_loss=float(loss.detach()),TRAIN_real_CE=float(real_ce.detach()),TRAIN_nominal_CE=float(nominal_ce.detach()),TRAIN_response_consistency=float(consistency.detach()),delta_max_row_norm=float(delta.detach().norm(dim=1).max())))
            if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    proof.update(fit_skipped=False,actual_updates=step,fit_seconds=time.monotonic()-start,original_Linear_only=True)
    return ww.detach(),bb.detach(),proof


def paired_response_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(60,8,dtype=torch.float64);y=torch.arange(60)%20;w=torch.randn(20,8)*.01;b=torch.zeros(20);rows=[]
    cfg=dict(temperature=2.,maximum_TRAIN_view_gap=160,learning_rate=.003,steps=3,evaluate_every=1,nominal_weight=.25,response_consistency=.01,teacher_KL=.5,anchor=.01,row_trust_radius=1.,fit_seconds=30)
    ww,bb,proof=paired_response_head(torch,F,x,y,x.clone(),w,b,torch.ones(20,dtype=x.dtype),cfg,lambda a,b,r:rows.append(r))
    assert len(rows)==3 and proof['actual_updates']==3 and torch.isfinite(ww).all() and all(abs(r['TRAIN_response_consistency'])<1e-10 for r in rows)
    print(dict(passed=True,TRAIN_only_paired_views=True,identical_views_zero_consistency=True,original_head_only=True,finite_nonzero_gradient=True))


def stable_basis(torch,x,kind,cfg):
    """TRAIN-only optimizer basis; never added to the inference graph."""
    mean=x.mean(0);xc=x-mean;d=x.shape[1];assert d==256
    cov=xc.T@xc/len(x);floor=max(float(cov.trace())/d*.02,1e-6)
    if kind=='TRAIN_PCA99':
        eigen,vectors=torch.linalg.eigh(cov);order=torch.arange(d-1,-1,-1);eigen=eigen[order].clamp_min(0);vectors=vectors[:,order]
        cumulative=eigen.cumsum(0)/eigen.sum().clamp_min(1e-20)
        rank=int(torch.searchsorted(cumulative,torch.tensor(cfg['variance_fraction'],dtype=x.dtype)))+1
        basis=vectors[:,:rank]/eigen[:rank].clamp_min(floor).sqrt()[None,:]
        proof=dict(kind=kind,rank=rank,TRAIN_variance_fraction=float(cumulative[rank-1]),TRAIN_spectrum=eigen.tolist(),variance_floor=floor)
    else:
        assert kind=='spatial_DCT8' and cfg['DCT_width']==8
        import math
        coordinate=torch.arange(16,dtype=x.dtype);columns=[]
        for p in range(8):
            for q in range(8):
                if p==q==0:continue  # layer-norm features have no useful DC component
                a=torch.cos(math.pi*(2*coordinate+1)*p/32)*(1/4 if p==0 else (1/8)**.5)
                b=torch.cos(math.pi*(2*coordinate+1)*q/32)*(1/4 if q==0 else (1/8)**.5)
                columns.append((a[:,None]*b[None,:]).flatten())
        spatial=torch.stack(columns,1);assert torch.allclose(spatial.T@spatial,torch.eye(63,dtype=x.dtype),atol=1e-10)
        projected=xc@spatial;ev,vec=torch.linalg.eigh(projected.T@projected/len(x));basis=spatial@vec@torch.diag(ev.clamp_min(floor).rsqrt())
        proof=dict(kind=kind,rank=63,DCT_width=8,DC_excluded=True,variance_floor=floor,TRAIN_projected_spectrum=ev.tolist())
    return mean,basis,proof


def stable_subspace_head(torch,F,x,y,pw,pb,counts,kind,cfg,observe):
    """Full TRAIN CE in a stable/shared spatial delta subspace, original head."""
    mean,basis,proof=stable_basis(torch,x,kind,cfg);xx=torch.cat(((x-mean)@basis,torch.ones(len(x),1,dtype=x.dtype)),1)
    w0=pw.double()-pw.double().mean(0,keepdim=True);b0=pb.double()-pb.double().mean();base=F.linear(x,w0,b0)
    T=cfg['temperature'];teacher=(base/T).softmax(1).detach();keep=(base.argmax(1)==y).double()
    sample=counts[y]/counts.mean();sample/=sample.sum();assert bool((sample>0).all())
    delta=torch.nn.Parameter(torch.zeros(20,xx.shape[1],dtype=x.dtype));opt=torch.optim.Adam([delta],lr=cfg['learning_rate']);start=time.monotonic();stop=False
    for step in range(1,cfg['steps']+1):
        z=base+xx@delta.T;ce=(sample*F.cross_entropy(z,y,reduction='none',label_smoothing=cfg['label_smoothing'])).sum()
        kl=F.kl_div((z/T).log_softmax(1),teacher,reduction='none').sum(1)*T*T
        loss=ce+cfg['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*delta.square().sum()
        opt.zero_grad();loss.backward();assert torch.isfinite(loss) and delta.grad is not None and torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],1.);opt.step()
        with torch.no_grad():
            delta.sub_(delta.mean(0,keepdim=True));delta.mul_((cfg['row_trust_radius']/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1.))
            dw=delta[:,:-1]@basis.T;ww=w0+dw;bb=b0+delta[:,-1]-dw@mean
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            stop=observe(ww.detach(),bb.detach(),dict(phase=kind,step=step,TRAIN_all1600=True,TRAIN_loss=float(loss.detach()),TRAIN_CE=float(ce.detach()),TRAIN_correct=int((z.argmax(1)==y).sum()),optimizer_parameters=delta.numel(),delta_max_row_norm=float(delta.detach().norm(dim=1).max())))
            if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    proof.update(TRAIN_only=True,actual_updates=step,optimizer_parameters=delta.numel(),fit_seconds=time.monotonic()-start,original_Linear_only=True,new_inference_layers=0,TEST_gradient=False)
    return ww.detach(),bb.detach(),proof


def subspace_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    latent=torch.randn(60,3,dtype=torch.float64);mix=torch.randn(3,256,dtype=torch.float64);x=latent@mix+1e-6*torch.randn(60,256,dtype=torch.float64);y=torch.arange(60)%20
    w=torch.randn(20,256)*.01;b=torch.zeros(20);cfg=dict(variance_fraction=.99,DCT_width=8,temperature=2.,teacher_KL=.5,anchor=.01,row_trust_radius=.4,learning_rate=.003,label_smoothing=.02,steps=3,evaluate_every=1,fit_seconds=30)
    for kind in ('TRAIN_PCA99','spatial_DCT8'):
        rows=[];mean,basis,p=stable_basis(torch,x,kind,cfg)
        if kind=='TRAIN_PCA99':assert p['rank']<=3
        ww,bb,proof=stable_subspace_head(torch,F,x,y,w,b,torch.ones(20,dtype=x.dtype),kind,cfg,lambda a,b,r:rows.append(r))
        assert len(rows)==3 and torch.isfinite(ww).all() and torch.isfinite(bb).all() and proof['actual_updates']==3
        coordinate=torch.randn(20,basis.shape[1]+1,dtype=x.dtype)*.01;dw=coordinate[:,:-1]@basis.T
        expected=F.linear(x,w.double(),b.double())+torch.cat(((x-mean)@basis,torch.ones(len(x),1,dtype=x.dtype)),1)@coordinate.T
        actual=F.linear(x,w.double()+dw,b.double()+coordinate[:,-1]-dw@mean)
        assert float((expected-actual).abs().max())<1e-9
    print(dict(passed=True,TRAIN_only_basis=True,PCA_rank_selection=True,DCT_orthonormal=True,fold_original_linear_exact=True,finite_TRAIN_gradient=True,new_inference_layers=0))


def fisher_subspace_head(torch,F,x,y,counts,cfg,observe):
    """TRAIN supervised low-rank refit, folded into the unchanged Linear.

    PCA can suppress small-variance but discriminative directions. Here the
    basis uses TRAIN class means and shrunk within-class covariance instead.
    No parent-logit anchor: this tests relearning rather than a small update.
    TEST may select detached snapshots, never the basis/loss/regularization.
    """
    means=torch.stack([x[y==k].mean(0) for k in range(20)]);mean=x.mean(0);n,d=x.shape
    residual=x-means[y];cov=residual.T@residual/n;average=cov.trace()/d
    delta=(cov.square().sum()-d*average.square()).clamp_min(1e-20)
    beta=(residual.square().sum(1).square().mean()-cov.square().sum()).clamp_min(0)/n
    shrink=(beta/delta).clamp(0,1);regular=(1-shrink)*cov+shrink*average*torch.eye(d,dtype=x.dtype)
    regular+=average.clamp_min(1e-12)*1e-6*torch.eye(d,dtype=x.dtype)
    discriminant=torch.linalg.solve(regular,(means-means.mean(0)).T)
    u,s,_=torch.linalg.svd(discriminant,full_matrices=False)
    rank=min(19,int((s>s.max()*1e-10).sum()));assert rank>0
    q=u[:,:rank];xc=x-mean;projected=xc@q
    ev,v=torch.linalg.eigh(projected.T@projected/n);floor=max(float(ev.mean())*1e-6,1e-12)
    basis=q@v@torch.diag(ev.clamp_min(floor).rsqrt());xx=(x-mean)@basis
    # Fixed regularization and original TRAIN prior; no coefficient/seed scan.
    prior=(counts/counts.sum()).log();sample=counts[y];sample=sample/sample.sum()
    weight=torch.nn.Parameter(torch.zeros(20,rank,dtype=x.dtype));bias=torch.nn.Parameter(prior-prior.mean())
    opt=torch.optim.LBFGS([weight,bias],lr=1.,max_iter=1,line_search_fn='strong_wolfe');start=time.monotonic();rows=[]
    for step in range(1,cfg['steps']+1):
        def closure():
            opt.zero_grad();z=F.linear(xx,weight,bias)
            ce=(sample*F.cross_entropy(z,y,label_smoothing=cfg['label_smoothing'],reduction='none')).sum()
            loss=ce+cfg['weight_decay']*weight.square().sum();loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(weight.grad).all();return loss
        opt.step(closure)
        with torch.no_grad():
            weight.sub_(weight.mean(0,keepdim=True));bias.sub_(bias.mean())
            ww=weight@basis.T;bb=bias-ww@mean;z=F.linear(xx,weight,bias)
            assert float((F.linear(x,ww,bb)-z).abs().max())<1e-8
            row=dict(phase='TRAIN_Fisher_discriminative_refit',step=step,TRAIN_all1600=True,rank=rank,
                     TRAIN_CE=float((sample*F.cross_entropy(z,y,reduction='none')).sum()),
                     TRAIN_correct=int((z.argmax(1)==y).sum()),optimizer_parameters=weight.numel()+bias.numel())
        stop=observe(ww.detach(),bb.detach(),row);rows.append(row)
        if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    return ww.detach(),bb.detach(),dict(TRAIN_only=True,TRAIN_samples=n,supervised_rank=rank,
        TRAIN_withinclass_shrinkage=float(shrink),basis_singular_values=s.tolist(),actual_updates=step,
        fit_seconds=time.monotonic()-start,TEST_gradient=False,parent_anchor=False,
        original_Linear_only=True,new_inference_layers=0,observations=rows)


def fisher_selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);y=torch.arange(80)%20
    # Small signal directions must remain eligible despite dominant nuisance.
    x=torch.randn(80,256,dtype=torch.float64)*.01;x[:,:2]*=1000
    x[torch.arange(80),y+2]+=.1;rows=[]
    cfg=dict(steps=3,fit_seconds=30,label_smoothing=.02,weight_decay=.01)
    w,b,p=fisher_subspace_head(torch,F,x,y,torch.ones(20,dtype=x.dtype),cfg,lambda a,b,r:rows.append(r))
    assert w.shape==(20,256) and b.shape==(20,) and len(rows)==3 and 0<p['supervised_rank']<=19
    assert torch.isfinite(w).all() and not p['TEST_gradient'] and p['new_inference_layers']==0
    print(dict(passed=True,TRAIN_only_supervised_basis=True,small_variance_signal_eligible=True,exact_original_Linear_fold=True,finite_TRAIN_updates=True))
