"""Bounded TRAIN-only focal correction and read-only class/domain diagnostics.

No new inference layer: condition optimizer coordinates using TRAIN and fold
the update into original Linear. TEST may choose TRAIN indices/checkpoints;
the differentiable fitting function accepts no TEST features or TEST labels.
"""
import time


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
