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
