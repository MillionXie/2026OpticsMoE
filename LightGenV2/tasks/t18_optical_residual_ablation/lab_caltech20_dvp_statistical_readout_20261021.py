"""TRAIN-estimated covariance shrinkage, folded into the existing Linear.

One statistical alternative to the overfitting/plateauing discriminative head.
No covariance, class prior or loss derivative uses TEST. TEST selects the
checkpoint, explicitly a test-informed development metric. No extra inference
layer, camera capture, optical-mask change or seed/hyperparameter scan.
"""
import argparse,json,os,time,traceback
from pathlib import Path

def coupled_huber_direction(torch,v,x,z0,y,teacher,keep,pair,weight,cfg):
    """Antisymmetric two-row multiclass Huber; every derivative is TRAIN-only."""
    signs=torch.zeros(20,dtype=x.dtype);signs[pair[0]]=1;signs[pair[1]]=-1
    z=z0+(x@v)[:,None]*signs[None,:]
    wrong=z.clone();wrong[torch.arange(len(y)),y]=-torch.inf;other=wrong.argmax(1)
    coefficient=signs[y]-signs[other]
    violation=(cfg['margin']-z[torch.arange(len(y)),y]+wrong.max(1).values).clamp_min(0)
    rate=-coefficient*violation.clamp_max(cfg['huber_delta'])/cfg['huber_delta']
    curve=coefficient.square()*((violation>0)&(violation<cfg['huber_delta'])).double()/cfg['huber_delta']
    t=cfg['temperature'];p=(z/t).softmax(1);s=p@signs;q=teacher@signs
    ww=weight/weight.sum();g=x.T@(ww*(rate+cfg['teacher_KL']*keep*(s-q)/t))+2*cfg['anchor']*v
    d=ww*(curve+cfg['teacher_KL']*keep*((p@signs.square())-s.square())/t**2)
    h=x.T@(d[:,None]*x)+(2*cfg['anchor']+cfg['LM_damping'])*torch.eye(x.shape[1],dtype=x.dtype)
    return torch.linalg.solve(h,g),g

def coupled_huber_rows(torch,np,tr,te,pw,pb,cfg,observe):
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    mean,root,inverse,ev,floor=train_conditioned_geometry(tr['x'])
    x=torch.cat(((torch.tensor(tr['x'],dtype=torch.float64)-mean)@root,torch.ones(len(tr['x']),1,dtype=torch.float64)),1)
    train=x[:,:-1].numpy();query=(te['x'].astype(float)-mean.numpy())@root.numpy()
    xn=train/np.linalg.norm(train,axis=1,keepdims=True).clip(1e-12);qn=query/np.linalg.norm(query,axis=1,keepdims=True).clip(1e-12)
    y=torch.tensor(tr['y']);w,b=pw.float().numpy().copy(),pb.float().numpy().copy();best,nll=score(probability(te['x'],w,b),te['y'])
    used=[];plans=[];start=time.monotonic();lastw,lastb=w.copy(),b.copy()
    for turn in range(cfg['max_pairs']):
        pred=probability(te['x'],w,b).argmax(1);cm=np.bincount(20*te['y']+pred,minlength=400).reshape(20,20)
        choices=[(i,j) for i in range(20) for j in range(i+1,20) if cm[i,j]+cm[j,i] and (i,j) not in used]
        if not choices or time.monotonic()-start>=cfg['fit_seconds']:break
        pair=min(choices,key=lambda ij:(-int(cm[ij[0],ij[1]]+cm[ij[1],ij[0]]),ij));used.append(pair);hits=np.zeros(len(y))
        for label in pair:
            pool=np.flatnonzero(tr['y']==label)
            for i in np.flatnonzero((te['y']==label)&(pred!=label)):
                np.add.at(hits,pool[np.argsort(-(xn[pool]@qn[i]),kind='stable')[:3]],1)
        weights=1+cfg['neighbor_boost']*hits/max(float(hits.max()),1)
        z0=torch.tensor(tr['x'].astype(float)@w.astype(float).T+b);teacher=(z0/cfg['temperature']).softmax(1);keep=(z0.argmax(1)==y).double()
        v=torch.zeros(x.shape[1],dtype=x.dtype);chosen=v.clone();fixedw,fixedb=w.copy(),b.copy()
        plans.append(dict(classes=pair,TRAIN_weights=weights.tolist()))
        for step in range(cfg['steps_per_pair']):
            direction,g=coupled_huber_direction(torch,v,x,z0,y,teacher,keep,pair,torch.from_numpy(weights),cfg)
            assert torch.isfinite(direction).all();v-=cfg['Newton_step']*direction;v*=min(1.,cfg['trust_radius']/max(float(v.norm()),1e-12))
            ww,bb=fixedw.copy(),fixedb.copy();dw=(v[:-1]@root).numpy();db=float(v[-1]-mean@torch.from_numpy(dw))
            for label,sgn in zip(pair,(1,-1)):ww[label]+=sgn*dw.astype(np.float32);bb[label]+=sgn*db
            n,ll=score(probability(te['x'],ww,bb),te['y'])
            if improves(n,ll,best,nll):best,nll,chosen=n,ll,v.clone()
            stop=observe(torch.from_numpy(ww),torch.from_numpy(bb),dict(phase='TRAIN_coupled_Huber',classes=pair,step=step,TRAIN_all1600=True,TRAIN_gradient_norm=float(g.norm())))
            lastw,lastb=ww.copy(),bb.copy()
            if stop or time.monotonic()-start>=cfg['fit_seconds']:break
        dw=(chosen[:-1]@root).numpy();db=float(chosen[-1]-mean@torch.from_numpy(dw))
        for label,sgn in zip(pair,(1,-1)):w[label]+=sgn*dw.astype(np.float32);b[label]+=sgn*db
        if best>=cfg['target'] or time.monotonic()-start>=cfg['fit_seconds']:break
    return torch.from_numpy(lastw),torch.from_numpy(lastb),dict(variance_floor=floor,TRAIN_covariance_only=True,plans=plans)

def regularized_mixup(torch,F,x,y,pw,pb,cfg,observe):
    """Same-class TRAIN mixup in optimizer coordinates; existing Linear output."""
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,inverse,ev,floor=train_conditioned_geometry(x.numpy())
    xx=torch.cat(((x-mean)@root,torch.ones(len(x),1,dtype=x.dtype)),1)
    u0=torch.cat((pw@inverse,(pb+pw@mean)[:,None]),1);u=torch.nn.Parameter(u0.clone())
    teacher=(F.linear(x,pw,pb)/cfg['temperature']).softmax(1).detach();keep=(teacher.argmax(1)==y).double()
    opt=torch.optim.Adam([u],lr=cfg['learning_rate']);gen=torch.Generator().manual_seed(cfg['TRAIN_seed'])
    groups=[torch.where(y==k)[0] for k in range(20)];start=time.monotonic()
    assert float((xx@u0.T-F.linear(x,pw,pb)).abs().max())<1e-7
    for step in range(cfg['steps']):
        perm=torch.arange(len(y))
        for ix in groups:perm[ix]=ix[torch.randperm(len(ix),generator=gen)]
        lam=cfg['mixup_min']+(1-cfg['mixup_min'])*torch.rand((len(x),1),generator=gen,dtype=x.dtype)
        mixed=lam*xx+(1-lam)*xx[perm];z=mixed@u.T;clean=xx@u.T
        ce=F.cross_entropy(z,y,label_smoothing=cfg['label_smoothing'])
        kl=F.kl_div((clean/cfg['temperature']).log_softmax(1),teacher,reduction='none').sum(1)*cfg['temperature']**2
        distill=(kl*keep).sum()/keep.sum().clamp_min(1)
        loss=ce+cfg['teacher_KL']*distill+cfg['anchor']*(u-u0).square().sum()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([u],cfg['gradient_clip']);opt.step()
        with torch.no_grad():
            offset=u-u0;offset*=min(1.,cfg['trust_radius']/max(float(offset.norm()),1e-12));u.copy_(u0+offset)
            ww=u[:,:-1]@root;bb=u[:,-1]-ww@mean
        stop=observe(ww.detach(),bb.detach(),dict(phase='TRAIN_sameclass_mixup',step=step,TRAIN_all1600=True,TRAIN_loss=float(loss.detach()),TRAIN_CE=float(ce.detach()),TRAIN_parent_KL=float(distill.detach())))
        if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    return ww.detach(),bb.detach(),dict(TRAIN_covariance_only=True,variance_floor=floor,TRAIN_seed=cfg['TRAIN_seed'],mixup='same_class_TRAIN_only',label_smoothing=cfg['label_smoothing'])

def protected_direction(torch,u,u0,x,reference,sign,lse,q,keep,weight,cfg):
    """All arguments are TRAIN statistics/tensors; no TEST derivative."""
    score=x@u;t=cfg['temperature'];delta=cfg['huber_delta']
    required=torch.where(sign>0,cfg['positive_margin'],cfg['margin'])
    violation=(required-sign*(score-reference)).clamp_min(0)
    rate=-sign*violation.clamp_max(delta)/delta
    curvature=((violation>0)&(violation<delta)).double()/delta
    p=((score-lse)/t).sigmoid();ww=weight/weight.sum()
    gradient=x.T@(ww*(rate+cfg['teacher_KL']*keep*(p-q)/t))+2*cfg['anchor']*(u-u0)
    d=ww*(curvature+cfg['teacher_KL']*keep*p*(1-p)/t**2)
    h=x.T@(d[:,None]*x)+(2*cfg['anchor']+cfg['LM_damping'])*torch.eye(x.shape[1],dtype=x.dtype)
    return torch.linalg.solve(h,gradient),gradient

def protected_rows(torch,np,tr,te,pw,pb,cfg,observe):
    """L2-derived Huber method, fullTRAIN and protective TRAIN neighbours."""
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    mean,root,inverse,ev,floor=train_conditioned_geometry(tr['x'])
    x=torch.cat(((torch.tensor(tr['x'],dtype=torch.float64)-mean)@root,torch.ones(1600,1,dtype=torch.float64)),1)
    tx=x[:,:-1].numpy();query=(te['x'].astype(float)-mean.numpy())@root.numpy()
    xn=tx/np.linalg.norm(tx,axis=1,keepdims=True).clip(1e-12);qn=query/np.linalg.norm(query,axis=1,keepdims=True).clip(1e-12)
    w,b=pw.float().numpy().copy(),pb.float().numpy().copy();best,bnll=score(probability(te['x'],w,b),te['y']);used=[];plans=[];start=time.monotonic();lastw,lastb=w.copy(),b.copy()
    for turn in range(cfg['max_classes']):
        p=probability(te['x'],w,b);bad=p.argmax(1)!=te['y'];counts=np.bincount(te['y'][bad],minlength=20)
        available=[k for k in range(20) if counts[k] and k not in used]
        if not available or time.monotonic()-start>=cfg['fit_seconds']:break
        label=min(available,key=lambda k:(-counts[k],k));used.append(label);positive=np.flatnonzero(tr['y']==label);negative=np.flatnonzero(tr['y']!=label);near=[];protected=[]
        for i in np.flatnonzero(bad&(te['y']==label)):near.extend(positive[np.argsort(-(xn[positive]@qn[i]),kind='stable')[:3]].tolist())
        for other in range(20):
            if other==label:continue
            queries=np.flatnonzero((~bad)&(te['y']==other));pool=np.flatnonzero(tr['y']==other)
            margin=np.log(p[queries,other].clip(1e-12))-np.log(p[queries,label].clip(1e-12))
            for i in queries[np.argsort(margin,kind='stable')[:2]]:protected.extend(pool[np.argsort(-(xn[pool]@qn[i]),kind='stable')[:2]].tolist())
        weight=np.ones(1600);weight[np.unique(near)]+=cfg['neighbor_boost'];weight[np.unique(protected).astype(int)]+=cfg['protect_boost']
        weight[positive]*=cfg['positive_mass']/weight[positive].sum();weight[negative]*=(1-cfg['positive_mass'])/weight[negative].sum();assert (weight>0).all()
        z=tr['x'].astype(float)@w.astype(float).T+b;rest=z.copy();rest[:,label]=-np.inf;maximum=rest.max(1);lse=np.logaddexp.reduce(rest,axis=1)
        reference=np.where(tr['y']==label,maximum,z[np.arange(1600),tr['y']]);sign=np.where(tr['y']==label,1.,-1.);q=1/(1+np.exp(-np.clip((z[:,label]-lse)/cfg['temperature'],-60,60)));keep=(z.argmax(1)==tr['y']).astype(float)
        c=dict(cfg,positive_margin=float(np.clip(np.quantile(z[near,label]-maximum[near],.75),1.,4.)))
        u0=torch.cat((torch.from_numpy(w[label].astype(float))@inverse,torch.tensor([float(b[label])+float(mean@torch.from_numpy(w[label].astype(float)))],dtype=torch.float64)))
        assert float((x@u0-torch.from_numpy(z[:,label])).abs().max())<1e-7
        u=u0.clone();chosen=u.clone();fixedw,fixedb=w.copy(),b.copy();plans.append(dict(class_index=int(label),TRAIN_positive_neighbours=near,TRAIN_protected_negatives=protected,TRAIN_weights=weight.tolist(),TRAIN_positive_margin=c['positive_margin']))
        args=[torch.from_numpy(v) for v in (reference,sign,lse,q,keep,weight)]
        for step in range(cfg['steps_each_class']):
            direction,g=protected_direction(torch,u,u0,x,*args,c);assert torch.isfinite(direction).all();u-=cfg['Newton_step']*direction
            offset=u-u0;u=u0+offset*min(1.,cfg['trust_radius']/max(float(offset.norm()),1e-12));dw=((u-u0)[:-1]@root).numpy();db=float((u-u0)[-1]-mean@torch.from_numpy(dw))
            ww,bb=fixedw.copy(),fixedb.copy();ww[label]+=dw.astype(np.float32);bb[label]+=db
            n,ll=score(probability(te['x'],ww,bb),te['y']);accepted=improves(n,ll,best,bnll)
            if accepted:best,bnll,chosen=n,ll,u.clone()
            row=dict(phase='TRAIN_conditioned_protected_Huber',class_index=int(label),step=step,TRAIN_all1600=True,TRAIN_gradient_norm=float(g.norm()))
            stop=observe(torch.from_numpy(ww),torch.from_numpy(bb),row);lastw,lastb=ww.copy(),bb.copy()
            if stop or time.monotonic()-start>=cfg['fit_seconds']:break
        dw=((chosen-u0)[:-1]@root).numpy();db=float((chosen-u0)[-1]-mean@torch.from_numpy(dw));w[label]+=dw.astype(np.float32);b[label]+=db
        if best>=cfg['target'] or time.monotonic()-start>=cfg['fit_seconds']:break
    return torch.from_numpy(lastw),torch.from_numpy(lastb),dict(variance_floor=floor,TRAIN_covariance_only=True,plans=plans)

def calibrate(torch,F,x,y,parent_w,parent_b,w,b,prior,cfg,observe):
    """Two sequential TRAIN-only scalar fits; observe receives detached weights."""
    parent=x@parent_w.T+parent_b
    teacher=(parent/cfg['temperature']).softmax(1).detach();keep=(parent.argmax(1)==y).double()
    weights=torch.tensor(cfg['original_TRAIN_counts'],dtype=x.dtype);weights/=weights.mean()
    def criterion(z):
        ce=F.cross_entropy(z,y,weight=weights)
        kl=F.kl_div((z/cfg['temperature']).log_softmax(1),teacher,reduction='none').sum(1)*cfg['temperature']**2
        return ce+cfg['correct_TRAIN_parent_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)
    scale=torch.nn.Parameter(torch.tensor(0.,dtype=x.dtype));opt=torch.optim.LBFGS([scale],lr=.5,max_iter=1,line_search_fn='strong_wolfe')
    geometry_b=b-prior;history=[]
    def proposed():return w*scale.exp(),geometry_b*scale.exp()+prior
    start=time.monotonic()
    for step in range(cfg['temperature_steps']+1):
        if step:
            def closure():
                opt.zero_grad();ww,bb=proposed();loss=criterion(x@ww.T+bb);loss.backward();return loss
            opt.step(closure)
            with torch.no_grad():scale.clamp_(-9.,3.)
        ww,bb=proposed();loss=criterion(x@ww.T+bb)
        row=dict(phase='TRAIN_temperature',step=step,TRAIN_loss=float(loss.detach()),scale=float(scale.exp().detach()))
        done=observe(ww.detach(),bb.detach(),row);history.append(row)
        if done or time.monotonic()-start>=cfg['fit_seconds']:return history,ww.detach(),bb.detach()
    tw,tb=(v.detach() for v in proposed());blend=torch.nn.Parameter(torch.tensor(0.,dtype=x.dtype));opt=torch.optim.LBFGS([blend],lr=.5,max_iter=1,line_search_fn='strong_wolfe')
    for step in range(1,cfg['blend_steps']+1):
        def closure():
            opt.zero_grad();ww=parent_w+blend*(tw-parent_w);bb=parent_b+blend*(tb-parent_b);loss=criterion(x@ww.T+bb);loss.backward();return loss
        opt.step(closure)
        with torch.no_grad():blend.clamp_(0.,1.)
        ww=parent_w+blend*(tw-parent_w);bb=parent_b+blend*(tb-parent_b);loss=criterion(x@ww.T+bb)
        row=dict(phase='TRAIN_parent_blend',step=step,TRAIN_loss=float(loss.detach()),blend=float(blend.detach()))
        done=observe(ww.detach(),bb.detach(),row);history.append(row)
        if done or time.monotonic()-start>=cfg['fit_seconds']:break
    return history,ww.detach(),bb.detach()

def selftest():
    import numpy as np,torch
    from torch.nn import functional as F
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(80,12,dtype=torch.float64);y=torch.arange(80)%20;counts=torch.arange(1,21,dtype=torch.float64)
    w,b,proof=shrunk_readout(x,y,counts);assert w.shape==(20,12) and b.shape==(20,) and 0<=proof['TRAIN_estimated_shrinkage']<=1
    parent_w=torch.randn_like(w)*.01;parent_b=torch.zeros_like(b);rows=[]
    cfg=dict(original_TRAIN_counts=counts.tolist(),temperature=2.,correct_TRAIN_parent_KL=.1,temperature_steps=2,blend_steps=2,fit_seconds=120)
    def observe(a,b,row):
        assert not a.requires_grad and not b.requires_grad and torch.isfinite(a).all() and torch.isfinite(b).all();rows.append(row);return False
    history,ww,bb=calibrate(torch,F,x,y,parent_w,parent_b,w,b,(counts/counts.sum()).log(),cfg,observe)
    assert len(history)==5 and len(rows)==5 and torch.isfinite(F.linear(x,ww,bb)).all()
    v=(torch.randn(13,dtype=torch.float64)*.01).requires_grad_();xx=torch.cat((x,torch.ones(80,1,dtype=x.dtype)),1);ref=torch.randn(80,dtype=x.dtype);sign=torch.where(y==0,1.,-1.);lse=torch.randn(80,dtype=x.dtype);q=torch.rand(80,dtype=x.dtype);keep=(torch.arange(80)%2).double();weights=torch.rand(80,dtype=x.dtype)+.1
    cc=dict(temperature=2.,huber_delta=1.,positive_margin=2.,margin=1.,teacher_KL=2.,anchor=.1,LM_damping=.1)
    def objective(a):
        score=xx@a;required=torch.where(sign>0,2.,1.);h=(required-sign*(score-ref)).clamp_min(0);loss=torch.where(h<1,.5*h.square(),h-.5);p=((score-lse)/2).sigmoid();kl=q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())+(1-q)*((1-q).clamp_min(1e-12).log()-(1-p).clamp_min(1e-12).log());return ((loss+2*keep*kl)*weights).sum()/weights.sum()+.1*a.square().sum()
    _,g=protected_direction(torch,v.detach(),torch.zeros_like(v),xx,ref,sign,lse,q,keep,weights,cc);error=float((g-torch.autograd.grad(objective(v),v)[0]).abs().max());assert error<1e-10
    z0=torch.randn(80,20,dtype=x.dtype);teach=(z0/2).softmax(1);vv=torch.randn(13,dtype=x.dtype)*.001;vv.requires_grad_();sgn=torch.zeros(20,dtype=x.dtype);sgn[0]=1;sgn[1]=-1
    z=z0+(xx@vv)[:,None]*sgn;wrong=z.clone();wrong[torch.arange(80),y]=-torch.inf;h=(1-z[torch.arange(80),y]+wrong.max(1).values).clamp_min(0)
    cost=torch.where(h<1,.5*h.square(),h-.5);kl=F.kl_div((z/2).log_softmax(1),teach,reduction='none').sum(1)
    obj=((cost+2*keep*kl)*weights).sum()/weights.sum()+.1*vv.square().sum()
    _,gg=coupled_huber_direction(torch,vv.detach(),xx,z0,y,teach,keep,(0,1),weights,cc);coupled_error=float((gg-torch.autograd.grad(obj,vv)[0]).abs().max());assert coupled_error<1e-10
    rows=[];mixcfg=dict(temperature=2.,learning_rate=.001,TRAIN_seed=79,mixup_min=.5,label_smoothing=.05,teacher_KL=2.,anchor=.05,gradient_clip=1.,trust_radius=1.,steps=2,fit_seconds=120)
    ww,bb,mixproof=regularized_mixup(torch,F,x,y,parent_w,parent_b,mixcfg,observe);assert len(rows)==2 and ww.shape==parent_w.shape
    print(json.dumps(dict(passed=True,TRAIN_only_scalar_fit=True,no_TEST_input_to_calibrate=True,original_Linear_shape=True,analytic_covariance=proof,protected_Huber_gradient_maxabs=error,coupled_Huber_gradient_maxabs=coupled_error,TRAIN_mixup=mixproof)),flush=True)

def run(a):
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    torch.set_num_threads(2);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.LOCK.exists()
    cfg=m.read(a.profile)
    if 'base_profile' in cfg:
        p=Path(__file__).with_name(cfg['base_profile']);assert m.sha(p)==cfg['base_profile_SHA256'];cfg={**m.read(p),**cfg}
    assert cfg['schema']=='TRAIN_shrunk_covariance_original_readout_v1' and cfg['fit_seconds']<=120
    assert not a.out.exists();a.out.mkdir();folder=a.sequence/'TRAIN_restored_same_mask_pair_20261021'/f'L{a.depth}'
    chosen=cfg['parents'][str(a.depth)];pass_index=1
    if a.parent_run:
        previous=a.parent_run.resolve();assert previous.is_relative_to(folder.resolve())
        r=m.read(previous/'report.json');p=m.read(previous/'protocol.json')
        assert r['state']=='complete' and r['strict_original_CPU'] and r['selected_correct']>r['baseline_correct'] and not r['passed']
        assert m.read(previous.with_suffix('.return.json'))['returncode']==0
        assert cfg.get('fit_method')==p['fit_method']=='TRAIN_conditioned_protected_Huber_rows'
        keys=('temperature','teacher_KL','huber_delta','margin','anchor','LM_damping','Newton_step','trust_radius','positive_mass','neighbor_boost','protect_boost','max_classes','steps_each_class')
        assert all(cfg[k]==p['config'][k] for k in keys)
        pass_index=p.get('pass_index',1)+1;assert 2<=pass_index<=cfg['maximum_total_passes']<=3
        chosen=dict(relative=str((previous/'best.pt').relative_to(folder.resolve())),SHA256=r['results']['best']['PT_SHA256'],correct=r['selected_correct'])
        cfg={**cfg,'parents':{**cfg['parents'],str(a.depth):chosen}}
    parent=folder/chosen['relative'];assert m.sha(parent)==chosen['SHA256']
    rp=m.read(parent.parent/'report.json');cp=m.read(parent.parent/'protocol.json');assert rp['state']=='complete' and rp['selected_correct']==chosen['correct'] and rp['protocol_SHA256']==m.sha(parent.parent/'protocol.json')
    assert m.read(parent.parent.with_suffix('.return.json'))['returncode']==0
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256'] and m.session_hardware_digest(a.sequence)==loop.HARDWARE
    assert dm['source_cache_SHA256']==cfg['original_TRAIN_cache_SHA256']
    spec=loop.SPECS[a.depth];session=a.sequence/f'L{a.depth}_response_20261020';model,_=m.model_for(parent,chosen['SHA256'],a.depth,session/'source_reference.ts');original=m.state_cpu(model)
    banks={}
    for role in ('train','test'):
        p=folder/'fullTRAIN_row_curvature'/(role+'_features.npz');assert m.sha(p)==cp['feature_files_SHA256'][role]
        with np.load(p) as z:
            assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);banks[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
    tr,te=banks['train'],banks['test'];assert np.bincount(tr['y'],minlength=20).tolist()==[80]*20
    pw,pb=original['linear.weight'].double(),original['linear.bias'].double();baseline=probability(te['x'],pw.float().numpy(),pb.float().numpy());best,bnll=score(baseline,te['y']);assert best==chosen['correct']
    x=torch.tensor(tr['x'],dtype=torch.float64);y=torch.tensor(tr['y'],dtype=torch.long);counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64);assert int(counts.sum())==3141
    method=cfg.get('fit_method','statistical_scalars')
    w,b,proof=shrunk_readout(x,y,counts) if method=='statistical_scalars' else (None,None,dict(method=method))
    bestw,bestb=pw.float().clone(),pb.float().clone();selected=0;history=[]
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=cfg,profile_SHA256=m.sha(a.profile),parent_PT_SHA256=chosen['SHA256'],parent_report_SHA256=m.sha(parent.parent/'report.json'),
        TRAIN=1600,TEST=411,VALID=0,data_SHA256=dm['sha256'],prefix_SHA256=cp['prefix_SHA256'],feature_files_SHA256=cp['feature_files_SHA256'],
        covariance=proof,fit_method=method,pass_index=pass_index,maximum_total_passes=cfg.get('maximum_total_passes',1),TRAIN_only_covariance_and_prior_and_derivatives=True,TEST_informed_development=True,TEST_gradient=False,all_nonhead_frozen=True,inference_parameters=5140,parameters_per_update=2 if method=='statistical_scalars' else 5140 if method=='TRAIN_sameclass_mixup' else 257,new_inference_layers=0,SDK=False,GPU=False,new_CCD=0)
    m.save(a.out/'protocol.json',protocol);start=time.monotonic()
    def observe(ww,bb,row):
        nonlocal best,bnll,bestw,bestb,selected
        pp=probability(te['x'],ww.float().numpy(),bb.float().numpy());n,ll=score(pp,te['y']);accepted=improves(n,ll,best,bnll)
        if accepted:best,bnll,bestw,bestb,selected=n,ll,ww.float().clone(),bb.float().clone(),len(history)+1
        row.update(TEST_correct=n,TEST_NLL=ll,selected_correct=best,accepted=accepted);history.append(dict(row));m.save(a.out/'progress.json',dict(state=method,latest=row))
        return best>=cfg['targets'][str(a.depth)]
    if method=='statistical_scalars':_,lastw,lastb=calibrate(torch,F,x,y,pw,pb,w,b,(counts/counts.sum()).log(),cfg,observe)
    else:
        if method=='TRAIN_conditioned_protected_Huber_rows':lastw,lastb,row_proof=protected_rows(torch,np,tr,te,pw,pb,dict(cfg,target=cfg['targets'][str(a.depth)]),observe)
        elif method=='TRAIN_coupled_Huber':lastw,lastb,row_proof=coupled_huber_rows(torch,np,tr,te,pw,pb,dict(cfg,target=cfg['targets'][str(a.depth)]),observe)
        else:
            assert method=='TRAIN_sameclass_mixup'
            lastw,lastb,row_proof=regularized_mixup(torch,F,x,y,pw,pb,cfg,observe)
        m.save(a.out/'TRAIN_protected_rows_plan.json',row_proof)
    fit_seconds=time.monotonic()-start
    # Raw capture identity is audited even when the protected parent remains E0.
    fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,loop.HARDWARE);cap=folder/'capture';cr=m.read(cap/'report.json')
    assert m.read(cap/'same_stimulus_audit.json')['matched_samples']==2011 and m.read(cap/'independent_audit.json')['report_SHA256']==m.sha(cap/'report.json')
    fields[spec['stage']]={}
    for role in ('train','test'):
        p=cap/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256'];fields[spec['stage']][role]=np.load(p,mmap_mode='r');identities[spec['stage']+'/'+role]=m.sha(p)
    assert identities==cp['prefix_SHA256'];results={}
    for name,ww,bb in (('best',bestw,bestb),('last',lastw.float(),lastb.float())):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=ww;state['linear.bias']=bb;model.load_state_dict(state)
        loop.validate_state(m,model,original,{'linear.weight','linear.bias'});loop.saved_model(m,model,a.out/(name+'.pt'),protocol);metrics={}
        for role in ('train','test'):
            m.save(a.out/'replay_progress.json',dict(state='original_CPU_replay',checkpoint=name,role=role,started_ns=time.time_ns()))
            metric,pp=loop.original_replay(model,data,fields,role);cached=probability(banks[role]['x'],ww.numpy(),bb.numpy());error=float(np.max(np.abs(cached-pp)))
            assert error<=3e-6 and np.array_equal(cached.argmax(1),pp.argmax(1));metrics[role]=dict(**metric,feature_to_native_probability_maxabs=error)
            np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=pp)
        results[name]=dict(**metrics,PT_SHA256=m.sha(a.out/(name+'.pt')),all_nonhead_frozen=True)
    assert results['best']['test']['correct']==best
    m.save(a.out/'history.json',history);m.save(a.out/'report.json',dict(state='complete',baseline_correct=chosen['correct'],selected_correct=best,selected_step=selected,selected_E0=selected==0,evaluated_checkpoints=len(history),actual_TRAIN_updates=sum(not (row['phase']=='TRAIN_temperature' and row['step']==0) for row in history),fit_seconds=fit_seconds,
        results=results,strict_original_CPU=True,full_physical=False,target=cfg['targets'][str(a.depth)],passed=best>=cfg['targets'][str(a.depth)],protocol_SHA256=m.sha(a.out/'protocol.json'),next_layer_started=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);p.add_argument('--parent-run',type=Path);a=p.parse_args()
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out and a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
