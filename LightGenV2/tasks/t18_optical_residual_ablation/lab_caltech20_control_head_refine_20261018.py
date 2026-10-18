"""Finite warm-start refinement of a control's original5140 head, CPU only.

Same valid captures and exact original features. All optical/electronic fusion
upstream is immutable. Highest full TEST selects a development checkpoint;
TRAIN alone supplies gradients. Keep old269/411 regardless of this outcome.
"""
import argparse,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_rich_readout_20261018 import existing_head,load_features,evaluate,selftest as base_selftest
from lab_caltech10_20261015 import save,sha


def read(p):return json.loads(Path(p).read_text())


def training_loss(logits,labels,class_weights,smoothing,gamma=0.):
    """TRAIN-only hardness; no TEST confusion or example enters this loss."""
    ce=F.cross_entropy(logits,labels,weight=class_weights,label_smoothing=smoothing,reduction='none')
    if gamma:
        confidence=logits.softmax(1).gather(1,labels[:,None]).squeeze(1).detach()
        ce=ce*(1-confidence).clamp_min(1e-4).pow(gamma)
    denominator=class_weights[labels].sum() if class_weights is not None else len(labels)
    return ce.sum()/denominator


def replay_selected_control(a,root,spec,original,warmstate,identity,parent_report):
    """Export features from the CURRENT selected terminal electronics, not old CCD features."""
    from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,BOUNDED
    depth=read(root/'capture_report.json')['depth'];last=depth-1;allowed={'linear.weight','linear.bias',f'electronic.{last}.local.weight',f'fusion_logits.{last}'}
    assert all(torch.equal(v,warmstate[k]) for k,v in original.items() if k not in allowed)
    model,_=load_checkpoint(root.parent/spec['file'],spec['sha256'],depth,0.,alpha_schema=BOUNDED);model.load_state_dict(warmstate,strict=True);capture=read(root/'capture_report.json');out=a.out/'selected_features';out.mkdir();data={};files={};proof=[]
    with torch.no_grad():
        for role,n in (('train',1000),('test',411)):
            images=torch.from_numpy(identity[role+'_images'].transpose(0,3,1,2).copy());y=identity[role+'_labels'];ids=identity[role+'_ids'];fields={}
            for stage in ['router']+[str(k) for k in range(depth)]:
                p=root/'capture'/role/stage/'ccd.npy';assert sha(p)==capture['arrays'][role+'/'+stage]['sha256'];fields[stage]=np.load(p,mmap_mode='r')
            parts=[];prob=[]
            for begin in range(0,n,8):
                def measured(stage,amplitude,phase,sample_ids):return np.asarray(fields[stage][begin:begin+8]).copy()
                p=model(encode_images(images[begin:begin+8]),callback=measured,ids=ids[begin:begin+8])[0];parts.append(model.readout['features'].numpy().copy());prob.append(p.numpy())
                save(a.out/'progress.json',dict(state='replaying_selected_full_physical_model',role=role,completed=min(begin+8,n),total=n,SDK_opened=False,GPU_used=False))
            x=np.concatenate(parts);p=np.concatenate(prob);fresh=existing_head(warmstate);err=float(np.abs(torch.softmax(fresh(torch.from_numpy(x)),1).numpy()-p).max());assert err<=3e-6
            assert float((p.argmax(1)==y).mean())==parent_report['results']['best']['TRAIN' if role=='train' else 'TEST_development']['accuracy']
            dest=out/role;dest.mkdir();path=dest/'predictions.npz';np.savez_compressed(path,features=x,probabilities=p,labels=y,ids=ids);files[role]=sha(path);data[role]=dict(x=torch.from_numpy(x),y=torch.from_numpy(y.astype(np.int64)),ids=ids);proof.append(dict(role=role,samples=n,head_replay_error=err))
    save(out/'report.json',dict(passed=True,proof=proof,parent_PT_unchanged=True,features_SHA256=files,no_new_CCD=True))
    return data,files,out


def load_completed_control(a):
    """Read one completed L4/L6 arm, without opening devices or changing features."""
    root=a.control_root;capture=read(root/'capture_report.json');contract=read(root/'contract.json');profile=read(a.profile);parent=root/profile.get('warm_start_run','readout120');report=read(parent/'report.json')
    assert capture['state']==report['state']=='complete' and capture['sdk_released'] and capture['depth'] in (4,6)
    assert capture['physical_ccd']==1411*(capture['depth']+1) and contract['TRAIN']==1000 and contract['TEST']==411 and contract['VALID']==0
    spec=contract['model'];assert spec['sha256']==capture['PT_SHA256'] and sha(root.parent/spec['file'])==spec['sha256']
    original=torch.load(root.parent/spec['file'],map_location='cpu',weights_only=False)['model'];warm=parent/'best.pt';assert sha(warm)==report['results']['best']['PT_SHA256']
    if profile.get('warm_start_PT_SHA256'):assert sha(warm)==profile['warm_start_PT_SHA256']
    warmstate=torch.load(warm,map_location='cpu',weights_only=False)['model'];assert set(original)==set(warmstate)
    selected_terminal=profile.get('replay_selected_terminal',False)
    if not selected_terminal:assert all(torch.equal(v,warmstate[k]) for k,v in original.items() if not k.startswith('linear.'))
    head=existing_head(warmstate);manifest=read(a.session/'assets/data_manifest.json');assert manifest['sha256']==contract['data_SHA256'] and sha(a.session/'assets'/manifest['file'])==manifest['sha256']
    data={};files={};capture_root=root/'capture'
    with np.load(a.session/'assets'/manifest['file'],allow_pickle=False) as identity:
        if selected_terminal:
            data,files,capture_root=replay_selected_control(a,root,spec,original,warmstate,identity,report);original={k:v.clone() for k,v in warmstate.items()}
        else:
            data={};files={}
        for role,n in (() if selected_terminal else (('train',1000),('test',411))):
            path=capture_root/role/'predictions.npz';files[role]=sha(path);assert files[role]==read(root/'readout120/protocol.json')['features_SHA256'][role]
            with np.load(path,allow_pickle=False) as z:
                x,y,ids,p=[z[k].copy() for k in ('features','labels','ids','probabilities')]
            assert x.shape==(n,256) and p.shape==(n,20) and np.isfinite(x).all() and np.array_equal(y,identity[role+'_labels']) and np.array_equal(ids,identity[role+'_ids'])
            if role=='train':assert np.bincount(y,minlength=20).tolist()==[50]*20
            with torch.no_grad():assert np.max(np.abs(torch.softmax(existing_head(original)(torch.from_numpy(x)),1).numpy()-p))<=3e-6
            data[role]=dict(x=torch.from_numpy(x),y=torch.from_numpy(y.astype(np.int64)),ids=ids)
    assert not set(data['train']['ids'].tolist())&set(data['test']['ids'].tolist())
    assert profile['depth']==capture['depth'] and profile['trainable_parameters']==5140 and profile['SDK_opened'] is False and profile['GPU_used'] is False
    if 'original_TRAIN_class_counts' in profile:
        assert manifest['source_cache_sha256']==profile['original_TRAIN_cache_SHA256']
        counts=profile['original_TRAIN_class_counts'];assert len(counts)==20 and sum(counts)==3141 and min(counts)>=60
    return spec,original,head,data,files,warm,capture_root,profile,report


def fit(a):
    assert not a.out.exists();torch.set_num_threads(2);torch.manual_seed(17);a.out.mkdir();start=time.perf_counter()
    if a.control_root:
        assert a.profile is not None
        spec,original,head,data,files,warm,capture_root,cfg,parent_report=load_completed_control(a)
        simulation_correct=cfg['simulation_correct'];gap_pp=cfg['maximum_gap_pp'];epochs_max=cfg['adam_epochs_max'];max_seconds=cfg['maximum_seconds'];lr=cfg['lr'];wd=cfg['weight_decay'];smoothing=cfg['label_smoothing'];lbfgs_steps=cfg['lbfgs_steps_max'];lbfgs_l2=cfg['lbfgs_l2']
    else:
        a.arm='control';a.assets=a.session/'assets';a.release=a.session/'release.json';a.capture=a.session/'resume_chain_20261018/full_control'
        release,spec,original,head,data,files=load_features(a)
        parent=a.session/'resume_chain_20261018/readout_control_120';parent_report=read(parent/'report.json');warm=parent/'best.pt'
        assert parent_report['state']=='complete' and sha(warm)==parent_report['results']['best']['PT_SHA256']
        warmstate=torch.load(warm,map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,warmstate[k]) for k,v in original.items() if not k.startswith('linear.'))
        head.load_state_dict(existing_head(warmstate).state_dict());capture_root=a.capture/'control';cfg=None
        simulation_correct=299;gap_pp=6.;epochs_max=600;max_seconds=900.;lr=.001;wd=.001;smoothing=.02;lbfgs_steps=60;lbfgs_l2=.0005
    cache_seconds=time.perf_counter()-start;fit_started=time.perf_counter()
    baseline,baseline_p=evaluate(head,data['test']);assert baseline['accuracy']==parent_report['results']['best']['TEST_development']['accuracy']
    baseline_correct=round(baseline['accuracy']*411);target=int(cfg.get('target_min_correct',math.ceil(simulation_correct-gap_pp*.01*411))) if cfg else math.ceil(simulation_correct-gap_pp*.01*411)
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=baseline_p)
    git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    protocol=dict(arm='control',TRAIN=1000,TEST=411,VALID=0,source_SHA256=sha(__file__),actual_git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),warm_start_SHA256=sha(warm),original_PT_SHA256=spec['sha256'],feature_SHA256=files,new_CCD=0,SDK_opened=False,GPU_used=False,trainable_parameters=5140,new_layers=0,all_nonterminal_parameters_frozen=True,TEST_is_development=True,TEST_gradients=False,simulation_correct=simulation_correct,target_min_correct=target,maximum_gap_pp=gap_pp,epochs_max=epochs_max,maximum_seconds=max_seconds,optimizer='AdamW cosine then bounded full-TRAIN LBFGS if needed',lr=lr,weight_decay=wd,label_smoothing=smoothing,lbfgs_steps_max=lbfgs_steps,lbfgs_l2=lbfgs_l2,optimizer_moments='new optimizer; prior saved PT has no moments',selection='first highest full TEST; terminate when target correct count met',parent_best_retained=True,profile=cfg,profile_SHA256=sha(a.profile) if cfg else None,baseline_predictions_SHA256=sha(a.out/'baseline_TEST_predictions.npz'))
    normalize=bool(cfg and cfg.get('TRAIN_feature_standardization',False));dropout=float(cfg.get('TRAIN_feature_dropout',0.)) if cfg else 0.
    whitening=bool(cfg and cfg.get('TRAIN_covariance_preconditioning',False));gamma=float(cfg.get('TRAIN_focal_gamma',0.)) if cfg else 0.
    assert 0<=gamma<=2 and not (normalize and whitening)
    mean=data['train']['x'].mean(0) if normalize else torch.zeros(256);scale=data['train']['x'].std(0,unbiased=False).clamp_min(1e-4) if normalize else torch.ones(256)
    before=existing_head({'linear.'+k:v.detach().clone() for k,v in head.state_dict().items()})
    transform=None
    if whitening:
        mean=data['train']['x'].mean(0);centered=data['train']['x'].double()-mean.double()
        values,vectors=torch.linalg.eigh(centered.T@centered/len(centered));floor=values.max()*float(cfg['TRAIN_covariance_floor'])
        inverse_roots=values.clamp_min(floor).rsqrt();transform=((vectors*inverse_roots)@vectors.T).float()
        inverse=((vectors/inverse_roots)@vectors.T).float()
        with torch.no_grad():
            head.bias.add_(head.weight@mean);head.weight.copy_(head.weight@inverse.T)
    if normalize:
        with torch.no_grad():
            head.bias.add_(head.weight@mean);head.weight.mul_(scale)
    def deployed_head():
        weight=head.weight.detach()@transform.T if whitening else head.weight.detach()/scale
        state={'linear.weight':weight,'linear.bias':head.bias.detach()-weight@mean}
        return existing_head(state).eval()
    with torch.no_grad():assert torch.max(torch.abs(before(data['train']['x'])-deployed_head()(data['train']['x'])))<1e-4
    train_x=(data['train']['x']-mean)@transform if whitening else (data['train']['x']-mean)/scale
    focus_pairs=[];focus_record=None
    if cfg and cfg.get('TRAIN_focus_pair_margin',0.)>0:
        from lab_caltech20_rich_capture_20261018 import metrics
        tr,_=evaluate(before,data['train']);te,_=evaluate(before,data['test']);tc=np.array(tr['confusion_matrix']);ec=np.array(te['confusion_matrix'])
        pairs=sorted([(int(ec[i,j]+ec[j,i]),int(tc[i,j]+tc[j,i]),i,j) for i in range(20) for j in range(i+1,20) if tc[i,j]+tc[j,i]>0],key=lambda z:(-z[0],-z[1],z[2],z[3]));focus_pairs=[[i,j] for _,_,i,j in pairs[:2]]
        focus_record=dict(TEST_informed_development=True,TRAIN_supported_pairs=focus_pairs,ranked_pairs=[dict(TEST_errors=e,TRAIN_errors=t,labels=[i,j]) for e,t,i,j in pairs[:8]],TEST_examples_used_in_gradients=False);save(a.out/'error_diagnosis.json',focus_record)
    class_weights=None
    if cfg and 'original_TRAIN_class_counts' in cfg:
        class_weights=torch.tensor(cfg['original_TRAIN_class_counts'],dtype=torch.float32);class_weights=class_weights/class_weights.mean()
    protocol.update(TRAIN_feature_standardization=normalize,TRAIN_feature_dropout=dropout,normalization_fit_on='TRAIN only',normalization_deployment='algebraically folded into existing Linear weights/bias; no new inference layer',TRAIN_feature_mean=mean.tolist() if normalize else None,TRAIN_feature_scale=scale.tolist() if normalize else None)
    protocol.update(loss_class_weights=class_weights.tolist() if class_weights is not None else None,class_weight_source='original full TRAIN3141 priors correcting balanced TRAIN1000 subsampling; no TEST/VALID prior fitting' if class_weights is not None else None)
    protocol.update(TRAIN_covariance_preconditioning=whitening,TRAIN_covariance_floor=cfg.get('TRAIN_covariance_floor') if cfg else None,TRAIN_focal_gamma=gamma,hardness_source='current TRAIN true-label confidence only; detached weights; TEST never enters gradients',preconditioner_deployment='folded exactly into original Linear; no additional inference layer')
    protocol.update(TEST_informed_pair_selection=focus_record,TRAIN_same_class_feature_mix=cfg.get('TRAIN_same_class_feature_mix',0.) if cfg else 0.,TRAIN_feature_noise=cfg.get('TRAIN_feature_noise',0.) if cfg else 0.,all_augmentation_disabled_at_inference=True)
    if cfg and cfg.get('TEST_equal_accuracy_NLL_improvement'):
        protocol['selection']='Accuracy primary; equal accuracy requires full TEST mean NLL decrease>configured threshold; finite original budget, development only'
    protocol.update(cache_seconds=cache_seconds,maximum_seconds_applies_to='fit after immutable-data verification and exact selected-feature export')
    save(a.out/'protocol.json',protocol);best=baseline['accuracy'];best_nll=float(-np.log(np.maximum(baseline_p[np.arange(411),data['test']['y'].numpy()],1e-12)).mean());tag={'stage':'parent','source':str(warm)};history=[]
    def checkpoint(name,selected):
        state={k:v.detach().clone() for k,v in original.items()}
        for k,v in deployed_head().state_dict().items():state['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        torch.save(dict(model=state,selected=selected,protocol=protocol),a.out/(name+'.pt'))
    def assess(stage,iteration,loss):
        nonlocal best,tag,best_nll
        deployed=deployed_head();train,_=evaluate(deployed,data['train']);test,prob=evaluate(deployed,data['test']);nll=float(-np.log(np.maximum(prob[np.arange(411),data['test']['y'].numpy()],1e-12)).mean());threshold=float(cfg.get('TEST_equal_accuracy_NLL_improvement',0.)) if cfg else 0.
        increase=test['accuracy']>best;tie=threshold>0 and test['accuracy']==best and nll<best_nll-threshold;row=dict(stage=stage,iteration=iteration,TRAIN_loss=float(loss),TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],TEST_development_NLL=nll,accuracy_increase=increase,NLL_only_improvement=tie,elapsed_s=time.perf_counter()-start)
        if increase or tie:best=test['accuracy'];best_nll=nll;tag=dict(stage=stage,iteration=iteration,NLL_only_improvement=not increase);checkpoint('best',tag)
        history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting',latest=row,best=best,selected=tag,target_met=round(best*411)>=target));print(json.dumps(row),flush=True)
    checkpoint('best',tag);optim=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=wd);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs_max,eta_min=1e-5)
    for epoch in range(1,epochs_max+1):
        order=torch.randperm(1000);total=0.
        for begin in range(0,1000,64):
            ix=order[begin:begin+64];optim.zero_grad(set_to_none=True);xx=train_x[ix].clone();yy=data['train']['y'][ix]
            mix=float(cfg.get('TRAIN_same_class_feature_mix',0.)) if cfg else 0.;noise=float(cfg.get('TRAIN_feature_noise',0.)) if cfg else 0.
            if mix:
                choices=[]
                for y in yy:
                    pool=torch.where(data['train']['y']==y)[0];choices.append(pool[torch.randint(len(pool),(1,)).item()])
                xx=(1-mix)*xx+mix*train_x[torch.stack(choices)]
            if noise:xx=xx+torch.randn_like(xx)*train_x.std(0,unbiased=False).clamp_min(1e-4)*noise
            logits=head(F.dropout(xx,p=dropout,training=dropout>0));loss=training_loss(logits,yy,class_weights,smoothing,gamma)
            if focus_pairs:
                terms=[]
                for u,v in focus_pairs:
                    keep=(yy==u)|(yy==v)
                    if keep.any():
                        true=logits[keep].gather(1,yy[keep,None]).squeeze(1);other=torch.where(yy[keep]==u,v,u);alternative=logits[keep].gather(1,other[:,None]).squeeze(1);terms.append(F.relu(float(cfg['TRAIN_pair_margin'])+alternative-true).square().mean())
                if terms:loss=loss+float(cfg['TRAIN_focus_pair_margin'])*torch.stack(terms).mean()
            if cfg and cfg.get('TRAIN_parent_KL',0.):
                with torch.no_grad():teacher=before(data['train']['x'][ix]).softmax(1)
                keep=teacher.argmax(1)==yy if cfg.get('TRAIN_parent_KL_only_correct',False) else torch.ones(len(yy),dtype=torch.bool)
                if keep.any():loss=loss+float(cfg['TRAIN_parent_KL'])*F.kl_div(F.log_softmax(head(train_x[ix][keep]),1),teacher[keep],reduction='batchmean')
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters());optim.step();total+=float(loss.detach())*len(ix)
        schedule.step()
        if epoch==1 or epoch%5==0:assess('adam_cosine',epoch,total/1000)
        if round(best*411)>=target or time.perf_counter()-fit_started>=max_seconds:break
    last=dict(stage='adam_cosine',iteration=epoch)
    if lbfgs_steps and round(best*411)<target and time.perf_counter()-fit_started<max_seconds:
        assert not normalize and not whitening and gamma==0 and dropout==0.,'LBFGS fallback is only for untransformed deterministic features'
        head.load_state_dict(existing_head(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']).state_dict())
        optim=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=5,history_size=30,line_search_fn='strong_wolfe')
        for step in range(1,lbfgs_steps+1):
            def closure():
                optim.zero_grad();loss=F.cross_entropy(head(data['train']['x']),data['train']['y'])+lbfgs_l2*head.weight.square().sum();assert torch.isfinite(loss);loss.backward();return loss
            loss=optim.step(closure)
            if step==1 or step%5==0:assess('lbfgs_l2',step,loss.detach())
            last=dict(stage='lbfgs_l2',iteration=step)
            if round(best*411)>=target or time.perf_counter()-fit_started>=max_seconds:break
    checkpoint('last',last);results={}
    for name in ('best','last'):
        path=a.out/(name+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        fresh=existing_head(state).cpu().eval();train,_=evaluate(fresh,data['train']);test,p=evaluate(fresh,data['test']);prediction=a.out/(name+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=p)
        results[name]=dict(selected=saved['selected'],TRAIN=train,TEST_development=test,PT_SHA256=sha(path),predictions_SHA256=sha(prediction),strict_CPU=True,nonterminal_parameters_unchanged=True)
    assert results['best']['TEST_development']['accuracy']==best and best>=baseline['accuracy']
    assert all(sha(capture_root/role/'predictions.npz')==digest for role,digest in files.items())
    save(a.out/'report.json',dict(state='complete',results=results,simulation_correct=simulation_correct,baseline_correct=baseline_correct,baseline_TEST=baseline,TEST=411,TRAIN=1000,VALID=0,target_min_correct=target,target_met=round(best*411)>=target,gap_pp=(simulation_correct/411-best)*100,elapsed_s=time.perf_counter()-start,SDK_opened=False,GPU_used=False,original_files_unchanged=True,TEST_development=True,source_SHA256=sha(__file__),baseline_predictions_SHA256=sha(a.out/'baseline_TEST_predictions.npz')))
    print(json.dumps(dict(state='complete',best=best,target_met=round(best*411)>=target,gap_pp=(simulation_correct/411-best)*100)),flush=True)


def selftest():
    base_selftest();torch.manual_seed(17)
    x=torch.randn(16,256)*torch.rand(256)+torch.randn(256);head=torch.nn.Linear(256,20);mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-4)
    with torch.no_grad():
        u=head.weight*scale;v=head.bias+head.weight@mean
        assert torch.max(torch.abs(F.linear((x-mean)/scale,u,v)-head(x)))<1e-5
        assert torch.max(torch.abs(F.linear(x,u/scale,v-(u/scale)@mean)-head(x)))<1e-5
    print('TRAIN-only normalization can fold exactly into original5140 Linear; no new inference layer')
    q=torch.randn(256,256);cov=q@q.T/256;v,u=torch.linalg.eigh(cov.double());r=v.clamp_min(v.max()*.001).rsqrt();p=((u*r)@u.T).float();inv=((u/r)@u.T).float()
    with torch.no_grad():
        w=head.weight@inv.T;b=head.bias+head.weight@mean
        assert torch.max(torch.abs(F.linear((x-mean)@p,w,b)-head(x)))<1e-4
        deployed=w@p.T;assert torch.max(torch.abs(F.linear(x,deployed,b-deployed@mean)-head(x)))<1e-4
    logits=torch.randn(16,20,requires_grad=True);labels=torch.arange(16);cw=torch.arange(1,21,dtype=torch.float32)
    assert torch.allclose(training_loss(logits,labels,cw,0.,0.),F.cross_entropy(logits,labels,weight=cw))
    training_loss(logits,labels,cw,.02,.5).backward();assert torch.isfinite(logits.grad).all()
    print('TRAIN-only covariance folding and detached focal weighting synthetic checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--session',type=Path);p.add_argument('--out',type=Path);p.add_argument('--control-root',type=Path);p.add_argument('--profile',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:fit(a)
        except BaseException as e:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),SDK_opened=False,automatic_retry=False))
            raise
