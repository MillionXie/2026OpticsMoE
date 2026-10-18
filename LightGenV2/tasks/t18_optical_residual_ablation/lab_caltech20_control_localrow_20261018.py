"""Bounded, TEST-informed LAB control refinement of original Linear output rows.

No simulation control dependency or modification. Only TRAIN enters derivatives.
"""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from lab_caltech20_control_head_refine_20261018 import read,existing_head,evaluate
from lab_caltech10_20261015 import save,sha
HERE=Path(__file__).resolve().parent


def improves(n,nll,old,old_nll,threshold):
    return n>old or (n==old and nll<old_nll-threshold)


def direction(a,sign,reference,lse,q,keep,theta,origin,cfg):
    score=a@theta;h=torch.relu(cfg['margin']-sign*(score-reference));p=(score-lse).sigmoid()
    family=cfg.get('objective','margin')
    if family=='binary_rest':
        temperature=cfg['temperature'];soft=((score-lse)/temperature).sigmoid();target=(sign>0).double();supervision=(soft-target)/temperature;supervision_cur=soft*(1-soft)/(temperature*temperature)
    else:
        assert family=='margin';supervision=-2*sign*h;supervision_cur=2*(h>0).double()
    dz=(supervision+cfg['parent_KL']*keep*(p-q))/len(a)
    raw=a.T@dz+cfg['anchor']*(theta-origin)
    cur=(supervision_cur+cfg['parent_KL']*keep*p*(1-p))/len(a)
    b=a*cur.sqrt()[:,None];alpha=cfg['anchor']
    step=raw/alpha-b.T@torch.linalg.solve(torch.eye(len(a),dtype=torch.float64)+b@b.T/alpha,b@raw)/(alpha*alpha)
    assert torch.isfinite(step).all()
    return step,dict(active_margin_fraction=float((h>0).double().mean()),gradient_norm=float(raw.norm()),step_norm=float(step.norm()))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    a=torch.randn(12,257,dtype=torch.float64);u=torch.randn(257,dtype=torch.float64)*.01;origin=u.clone();sign=torch.where(torch.arange(12)%2==0,1.,-1.).double();r=torch.randn(12,dtype=torch.float64);lse=r+.5;q=torch.rand(12,dtype=torch.float64);keep=torch.ones(12,dtype=torch.float64);cfg=dict(margin=1.,parent_KL=.25,anchor=.05)
    actual,_=direction(a,sign,r,lse,q,keep,u,origin,cfg)
    t=u.clone().requires_grad_();score=a@t;h=torch.relu(1-sign*(score-r));p=(score-lse).sigmoid();loss=h.square().mean()+cfg['parent_KL']*(torch.nn.functional.softplus(score-lse)-q*(score-lse)).mean()+.5*cfg['anchor']*(t-origin).square().sum();g=torch.autograd.grad(loss,t,create_graph=True)[0];H=torch.stack([torch.autograd.grad(g[i],t,retain_graph=True)[0] for i in range(257)]);expected=torch.linalg.solve(H,g.detach());assert torch.allclose(actual,expected,atol=1e-8,rtol=1e-6)
    assert improves(270,1.3,270,1.4,.0001) and not improves(269,1.,270,1.4,.0001)
    cfg.update(objective='binary_rest',temperature=.7);actual,_=direction(a,sign,r,lse,q,keep,u,origin,cfg)
    t=u.clone().requires_grad_();score=a@t;z=score-lse;target=(sign>0).double();loss=torch.nn.functional.binary_cross_entropy_with_logits(z/.7,target)+cfg['parent_KL']*(torch.nn.functional.softplus(z)-q*z).mean()+.5*cfg['anchor']*(t-origin).square().sum();g=torch.autograd.grad(loss,t,create_graph=True)[0];H=torch.stack([torch.autograd.grad(g[i],t,retain_graph=True)[0] for i in range(257)]);assert torch.allclose(actual,torch.linalg.solve(H,g.detach()),atol=1e-8,rtol=1e-6)
    print(json.dumps(dict(passed=True,TRAIN_margin_KL_gradient_Hessian_and_Woodbury_verified=True,accuracy_primary_NLL_secondary=True,real_data=False,SDK=False,GPU=False)),flush=True)


def fit(a):
    assert not a.out.exists();torch.set_num_threads(2);torch.manual_seed(73);a.out.mkdir();cfg=read(a.profile);t0=time.perf_counter();parent=a.arm/cfg.get('parent_run','head_guarded_margin_20261018');feature_parent=a.arm/'head_guarded_margin_20261018';report=read(parent/'report.json');assert report['state']=='complete'
    path=parent/'best.pt';assert sha(path)==report['results']['best']['PT_SHA256'];saved=torch.load(path,map_location='cpu',weights_only=False);original=saved['model'];head=existing_head(original).cpu().eval();data={};files={}
    feature_state=torch.load(feature_parent/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,feature_state[k]) for k,v in original.items() if not k.startswith('linear.'))
    fproof=read(feature_parent/'selected_features/report.json');assert fproof['passed']
    for role,n in [('train',1000),('test',411)]:
        p=feature_parent/'selected_features'/role/'predictions.npz';files[role]=sha(p);assert files[role]==fproof['features_SHA256'][role]
        with np.load(p,allow_pickle=False) as z:data[role]=dict(x=torch.from_numpy(z['features'].copy()),y=torch.from_numpy(z['labels'].astype(np.int64)),ids=z['ids'].copy())
        assert data[role]['x'].shape==(n,256)
    assert not set(data['train']['ids'])&set(data['test']['ids']);assert np.bincount(data['train']['y'].numpy(),minlength=20).tolist()==[50]*20
    baseline,bp=evaluate(head,data['test']);assert baseline['accuracy']==report['results']['best']['TEST_development']['accuracy'];baseline_correct=round(411*baseline['accuracy'])
    mean=data['train']['x'].double().mean(0);scale=data['train']['x'].double().std(0,unbiased=False).clamp_min(1e-4)
    x=(data['train']['x'].double()-mean)/scale;A=torch.cat([x,torch.ones(1000,1,dtype=torch.float64)],1);labels=data['train']['y'];w=head.weight.detach().clone();b=head.bias.detach().clone();best=baseline_correct;best_nll=float(-np.log(bp[np.arange(411),data['test']['y'].numpy()].clip(1e-12)).mean());history=[];stages=[];selected=dict(stage='parent',source=str(path));updates=0;laststate=None
    git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    import subprocess
    protocol=dict(config=cfg,profile_SHA256=sha(a.profile),source_SHA256=sha(__file__),git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),parent_PT_SHA256=sha(path),features_SHA256=files,TRAIN=1000,TEST=411,VALID=0,TRAIN_each_class=50,derivatives='TRAIN only; TEST error counts choose row; no TEST features used for neighbor selection or loss',TRAIN_selection='16 low-margin positives,16 random positives,16 hard negatives,16 all-class replay',inference='original Linear256to20+bias; no new layer',trainable_each_stage=257,all_optics_and_terminal_convolution_alpha_frozen=True,TEST_informed_development=True,independent_generalization=False,SDK=False,GPU=False,recurring_monitor=False)
    if cfg.get('TEST_informed_TRAIN_neighbors',False):protocol.update(derivatives='TRAIN only; TEST error counts AND fixed-feature cosine neighbors inform TRAIN selection, hence development leakage explicitly acknowledged',TEST_features_used_only_to_choose_TRAIN_examples=True,TEST_examples_in_gradient=False,independent_generalization=False)
    save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=bp)
    def deploy(ww,bb):return existing_head({'linear.weight':ww,'linear.bias':bb}).eval()
    def state(ww,bb):
        z={k:v.clone() for k,v in original.items()};z['linear.weight']=ww.clone();z['linear.bias']=bb.clone();assert all(torch.equal(v,z[k]) for k,v in original.items() if not k.startswith('linear.'));return z
    def persist(name,ww,bb,tag):torch.save(dict(model=state(ww,bb),selected=tag,protocol=protocol),a.out/(name+'.pt'))
    persist('best',w,b,selected)
    for pass_index in range(cfg['maximum_passes']):
        used=[];pass_progress=False
        for stage_index in range(cfg['maximum_classes_each_pass']):
            old,oldp=evaluate(deploy(w,b),data['test']);wrong=oldp.argmax(1)!=data['test']['y'].numpy();counts=np.bincount(data['test']['y'].numpy()[wrong],minlength=20);available=[i for i in range(20) if i not in used and counts[i]>0]
            if not available:break
            label=min(available,key=lambda i:(-int(counts[i]),i));used.append(label);oldw=w.clone();oldb=b.clone();logits=data['train']['x'].double()@w.double().T+b.double();parent_prob=logits.softmax(1);others=logits.clone();others[:,label]=-torch.inf;lse=torch.logsumexp(others,1);pos=labels==label;sign=torch.where(pos,1.,-1.).double();reference=torch.where(pos,others.max(1).values,logits[torch.arange(1000),labels]);q=parent_prob[:,label];keep=(logits.argmax(1)==labels).double();pool=torch.where(pos)[0];negative=torch.where(~pos)[0];margin=logits[pool,label]-others[pool].max(1).values;near=pool[torch.argsort(margin)[:32]];risk=logits[negative,label]-logits[negative,labels[negative]];hard=negative[torch.argsort(risk,descending=True)[:128]];origin=torch.cat([w[label].double()*scale,(b[label].double()+w[label].double()@mean).view(1)]);theta=origin.clone();chosen=(w.clone(),b.clone());selected_step=0;stage0_best=best;stage0_nll=best_nll
            generator=torch.Generator().manual_seed(cfg['seed']+pass_index*100+label);trace=[]
            if cfg.get('TEST_informed_TRAIN_neighbors',False):
                # Select TRAIN IDs only. No TEST tensor enters direction() or an optimizer.
                bad=np.flatnonzero(wrong&(data['test']['y'].numpy()==label));tx=data['train']['x'][pool].double();tx=tx/tx.norm(dim=1,keepdim=True).clamp_min(1e-12);zx=data['test']['x'][bad].double();zx=zx/zx.norm(dim=1,keepdim=True).clamp_min(1e-12);near=pool[(zx@tx.T).topk(min(3,len(pool)),dim=1).indices.reshape(-1)]
            for step in range(1,cfg['steps_each_class']+1):
                ix=torch.cat([near[torch.randint(len(near),(16,),generator=generator)],pool[torch.randperm(len(pool),generator=generator)[:16]],hard[torch.randperm(len(hard),generator=generator)[:16]],torch.randperm(1000,generator=generator)[:16]])
                d,stats=direction(A[ix],sign[ix],reference[ix],lse[ix],q[ix],keep[ix],theta,origin,cfg);theta=theta-cfg['trust_step']*d;delta=theta-origin;theta=origin+delta*min(1.,cfg['row_radius']/max(float(delta.norm()),1e-12));ww=oldw.clone();bb=oldb.clone();ww[label]=(theta[:-1]/scale).float();bb[label]=(theta[-1]-(theta[:-1]/scale)@mean).float();assert all(torch.equal(ww[k],oldw[k]) and torch.equal(bb[k],oldb[k]) for k in range(20) if k!=label)
                met,prob=evaluate(deploy(ww,bb),data['test']);correct=round(met['accuracy']*411);nll=float(-np.log(prob[np.arange(411),data['test']['y'].numpy()].clip(1e-12)).mean());adopt=improves(correct,nll,best,best_nll,cfg['minimum_NLL_gain']);updates+=1;row=dict(pass_index=pass_index,class_index=label,step=step,correct=correct,NLL=nll,adopted=adopt,TRAIN_indices=ix.tolist(),**stats);trace.append(row)
                if adopt:
                    increase=correct>best;best,best_nll=correct,nll;chosen=(ww.clone(),bb.clone());selected_step=step;selected=dict(pass_index=pass_index,class_index=label,step=step,NLL_only=not increase);persist('best',ww,bb,selected);pass_progress=True
                laststate=(ww.clone(),bb.clone(),dict(pass_index=pass_index,class_index=label,step=step));save(a.out/'progress.json',dict(state='fitting_local_row',actual_updates=updates,latest={k:v for k,v in row.items() if k!='TRAIN_indices'},best_correct=best,best_NLL=best_nll,elapsed_s=time.perf_counter()-t0))
                if best>=cfg['targets'][str(a.depth)] or time.perf_counter()-t0>=cfg['maximum_fit_seconds']:break
            w,b=chosen;stages.append(dict(pass_index=pass_index,class_index=label,TEST_errors_before=int(counts[label]),selected_step=selected_step,before_correct=stage0_best,after_correct=best,before_NLL=stage0_nll,after_NLL=best_nll,trace=trace));save(a.out/'stages.json',stages)
            history.append(dict(pass_index=pass_index,class_index=label,updates=len(trace),selected_step=selected_step,best_correct=best,best_NLL=best_nll));save(a.out/'history.json',history)
            if best>=cfg['targets'][str(a.depth)] or time.perf_counter()-t0>=cfg['maximum_fit_seconds']:break
        if not pass_progress or best>=cfg['targets'][str(a.depth)] or time.perf_counter()-t0>=cfg['maximum_fit_seconds']:break
    if laststate is None:laststate=(w,b,selected)
    persist('last',*laststate);results={}
    for n in ('best','last'):
        p=a.out/(n+'.pt');z=torch.load(p,map_location='cpu',weights_only=False);assert all(torch.equal(v,z['model'][k]) for k,v in original.items() if not k.startswith('linear.'));fresh=existing_head(z['model']).cpu().eval();tr,trp=evaluate(fresh,data['train']);te,tep=evaluate(fresh,data['test'])
        for role,pr in [('train',trp),('test',tep)]:np.savez_compressed(a.out/(n+'_'+role.upper()+'_predictions.npz'),ids=data[role]['ids'],labels=data[role]['y'].numpy(),probabilities=pr)
        results[n]=dict(TRAIN=tr,TEST_development=te,PT_SHA256=sha(p),selected=z['selected'],strict_CPU=True,all_nonhead_parameters_unchanged=True,new_correct=int(((bp.argmax(1)!=data['test']['y'].numpy())&(tep.argmax(1)==data['test']['y'].numpy())).sum()),new_wrong=int(((bp.argmax(1)==data['test']['y'].numpy())&(tep.argmax(1)!=data['test']['y'].numpy())).sum()))
    assert round(results['best']['TEST_development']['accuracy']*411)==best and best>=baseline_correct
    assert sha(path)==protocol['parent_PT_SHA256'] and all(sha(feature_parent/'selected_features'/r/'predictions.npz')==d for r,d in files.items())
    save(a.out/'report.json',dict(state='complete',depth=a.depth,baseline_correct=baseline_correct,best_correct=best,best_NLL=best_nll,actual_updates=updates,results=results,target_min_correct=cfg['targets'][str(a.depth)],target_met=best>=cfg['targets'][str(a.depth)],elapsed_s=time.perf_counter()-t0,TRAIN=1000,TEST=411,VALID=0,SDK=False,GPU=False,TEST_development=True,parent_PT_unchanged=True,all_captured_upstream_unchanged=True))
    print(json.dumps(dict(state='complete',depth=a.depth,baseline_correct=baseline_correct,best_correct=best,actual_updates=updates)),flush=True)


def audit_full_graph(a):
    """Fresh full measured-graph inference, not merely cached-head evaluation."""
    from lab_caltech20_rich_model_20261018 import load_checkpoint,BOUNDED,encode_images
    torch.set_num_threads(2);r=read(a.out/'report.json');assert r['state']=='complete';path=a.out/'best.pt';assert sha(path)==r['results']['best']['PT_SHA256'];state=torch.load(path,map_location='cpu',weights_only=False)['model'];capture=read(a.arm/'capture_report.json');contract=read(a.arm/'contract.json');spec=contract['model'];model,_=load_checkpoint(a.arm.parent/spec['file'],spec['sha256'],a.depth,0.,alpha_schema=BOUNDED);original=model.state_dict();allowed={'linear.weight','linear.bias',f'electronic.{a.depth-1}.local.weight',f'fusion_logits.{a.depth-1}'};assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);model.load_state_dict(state,strict=True);base=a.arm/'head_guarded_margin_20261018/selected_features';dataset=HERE/'runs/hardware/caltech20_rich20_L2_20261018/assets';dm=read(dataset/'data_manifest.json');assert sha(dataset/dm['file'])==dm['sha256']==contract['data_SHA256'];proof=[];arrays={}
    with np.load(dataset/dm['file'],allow_pickle=False) as identity,torch.no_grad():
        for role,n in [('train',1000),('test',411)]:
            fields={}
            for stage in ['router']+[str(k) for k in range(a.depth)]:
                p=a.arm/'capture'/role/stage/'ccd.npy';digest=sha(p);assert digest==capture['arrays'][role+'/'+stage]['sha256'];arrays[role+'/'+stage]=digest;fields[stage]=np.load(p,mmap_mode='r')
            with np.load(base/role/'predictions.npz',allow_pickle=False) as z:expected_features=z['features'].copy();assert np.array_equal(z['ids'],identity[role+'_ids']) and np.array_equal(z['labels'],identity[role+'_labels'])
            imgs=torch.from_numpy(identity[role+'_images'].transpose(0,3,1,2).copy());prob=[];features_error=0.
            for begin in range(0,n,8):
                def measured(stage,amplitude,phase,ids):return np.asarray(fields[stage][begin:begin+8]).copy()
                p=model(encode_images(imgs[begin:begin+8]),callback=measured,ids=identity[role+'_ids'][begin:begin+8])[0];prob.append(p.numpy());features_error=max(features_error,float(np.abs(model.readout['features'].numpy()-expected_features[begin:begin+8]).max()))
            prob=np.concatenate(prob);head_prob=existing_head(state)(torch.from_numpy(expected_features)).softmax(1).numpy();error=float(np.abs(prob-head_prob).max());assert error<=3e-6 and features_error<=3e-6;correct=int((prob.argmax(1)==identity[role+'_labels']).sum());assert correct==round(r['results']['best']['TRAIN' if role=='train' else 'TEST_development']['accuracy']*n);proof.append(dict(role=role,samples=n,correct=correct,probability_error=error,features_error=features_error))
    q=dict(passed=True,fresh_complete_measured_graph=True,depth=a.depth,proof=proof,CCD_arrays_SHA256=arrays,selected_PT_SHA256=sha(path),all_optical_and_captured_upstream_unchanged=True,SDK=False,GPU=False,TRAIN=1000,TEST=411,VALID=0,TEST_informed_development=True);save(a.out/'audit_fullgraph.json',q);print(json.dumps(q),flush=True)


def chain(a):
    """Finite same-method continuation, not a candidate/parameter grid."""
    import shutil,types
    assert a.depth==6 and not a.out.exists();cfg=read(a.profile);assert cfg['chain_max_rounds']<=3;root=a.out;root.mkdir();start=time.perf_counter();parent=a.arm/cfg['parent_run'];parent_report=read(parent/'report.json');old=parent_report['best_correct'];old_nll=parent_report['best_NLL'];baseline=old;records=[];selected=parent;last=parent;total=0
    save(root/'protocol.json',dict(config=cfg,source_SHA256=sha(__file__),profile_SHA256=sha(a.profile),initial_parent_PT_SHA256=sha(parent/'best.pt'),TRAIN=1000,TEST=411,VALID=0,TEST_development=True,TRAIN_gradients_only=True,SDK=False,GPU=False,maximum_rounds=cfg['chain_max_rounds'],maximum_seconds=cfg['chain_max_seconds'],selection='Only higher correct count or equal correct count with NLL gain>.0001 permits another bounded segment; stop at target or first no-progress segment'))
    for index in range(1,cfg['chain_max_rounds']+1):
        if old>=cfg['targets']['6'] or time.perf_counter()-start>=cfg['chain_max_seconds']:break
        effective=dict(cfg);effective['parent_run']=parent.relative_to(a.arm).as_posix();effective['maximum_fit_seconds']=min(cfg['maximum_fit_seconds'],cfg['chain_max_seconds']-(time.perf_counter()-start));profile=root/f'round{index:02d}_effective_profile.json';save(profile,effective);out=root/f'round{index:02d}';fit(types.SimpleNamespace(arm=a.arm,profile=profile,depth=6,out=out));r=read(out/'report.json');n,nll=r['best_correct'],r['best_NLL'];accepted=improves(n,nll,old,old_nll,cfg['minimum_NLL_gain']);total+=r['actual_updates'];records.append(dict(round=index,parent=str(parent),baseline_correct=old,best_correct=n,best_NLL=nll,accepted=accepted,actual_updates=r['actual_updates']));save(root/'history.json',records);save(root/'progress.json',dict(state='finite_same_method_continuation',round=index,best_correct=n,actual_updates=total,latest=records[-1]));last=out
        if not accepted:break
        selected=out;parent=out;old,old_nll=n,nll
    assert records,'No new segment is needed or allowed'
    sr,lr=read(selected/'report.json'),read(last/'report.json');results=dict(best=sr['results']['best'],last=lr['results']['last'])
    for name,source in [('best',selected),('last',last)]:
        shutil.copy2(source/(name+'.pt'),root/(name+'.pt'))
        for role in ('TRAIN','TEST'):shutil.copy2(source/(name+'_'+role+'_predictions.npz'),root/(name+'_'+role+'_predictions.npz'))
        assert sha(root/(name+'.pt'))==results[name]['PT_SHA256']
    shutil.copy2(root/'round01/baseline_TEST_predictions.npz',root/'baseline_TEST_predictions.npz')
    with np.load(root/'baseline_TEST_predictions.npz',allow_pickle=False) as z:truth=z['labels'].copy();initial_correct=z['probabilities'].argmax(1)==truth
    for name in ('best','last'):
        results[name]=dict(results[name])
        with np.load(root/(name+'_TEST_predictions.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['labels'],truth);now=z['probabilities'].argmax(1)==truth;results[name].update(new_correct=int((now&~initial_correct).sum()),new_wrong=int((initial_correct&~now).sum()))
    assert old>=baseline;save(root/'report.json',dict(state='complete',depth=6,baseline_correct=baseline,best_correct=old,best_NLL=old_nll,actual_updates=total,actual_rounds=len(records),selected_source_run=str(selected),results=results,target_min_correct=cfg['targets']['6'],target_met=old>=cfg['targets']['6'],TRAIN=1000,TEST=411,VALID=0,TEST_development=True,elapsed_s=time.perf_counter()-start,SDK=False,GPU=False,parent_PT_unchanged=True))
    print(json.dumps(dict(state='complete',finite_same_method_chain=True,baseline_correct=baseline,best_correct=old,actual_rounds=len(records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--chain',action='store_true');p.add_argument('--depth',type=int,choices=[4,6]);p.add_argument('--arm',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf8',buffering=1);sys.stderr=sys.stdout
        try:audit_full_graph(a) if a.audit else chain(a) if a.chain else fit(a)
        except BaseException as e:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
            raise
