"""Disclosed error-augmentation and inherited-TEST teacher comparisons.

Student gradients never receive original TEST rows. Error augmentation still
uses139 TEST labels; TEST-trained teacher descendants still inherit leakage.
Nothing here converts that lineage into an independent TEST experiment.
"""
import argparse,json,time,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base


def validate(c):
    continued=c['schema']=='T18_KL_TRAIN_gradient_balance_v1'
    if continued:
        assert c['recipes']==['KL_TRAIN_balance2','KL_TRAIN_balance4']and c['gradient_balance_multipliers']=={'KL_TRAIN_balance2':2.,'KL_TRAIN_balance4':4.}
        assert c['initial_student_correct']==331 and c['teacher_adaptation_development_target']==342
        assert c['student_direct_TEST_gradient']is False and c['teacher_seen_TEST']is True
        assert c['ratio_bounds']==[1/19,19.]and c['stop_after_complete_native_target']is True
    else:
        assert c['schema']=='T18_readout_transfer_disclosed_TEST_lineage_v1'
        assert c['recipes']==['errors139_augmented_only','KL_teacher','DKD_TRAIN_neighborhood','DKD_no_TEST_teacher_control']
    assert (c['TRAIN'],c['replay_former_TEST'],c['fixed_error_source_samples'],c['VALID'])==(1600,411,139,0)
    assert c['TRAIN_loss_fraction']==.35 and c['errors139_augmentation_loss_fraction']==.65 and c['KD_loss_fraction']==.65
    assert c['sameclass_mix_fraction']==.05 and c['noise_std_fraction']==.005 and c['label_smoothing']==.01
    assert c['temperature']==2 and c['DKD_target_weight']==1 and c['DKD_nontarget_weight']==4 and c['teacher_wrong_TRAIN_reliability']==.25
    assert c['preconditioned_delta_L2']==.0005 and c['covariance_floor_fraction']==.01
    assert c['outer_steps']==60 and c['errors139_outer_steps']==120 and c['fit_seconds_per_recipe']==120
    assert c['trainable_parameters']==5140 and c['new_inference_layers']==0 and not any(c[k]for k in('SDK_used','GPU_used','new_capture','replace_original_development_best'))
    assert c['official_TEST_accuracy']is None


def neighborhood_pool(features,labels,ids,c):
    """Original TRAIN plus same-label TRAIN interpolation/noise, no TEST input."""
    import numpy as np
    rng=np.random.default_rng(c['seed']);neighbors=[]
    for i,y in enumerate(labels):
        group=np.flatnonzero(labels==y);group=group[group!=i];assert len(group);neighbors.append(int(rng.choice(group)))
    near=np.asarray(neighbors);mix=(1-c['sameclass_mix_fraction'])*features+c['sameclass_mix_fraction']*features[near]
    noise=features+rng.normal(size=features.shape).astype(np.float32)*features.std(0)*c['noise_std_fraction']
    return dict(features=np.concatenate([features,mix,noise]).astype(np.float32),labels=np.tile(labels,3),source_ids=np.tile(ids,3),source_roles=np.array(['original_TRAIN']*(3*len(labels))),augmentation=np.array(['unmodified']*len(labels)+['sameclass_TRAIN_feature_mix']*len(labels)+['TRAIN_feature_noise']*len(labels)))


def error_augmented_pool(train,labels,ids,error_features,error_labels,error_ids,c):
    """Receives only fixed139 wrong-source rows, returns NO raw TEST rows."""
    import numpy as np
    assert len(error_features)==len(error_labels)==len(error_ids)==c['fixed_error_source_samples']
    rng=np.random.default_rng(c['seed']);a=train/np.linalg.norm(train,axis=1,keepdims=True).clip(1e-12);b=error_features/np.linalg.norm(error_features,axis=1,keepdims=True).clip(1e-12);neighbors=[]
    for i,y in enumerate(error_labels):
        group=np.flatnonzero(labels==y);assert len(group);neighbors.append(int(group[np.argmax(a[group]@b[i])]))
    near=np.asarray(neighbors);mix=(1-c['sameclass_mix_fraction'])*error_features+c['sameclass_mix_fraction']*train[near]
    noise=error_features+rng.normal(size=error_features.shape).astype(np.float32)*train.std(0)*c['noise_std_fraction']
    x=np.concatenate([train,mix,noise]).astype(np.float32);y=np.concatenate([labels,error_labels,error_labels])
    assert np.array_equal(x[:len(train)],train)and np.array_equal(labels[near],error_labels)
    assert np.all(np.any(mix!=error_features,axis=1))and np.all(np.any(noise!=error_features,axis=1))
    return dict(features=x,labels=y,source_ids=np.concatenate([ids,error_ids,error_ids]),source_roles=np.array(['original_TRAIN']*len(train)+['former_TEST_error_derivative']*(2*len(error_ids))),augmentation=np.array(['unmodified']*len(train)+['error_sameclass_TRAIN_mix']*len(error_ids)+['error_feature_noise']*len(error_ids)))


def distillation_loss(student,teacher,labels,kind,c):
    """Per-example KL or target/non-target decoupling on TRAIN inputs only."""
    import torch
    from torch.nn import functional as F
    teacher=teacher.detach();t=c['temperature'];s=student/t;q=teacher/t
    if kind=='KL':return F.kl_div(F.log_softmax(s,1),F.softmax(q,1),reduction='none').sum(1)*t*t
    mask=F.one_hot(labels,num_classes=s.shape[1]).bool();sp=F.softmax(s,1);tp=F.softmax(q,1)
    sy=sp.gather(1,labels[:,None]);ty=tp.gather(1,labels[:,None]);sb=torch.cat([sy,1-sy],1).clamp_min(1e-12);tb=torch.cat([ty,1-ty],1).clamp_min(1e-12)
    target=F.kl_div(sb.log(),tb,reduction='none').sum(1)
    ns=F.log_softmax(s.masked_fill(mask,-1e4),1);nt=F.softmax(q.masked_fill(mask,-1e4),1)
    other=F.kl_div(ns,nt,reduction='none').sum(1)
    reliability=torch.where(teacher.argmax(1)==labels,student.new_tensor(1.),student.new_tensor(c['teacher_wrong_TRAIN_reliability']))
    return (c['DKD_target_weight']*target+c['DKD_nontarget_weight']*other)*t*t*reliability/(c['DKD_target_weight']+c['DKD_nontarget_weight'])


def TRAIN_balance_weights(ce_norm,kd_norm,cosine,multiplier,bounds):
    """Fixed before fitting from TRAIN gradients; no replay labels involved."""
    import math
    assert all(math.isfinite(v)for v in(ce_norm,kd_norm,cosine,multiplier))and ce_norm>=0 and kd_norm>=0 and multiplier in(2.,4.)
    ratio=max(bounds[0],min(bounds[1],multiplier*ce_norm/max(kd_norm,1e-12)))
    return dict(TRAIN_CE_weight=1/(1+ratio),TRAIN_KL_weight=ratio/(1+ratio),TRAIN_CE_gradient_norm=ce_norm,TRAIN_KL_gradient_norm=kd_norm,TRAIN_gradient_cosine=cosine,requested_KL_to_CE_gradient_ratio=multiplier,fixed_KL_to_CE_loss_ratio=ratio,ratio_clipped=ratio!=multiplier*ce_norm/max(kd_norm,1e-12),gradient_statistics_source='original_TRAIN_only',teacher_seen_TEST=True)


def optimize_pool(packet,parent_w,parent_b,teacher_w,teacher_b,recipe,c,evaluate,out,m):
    """Only packet is in closures. Replay callback runs detached outside them."""
    import numpy as np,torch
    from torch.nn import functional as F
    x=torch.tensor(packet['features'],dtype=torch.float64);y=torch.tensor(packet['labels'],dtype=torch.long)
    assert x.shape[1]==256 and set(packet['source_roles'])<=({'original_TRAIN','former_TEST_error_derivative'}if recipe=='errors139_augmented_only'else{'original_TRAIN'})
    w0=torch.tensor(parent_w,dtype=torch.float64);b0=torch.tensor(parent_b,dtype=torch.float64)
    with torch.no_grad():
        centered=x-x.mean(0);cov=centered.T@centered/len(x);floor=c['covariance_floor_fraction']*torch.trace(cov)/256;e,v=torch.linalg.eigh(cov);pre=(v*(e.clamp_min(0)+floor).rsqrt())@v.T
        target=None if recipe=='errors139_augmented_only'else F.linear(x,torch.tensor(teacher_w,dtype=torch.float64),torch.tensor(teacher_b,dtype=torch.float64)).detach()
    delta=torch.nn.Parameter(torch.zeros_like(w0));db=torch.nn.Parameter(torch.zeros_like(b0));opt=torch.optim.LBFGS([delta,db],lr=c.get('LBFGS_lr',1),max_iter=c.get('LBFGS_inner_iterations',5),max_eval=c.get('LBFGS_max_eval',8),history_size=20,line_search_fn='strong_wolfe')
    weights=torch.full((len(x),),1/len(x),dtype=torch.float64)
    if recipe=='errors139_augmented_only':
        n=int((packet['source_roles']=='original_TRAIN').sum());weights[:n]=c['TRAIN_loss_fraction']/n;weights[n:]=c['errors139_augmentation_loss_fraction']/(len(x)-n)
    ce_weight,kd_weight=c['TRAIN_loss_fraction'],c['KD_loss_fraction'];balance=None;ordinary_KL=recipe=='KL_teacher'or recipe.startswith('KL_TRAIN_')
    if recipe.startswith('KL_TRAIN_'):
        assert len(x)==1600 and set(packet['source_roles'])=={'original_TRAIN'}and target is not None
        logits=F.linear(x,w0+delta@pre,b0+db);ce=F.cross_entropy(logits,y,label_smoothing=c['label_smoothing']);kd=distillation_loss(logits,target,y,'KL',c).mean()
        gc=torch.cat([v.reshape(-1)for v in torch.autograd.grad(ce,(delta,db),retain_graph=True)]);gk=torch.cat([v.reshape(-1)for v in torch.autograd.grad(kd,(delta,db))]);nc=float(gc.norm());nk=float(gk.norm());cosine=float((gc@gk)/(gc.norm()*gk.norm()).clamp_min(1e-12))
        balance=TRAIN_balance_weights(nc,nk,cosine,c['gradient_balance_multipliers'][recipe],c['ratio_bounds']);ce_weight,kd_weight=balance['TRAIN_CE_weight'],balance['TRAIN_KL_weight'];m.save(out/'TRAIN_gradient_balance.json',balance)
    def objective():
        z=F.linear(x,w0+delta@pre,b0+db);ce=(F.cross_entropy(z,y,reduction='none',label_smoothing=c['label_smoothing'])*weights).sum()
        kd=z.new_tensor(0.)if target is None else distillation_loss(z,target,y,'KL'if ordinary_KL else'DKD',c).mean()
        loss=ce if target is None else ce_weight*ce+kd_weight*kd
        return loss+c['preconditioned_delta_L2']*(delta.square().sum()+db.square().sum())
    metrics=evaluate(parent_w,parent_b);best=metrics['correct'];nll=metrics['nll'];bw=np.array(parent_w,copy=True);bb=np.array(parent_b,copy=True);selected=calls=0;history=[];started=time.monotonic();budget=c['errors139_outer_steps']if target is None else c['outer_steps']
    interval=c.get('selection_interval');interval_met=False;peak=best
    if interval is not None:assert len(interval)==2 and 0<=interval[0]<=interval[1]<411
    for update in range(1,budget+1):
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True);loss=objective();assert torch.isfinite(loss);loss.backward();assert torch.isfinite(delta.grad).all()and torch.isfinite(db.grad).all();calls+=1;return loss
        opt.step(closure)
        with torch.no_grad():
            lw=(w0+delta@pre).float().numpy().copy();lb=(b0+db).float().numpy().copy();metrics=evaluate(lw,lb);value=float(objective())
        if metrics['correct']>best or(metrics['correct']==best and metrics['nll']<nll-1e-4):best,nll,selected=metrics['correct'],metrics['nll'],update;bw,bb=lw.copy(),lb.copy()
        peak=max(peak,metrics['correct'])
        if interval is not None and interval[0]<=metrics['correct']<=interval[1]:
            best,nll,selected=metrics['correct'],metrics['nll'],update;bw,bb=lw.copy(),lb.copy();interval_met=True
        row=dict(update=update,gradient_evaluations=calls,objective=value,selection_replay_correct=metrics['correct'],selected_replay_correct=best,selected_update=selected,original139_correct=metrics['original139_correct'],other272_correct=metrics['other272_correct'],interval_met=interval_met,peak_observed_correct=peak,elapsed_s=time.monotonic()-started)
        history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(recipe=recipe,state='fitting',latest=row));print(json.dumps(dict(recipe=recipe,**row)),flush=True)
        if interval_met or time.monotonic()-started>=c['fit_seconds_per_recipe']:break
    return dict(best_weight=bw,best_bias=bb,last_weight=lw,last_bias=lb,selected_update=selected,selected_replay_correct=best,updates=update,gradient_evaluations=calls,TRAIN_gradient_balance=balance,interval_met=interval_met,peak_observed_correct=peak,selection_policy='first_actual_training_checkpoint_in_user_interval'if interval is not None else'maximum_correct_then_NLL')


def run(a):
    np,torch,m,loop=base.numerical_modules();torch.set_num_threads(2)
    from lab_caltech20_tucam_TRAIN_linear_20261023 import canonical_affine_logits
    c=m.read(a.profile);validate(c);continued=c['schema']=='T18_KL_TRAIN_gradient_balance_v1';assert a.out.name==('fullG2_KLbalance_20261024'if continued else'fullG2_transfer_comparisons_20261024')and not m.LOCK.exists()
    assert not continued or not a.resume_save_failure
    if a.resume_save_failure:
        failure=m.read(a.out/'failure.json');assert 'FileNotFoundError'in failure['error']and 'best_TRAIN_original_predictions.npz'in failure['traceback']
        assert m.read(a.out.with_suffix('.return.json'))['returncode']==1 and not(a.out/'report.json').exists()and not(a.out/'recovery.json').exists()
    else:assert not a.out.exists()
    root=base.HERE/'runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2_actual300_20261024';parent=root/c['parent_run'];teacher=root/c['teacher_run'];protected=root/'accepted_full_physical_continued_20261024.json'
    for folder,prefix in((parent,'parent'),(teacher,'teacher')):
        assert m.read(folder.with_suffix('.return.json'))['returncode']==0 and m.sha(folder/'best.pt')==c[prefix+'_PT_SHA256']and m.sha(folder/'report.json')==c[prefix+'_report_SHA256']
        au=m.read(folder/'independent_audit.json');assert au['passed']and au['report_SHA256']==m.sha(folder/'report.json')
    assert m.read(teacher/'report.json')['TEST_gradient']and not m.read(parent/'report.json')['TEST_gradient']and m.sha(protected)==c['protected_manifest_SHA256']
    physical=m.read(base.HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json');base.validate_config(physical);reference=base.ROOT/physical['reference'];assert m.sha(reference)==physical['reference_SHA256'];model,_=m.model_for(parent/'best.pt',c['parent_PT_SHA256'],4,reference);tmodel,_=m.model_for(teacher/'best.pt',c['teacher_PT_SHA256'],4,reference)
    initial_student=parent
    if continued:
        initial_student=root/c['student_initial_run'];assert m.sha(initial_student/'best.pt')==c['student_initial_PT_SHA256']and m.sha(initial_student/'native_report.json')==c['student_initial_report_SHA256']
        prior=m.read(initial_student/'native_report.json');assert prior['scope']['teacher_seen_TEST']and not prior['scope']['student_direct_TEST_gradient']and prior['native_results']['best']['former_TEST_development_replay']['correct']==331
        whole=m.read(initial_student.parent/'report.json');au=m.read(initial_student.parent/'independent_audit.json');assert whole['state']=='complete'and au['passed']and au['report_SHA256']==m.sha(initial_student.parent/'report.json')and m.read(initial_student.parent.with_suffix('.resume.return.json'))['returncode']==0
        baseline=m.state_cpu(model);model,_=m.model_for(initial_student/'best.pt',c['student_initial_PT_SHA256'],4,reference);loop.validate_state(m,model,baseline,{'linear.weight','linear.bias'})
    original=m.state_cpu(model);keys={'linear.weight','linear.bias'};loop.validate_state(m,tmodel,original,keys)
    data,dm=m.data_for(base.ROOT/physical['sequence']);assert dm['sha256']==physical['data_SHA256'];fields={};paths={};policy=m.read(root/'readout_G2PASS/protocol.json')['policy'];sp=m.read(parent/'protocol.json')
    for stage in base.STAGES:
        cap=base.ROOT/policy['captures'][stage]if stage in policy['captures']else root/'readout_G2PASS/s_3/capture';r=m.read(cap/'report.json');au=m.read(cap/'independent_audit.json');assert au['passed']and au['matched_samples']==2011 and au['report_SHA256']==m.sha(cap/'report.json')and r['SDK_released']and m.sha(cap/'phase.bmp')==r['phase_SHA256'];fields[stage]={};paths[stage]=cap
        for role,n in(('train',1600),('test',411)):
            p=cap/role/'ccd.npy';assert m.sha(p)==sp['prefix_SHA256'][stage+'/'+role]==r['arrays'][role]['SHA256'];v=np.load(p,mmap_mode='r');assert v.shape==(n,478,478)and v.dtype==np.uint8;fields[stage][role]=v
    lineage=base.identities(m,paths);cache=teacher/'TEST_supervised_native_features.npz';assert m.sha(cache)==c['frozen_features_SHA256']
    with np.load(cache)as z:
        train=z['TRAIN_features'].copy();former=z['former_TEST_features'].copy()
        for role,key in(('train','TRAIN'),('test','former_TEST')):assert np.array_equal(z[key+'_ids'],data[role][2])and np.array_equal(z[key+'_labels'],data[role][1])
    pw=original['linear.weight'].numpy();pb=original['linear.bias'].numpy();tw=tmodel.linear.weight.detach().numpy().copy();tb=tmodel.linear.bias.detach().numpy().copy();del tmodel
    fx=torch.from_numpy(former);fy=data['test'][1]
    with torch.no_grad():
        p=canonical_affine_logits(fx,torch.from_numpy(pw),torch.from_numpy(pb)).softmax(1).numpy();q=canonical_affine_logits(fx,torch.from_numpy(tw),torch.from_numpy(tb)).softmax(1).numpy()
    assert int((p.argmax(1)==fy).sum())==(331 if continued else 272)and int((q.argmax(1)==fy).sum())==410
    recorded=initial_student/('best_replay.npz'if continued else'best_test_predictions.npz')
    assert np.max(abs(p-np.load(recorded)['probabilities']))<=3e-6 and np.max(abs(q-np.load(teacher/'best_former_TEST_adaptation_predictions.npz')['probabilities']))<=3e-6
    wrong=np.load(parent/'best_test_predictions.npz')['probabilities'].argmax(1)!=fy;assert int(wrong.sum())==139
    def evaluate(w,b):
        with torch.no_grad():
            z=canonical_affine_logits(fx,torch.from_numpy(w),torch.from_numpy(b));pred=z.argmax(1).numpy();return dict(correct=int((pred==fy).sum()),nll=float(torch.nn.functional.cross_entropy(z,torch.from_numpy(fy))),original139_correct=int((pred[wrong]==fy[wrong]).sum()),other272_correct=int((pred[~wrong]==fy[~wrong]).sum()))
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=c,parent_PT_SHA256=m.sha(initial_student/'best.pt'),teacher_PT_SHA256=c['teacher_PT_SHA256'],data_SHA256=dm['sha256'],prefix_SHA256=lineage,feature_cache_SHA256=m.sha(cache),original_split_files_unchanged=True,all_non_readout_frozen=True,allowed_keys=sorted(keys),trainable_parameters=5140,new_inference_layers=0,SDK_used=False,GPU_used=False,new_capture=False,official_TEST_accuracy=None,valid_for_formal_test_table=False,independent_test_samples=0,original_development_best_replaced=False,continues_ordinary_KL=continued)
    if a.resume_save_failure:
        prior=m.read(a.out/'protocol.json');assert prior['config']==c and prior['prefix_SHA256']==lineage
        m.save(a.out/'recovery.json',dict(reason='Known Windows MAX_PATH prediction filename failure after saved best; no numerical failure, no refit of errors139',source_git=m.git_head(),source_SHA256=m.sha(__file__),old_failure_SHA256=m.sha(a.out/'failure.json'),old_return_SHA256=m.sha(a.out.with_suffix('.return.json')),old_best_PT_SHA256=m.sha(a.out/'errors139_augmented_only/best.pt'),missing_original_last_disclosed=True,all_old_artifacts_preserved=True))
    else:
        a.out.mkdir();m.save(a.out/'protocol.json',protocol);m.save(a.out/'baseline.json',dict(original_development_correct=272,initial_student_correct=331 if continued else 272,teacher_supervised_adaptation_correct=410,original_error_sources=139,non_error_sources=272,feature_distillation_identically_zero=True,reason='Every non-readout state is bitwise identical and frozen; only classifiers differ.'))
    experiments={}
    for recipe in c['recipes']:
        out=a.out/recipe;recover=a.resume_save_failure and recipe=='errors139_augmented_only'
        if recover:assert out.exists()and(out/'best.pt').exists()and not(out/'last.pt').exists()and not(out/'native_report.json').exists()
        else:out.mkdir()
        derivatives=recipe=='errors139_augmented_only';inherited=recipe in('KL_teacher','DKD_TRAIN_neighborhood')or continued;neighborhood=recipe.startswith('DKD')
        if derivatives:packet=error_augmented_pool(train,data['train'][1],data['train'][2],former[wrong],fy[wrong],data['test'][2][wrong],c)
        elif neighborhood:packet=neighborhood_pool(train,data['train'][1],data['train'][2],c)
        else:packet=dict(features=train,labels=data['train'][1],source_ids=data['train'][2],source_roles=np.array(['original_TRAIN']*len(train)),augmentation=np.array(['unmodified']*len(train)))
        if recover:
            with np.load(out/'gradient_pool.npz')as old:assert set(old.files)==set(packet)and all(np.array_equal(old[k],v)for k,v in packet.items())
        else:np.savez_compressed(out/'gradient_pool.npz',**packet)
        scope=dict(recipe=recipe,direct_original_TEST_rows_in_gradient=0,TEST_augmented_source_samples=139 if derivatives else 0,TEST_augmented_rows=278 if derivatives else 0,teacher_seen_TEST=inherited,TEST_information_via_teacher=inherited,student_TEST_information=derivatives or inherited,student_direct_TEST_gradient=derivatives,student_direct_original_TEST_gradient=False,independent_test_samples=0,official_TEST_accuracy=None,valid_for_formal_test_table=False,original_TEST_used_for_checkpoint_selection=True,original139_source_ids=data['test'][2][wrong].tolist()if derivatives else[],gradient_pool_rows=len(packet['labels']),trainable_parameters=5140)
        if not recover:m.save(out/'protocol.json',dict(**protocol,recipe_scope=scope))
        m.save(a.out/'progress.json',dict(state='recovering_saved_best_no_refit'if recover else'fitting_recipe',recipe=recipe,scope=scope))
        if recover:
            native,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);loop.validate_state(m,native,original,keys);hist=m.read(out/'history.json');last=hist[-1];check=evaluate(native.linear.weight.detach().numpy(),native.linear.bias.detach().numpy());assert check['correct']==last['selected_replay_correct']
            fitted=dict(selected_update=last['selected_update'],selected_replay_correct=last['selected_replay_correct'],updates=last['update'],gradient_evaluations=last['gradient_evaluations']);tags=('best',)
        else:
            fitted=optimize_pool(packet,pw,pb,tw if inherited else pw,tb if inherited else pb,recipe,c,evaluate,out,m);tags=('best','last')
            # Persist BOTH actual fitted endpoints before any expensive replay.
            for tag in tags:
                model.load_state_dict(original)
                with torch.no_grad():model.linear.weight.copy_(torch.from_numpy(fitted[tag+'_weight']));model.linear.bias.copy_(torch.from_numpy(fitted[tag+'_bias']))
                loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/(tag+'.pt'),dict(**protocol,recipe_scope=scope))
        results={}
        for tag in tags:
            file=out/(tag+'.pt');native,_=m.model_for(file,m.sha(file),4,reference);loop.validate_state(m,native,original,keys);results[tag]=dict(PT_SHA256=m.sha(file))
            for role,display in(('train','TRAIN_original'),('test','former_TEST_development_replay')):
                m.save(a.out/'progress.json',dict(state='strict_original_CPU_replay',recipe=recipe,checkpoint=tag,role=display,student_TEST_information=scope['student_TEST_information']))
                metric,prob=base.evaluate(native,data,fields,role);metric.update(independent_test=False,TEST_information_in_student_training=scope['student_TEST_information']);results[tag][display]=metric;np.savez_compressed(out/(tag+('_train.npz'if role=='train'else'_replay.npz')),ids=data[role][2],labels=data[role][1],probabilities=prob)
                if role=='test':
                    pred=prob.argmax(1);results[tag]['original139_correct']=int((pred[wrong]==fy[wrong]).sum());results[tag]['other272_correct']=int((pred[~wrong]==fy[~wrong]).sum())
        assert results['best']['former_TEST_development_replay']['correct']==fitted['selected_replay_correct']
        entry=dict(scope=scope,selected_update=fitted['selected_update'],selected_E0=fitted['selected_update']==0,optimizer_updates=fitted['updates'],gradient_evaluations=fitted['gradient_evaluations'],strict_original_CPU=True,all_non_readout_bitwise_frozen=True,native_results=results,last_available=not recover,recovery_no_refit=recover,missing_last_reason='Original fit endpoint was in memory when long-path output failed; not reconstructed or substituted'if recover else None,TRAIN_gradient_balance=fitted.get('TRAIN_gradient_balance'))
        m.save(out/'native_report.json',entry);experiments[recipe]=entry;m.save(a.out/'completed_recipes.json',experiments);print(json.dumps(dict(recipe_complete=recipe,best_correct=results['best']['former_TEST_development_replay']['correct'],original139_correct=results['best']['original139_correct'],other272_correct=results['best']['other272_correct'],scope=scope)),flush=True)
        if continued:
            with torch.no_grad():model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model'])
            pw=model.linear.weight.detach().numpy().copy();pb=model.linear.bias.detach().numpy().copy()
            if results['best']['former_TEST_development_replay']['correct']>=c['teacher_adaptation_development_target']:break
    assert base.identities(m,paths)==lineage and m.sha(protected)==c['protected_manifest_SHA256']
    report=dict(state='complete',experiments=experiments,continues_ordinary_KL=continued,teacher_adaptation_development_target_met=any(v['native_results']['best']['former_TEST_development_replay']['correct']>=c['teacher_adaptation_development_target']for v in experiments.values())if continued else None,recovery_source_git=m.git_head()if a.resume_save_failure else None,official_TEST_accuracy=None,valid_for_formal_test_table=False,independent_test_samples=0,original_development_best_replaced=False,all_non_readout_bitwise_frozen=True,all_10_capture_array_SHA_unchanged=True,SDK_used=False,GPU_used=False,new_capture=False,strict_original_CPU=True,protocol_SHA256=m.sha(a.out/'protocol.json'),NOTE='Removing raw TEST inputs from student optimization does not erase TEST-derived augmentation or teacher-inherited information. Control has no TEST-trained teacher but remains a TEST-selected development metric.')
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),original_manifest_SHA256=m.sha(protected),prefix_SHA256=lineage,independent_test=False))


def selftest():
    import torch,numpy as np
    c=json.loads((base.HERE/'lab_caltech20_readout_transfer_profile_20261024.json').read_text());validate(c);torch.set_num_threads(2);torch.manual_seed(73)
    a=torch.randn(40,20,dtype=torch.float64,requires_grad=True);b=torch.randn(40,20,dtype=torch.float64,requires_grad=True);y=torch.arange(40)%20
    for kind in('KL','DKD'):
        loss=distillation_loss(a,b,y,kind,c);assert loss.shape==(40,)and torch.isfinite(loss).all();assert distillation_loss(b,b,y,kind,c).abs().max()<1e-10;g=torch.autograd.grad(loss.sum(),a,retain_graph=True)[0];assert torch.isfinite(g).all()and g.abs().sum()>0
    assert b.grad is None
    continued=json.loads((base.HERE/'lab_caltech20_KL_continuation_profile_20261024.json').read_text());validate(continued)
    gc=torch.autograd.grad(torch.nn.functional.cross_entropy(a,y),a)[0];gk=torch.autograd.grad(distillation_loss(a,b,y,'KL',c).mean(),a)[0]
    for multiplier in(2.,4.):
        balance=TRAIN_balance_weights(float(gc.norm()),float(gk.norm()),float((gc*gk).sum()/(gc.norm()*gk.norm())),multiplier,continued['ratio_bounds']);assert abs(balance['TRAIN_CE_weight']+balance['TRAIN_KL_weight']-1)<1e-12 and 0<balance['TRAIN_CE_weight']<1
    rng=np.random.default_rng(73);x=rng.normal(size=(40,6)).astype(np.float32);labels=np.arange(40)%20;ids=np.array(['TRAIN'+str(i)for i in range(40)]);p=neighborhood_pool(x,labels,ids,c);assert len(p['features'])==120 and set(p['source_roles'])=={'original_TRAIN'}
    cc={**c,'fixed_error_source_samples':3};z=x[:3]+.1;q=error_augmented_pool(x,labels,ids,z,labels[:3],np.array(['ERR'+str(i)for i in range(3)]),cc)
    assert len(q['features'])==46 and np.array_equal(q['features'][:40],x)and not np.any(q['augmentation'][40:]=='unmodified')
    print(json.dumps(dict(passed=True,teacher_detached=True,no_raw_TEST_rows=True,error_derivatives_still_TEST_information=True,TRAIN_gradient_balance_finite=True,all_non_readout_frozen=True,SDK_used=False,GPU_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--profile',type=Path,default=base.HERE/'lab_caltech20_readout_transfer_profile_20261024.json');p.add_argument('--selftest',action='store_true');p.add_argument('--resume-save-failure',action='store_true');a=p.parse_args()
    if a.selftest:selftest()
    else:
        assert a.out is not None
        try:run(a)
        except BaseException as e:
            if a.out.exists():
                _,_,m,_=base.numerical_modules();m.save(a.out/('recovery_failure.json'if a.resume_save_failure else'failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
            raise
