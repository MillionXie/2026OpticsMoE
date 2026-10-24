"""Explicitly contaminated, human-authorized TEST-supervised head adaptation.

Never writes the original TRAIN/TEST data or the original development manifest.
Same-set replay is a supervised adaptation diagnostic, NOT TEST generalization.
All augmentation is after the frozen optical/electronic feature extractor.
"""
import argparse,json,time,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base


def validate(c):
    assert c['schema']=='T18_TEST_supervised_head_adaptation_v1'
    assert c['user_authorized_TEST_supervision'] is True
    assert c['independent_test_samples']==0 and not c['valid_for_formal_test_table']
    assert not c['replace_original_development_best'] and c['original_TRAIN']==1600 and c['former_TEST']==411 and c['VALID']==0
    assert (c['TRAIN_weight'],c['former_TEST_origin_weight'])==(.35,.65)
    assert c['former_TEST_error_weight']==3. and c['sameclass_TRAIN_interpolation']==.05 and c['feature_noise_std_fraction']==.005
    assert c['label_smoothing']==.01 and c['TRAIN_parent_KL']==.1 and c['preconditioned_delta_L2']==.0005
    assert c['covariance_floor_fraction']==.01 and c['rounds_max']==2 and c['outer_steps_per_round']==60 and c['fit_seconds_max']==120
    assert c['trainable_parameters']==5140 and not c['SDK_used'] and not c['GPU_used'] and not c['new_capture'] and c['new_inference_layers']==0


def supervised_pool(train,ty,train_ids,former,fy,former_ids,prediction,c,seed):
    import numpy as np
    rng=np.random.default_rng(seed);bad=np.flatnonzero(prediction!=fy)
    assert len(train)==len(ty)==len(train_ids) and len(former)==len(fy)==len(former_ids)
    original=np.asarray(former[bad],np.float32);neighbors=[]
    tn=train/np.linalg.norm(train,axis=1,keepdims=True).clip(1e-12)
    fn=former/np.linalg.norm(former,axis=1,keepdims=True).clip(1e-12)
    for i in bad:
        pool=np.flatnonzero(ty==fy[i]);assert len(pool)
        neighbors.append(int(pool[np.argmax(tn[pool]@fn[i])]))
    neighbors=np.asarray(neighbors,np.int64)
    mix=(1-c['sameclass_TRAIN_interpolation'])*original+c['sameclass_TRAIN_interpolation']*train[neighbors]
    noise=original+rng.normal(size=original.shape).astype(np.float32)*train.std(0)*c['feature_noise_std_fraction']
    x=np.concatenate([train,former,mix,noise]).astype(np.float32);y=np.concatenate([ty,fy,fy[bad],fy[bad]]).astype(np.int64)
    ids=np.concatenate([train_ids,former_ids,former_ids[bad],former_ids[bad]])
    roles=np.array(['original_TRAIN']*len(train)+['former_TEST']*(len(former)+2*len(bad)))
    kinds=np.array(['unmodified']*(len(train)+len(former))+['sameclass_feature_mix']*len(bad)+['feature_noise']*len(bad))
    weights=np.r_[np.full(len(train),c['TRAIN_weight']/len(train)),np.zeros(len(former)+2*len(bad))]
    adapted=np.r_[np.where(prediction==fy,1.,c['former_TEST_error_weight']),np.ones(2*len(bad))]
    weights[len(train):]=c['former_TEST_origin_weight']*adapted/adapted.sum()
    assert np.isclose(weights.sum(),1) and np.array_equal(x[:len(train)],train)
    assert np.array_equal(y[len(train):len(train)+len(former)],fy) and np.array_equal(ty[neighbors],fy[bad])
    return dict(features=x,labels=y,source_ids=ids,source_roles=roles,augmentation=kinds,loss_weights=weights),dict(former_TEST_wrong_indices=bad.tolist(),sameclass_TRAIN_neighbor_indices=neighbors.tolist(),original_TRAIN=len(train),former_TEST_clean=len(former),former_TEST_augmented=2*len(bad),total=len(x),TEST_supervised=True,independent_test_samples=0,synthetic_features_not_physical_CCD=True)


def objective(logits,labels,weights,parent_train_prob,delta,db,c):
    from torch.nn import functional as F
    ce=(F.cross_entropy(logits,labels,reduction='none',label_smoothing=c['label_smoothing'])*weights).sum()
    anchor=F.kl_div(F.log_softmax(logits[:len(parent_train_prob)],1),parent_train_prob,reduction='batchmean')
    penalty=c['preconditioned_delta_L2']*(delta.square().sum()+db.square().sum())
    return ce+c['TRAIN_parent_KL']*anchor+penalty


def fit_head(train,ty,train_ids,former,fy,former_ids,w,b,parent_train_prob,c,out,m):
    import numpy as np,torch
    from torch.nn import functional as F
    from lab_caltech20_tucam_TRAIN_linear_20261023 import canonical_affine_logits
    tx=torch.tensor(train,dtype=torch.float32);fx=torch.tensor(former,dtype=torch.float32);ft=torch.tensor(fy,dtype=torch.long)
    w=np.asarray(w,np.float32).copy();b=np.asarray(b,np.float32).copy()
    with torch.no_grad():
        baseline=canonical_affine_logits(fx,torch.from_numpy(w),torch.from_numpy(b));best=int((baseline.argmax(1)==ft).sum());bnll=float(F.cross_entropy(baseline,ft))
    bw,bb=w.copy(),b.copy();selected=updates=calls=0;history=[];plans=[];started=time.monotonic();actual_rounds=0
    parent=torch.tensor(parent_train_prob,dtype=torch.float64)
    for round_id in range(1,c['rounds_max']+1):
        actual_rounds=round_id
        with torch.no_grad():prediction=canonical_affine_logits(fx,torch.from_numpy(bw),torch.from_numpy(bb)).argmax(1).numpy()
        packet,plan=supervised_pool(train,ty,train_ids,former,fy,former_ids,prediction,c,c['seed']+round_id)
        np.savez_compressed(out/('TEST_supervised_pool_round'+str(round_id)+'.npz'),**packet);plans.append(plan)
        x=torch.tensor(packet['features'],dtype=torch.float64);y=torch.tensor(packet['labels'],dtype=torch.long);weights=torch.tensor(packet['loss_weights'],dtype=torch.float64)
        w0=torch.tensor(bw,dtype=torch.float64);b0=torch.tensor(bb,dtype=torch.float64)
        with torch.no_grad():
            centered=x-x.mean(0);cov=centered.T@centered/len(x);floor=c['covariance_floor_fraction']*torch.trace(cov)/256;e,v=torch.linalg.eigh(cov);precondition=(v*(e.clamp_min(0)+floor).rsqrt())@v.T
        delta=torch.nn.Parameter(torch.zeros_like(w0));db=torch.nn.Parameter(torch.zeros_like(b0))
        opt=torch.optim.LBFGS([delta,db],lr=1,max_iter=5,max_eval=8,history_size=20,line_search_fn='strong_wolfe')
        before=best;prior_nll=bnll
        for outer in range(1,c['outer_steps_per_round']+1):
            def closure():
                nonlocal calls
                opt.zero_grad(set_to_none=True);weight=w0+delta@precondition;bias=b0+db
                loss=objective(F.linear(x,weight,bias),y,weights,parent,delta,db,c)
                assert torch.isfinite(loss);loss.backward();assert torch.isfinite(delta.grad).all() and torch.isfinite(db.grad).all();calls+=1;return loss
            opt.step(closure);updates+=1
            with torch.no_grad():
                lw=(w0+delta@precondition).float().numpy().copy();lb=(b0+db).float().numpy().copy()
                q=canonical_affine_logits(fx,torch.from_numpy(lw),torch.from_numpy(lb));correct=int((q.argmax(1)==ft).sum());nll=float(F.cross_entropy(q,ft))
                if correct>best or(correct==best and nll<bnll-1e-4):best,bnll,selected=correct,nll,updates;bw,bb=lw.copy(),lb.copy()
                tr=canonical_affine_logits(tx,torch.from_numpy(lw),torch.from_numpy(lb));train_correct=int((tr.argmax(1).numpy()==ty).sum())
            row=dict(round=round_id,outer_step=outer,optimizer_updates=updates,gradient_evaluations=calls,supervised_pool=len(x),TRAIN_original_correct=train_correct,former_TEST_adaptation_correct=correct,selected_adaptation_correct=best,selected_update=selected,former_TEST_adaptation_NLL=nll,TEST_supervised=True,independent_test_samples=0,elapsed_s=time.monotonic()-started)
            history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='TEST_supervised_head_fit',latest=row));print(json.dumps(row),flush=True)
            if best==len(former) or time.monotonic()-started>=c['fit_seconds_max']:break
        if best==len(former) or time.monotonic()-started>=c['fit_seconds_max'] or(best==before and bnll>=prior_nll-1e-4):break
    m.save(out/'TEST_supervised_pool_plan.json',dict(rounds=plans,TEST_supervised=True,independent_test_samples=0,original_split_files_unchanged=True))
    return dict(best_weight=bw,best_bias=bb,last_weight=lw,last_bias=lb,selected=selected,selected_adaptation_correct=best,optimizer_updates=updates,gradient_evaluations=calls,rounds=actual_rounds)


def run(a):
    np,torch,m,loop=base.numerical_modules();torch.set_num_threads(2)
    from lab_caltech20_tucam_TRAIN_linear_20261023 import canonical_affine_logits
    c=m.read(a.profile);validate(c);assert a.out.name=='fullG2_TESTsupervised_20261024' and not a.out.exists() and not m.LOCK.exists()
    root=base.HERE/'runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2_actual300_20261024';source=root/c['parent_run'];pt=source/'best.pt'
    assert m.read(source.with_suffix('.return.json'))['returncode']==0 and m.sha(pt)==c['parent_PT_SHA256'] and m.sha(source/'report.json')==c['parent_report_SHA256']
    sr=m.read(source/'report.json');audit=m.read(source/'independent_audit.json');sp=m.read(source/'protocol.json')
    assert sr['selected_correct']==272 and sr['strict_original_CPU'] and sr['full_physical'] and not sr['TEST_gradient'] and audit['passed'] and audit['report_SHA256']==m.sha(source/'report.json')
    protected=root/'accepted_full_physical_continued_20261024.json';assert m.sha(protected)==c['protected_accepted_manifest_SHA256']
    physical=m.read(base.HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json');base.validate_config(physical)
    reference=base.ROOT/physical['reference'];assert m.sha(reference)==physical['reference_SHA256'];model,_=m.model_for(pt,m.sha(pt),4,reference)
    original=m.state_cpu(model);keys={'linear.weight','linear.bias'};assert sum(original[k].numel()for k in keys)==5140
    data,dm=m.data_for(base.ROOT/physical['sequence']);assert dm['sha256']==physical['data_SHA256'];fields={};paths={};policy=m.read(root/'readout_G2PASS/protocol.json')['policy']
    for stage in base.STAGES:
        cap=base.ROOT/policy['captures'][stage]if stage in policy['captures']else root/'readout_G2PASS/s_3/capture';r=m.read(cap/'report.json');au=m.read(cap/'independent_audit.json')
        assert au['passed'] and au['matched_samples']==2011 and au['report_SHA256']==m.sha(cap/'report.json') and r['SDK_released'] and r['data_SHA256']==dm['sha256'] and m.sha(cap/'phase.bmp')==r['phase_SHA256']
        fields[stage]={};paths[stage]=cap
        for role,n in(('train',1600),('test',411)):
            path=cap/role/'ccd.npy';assert m.sha(path)==sp['prefix_SHA256'][stage+'/'+role]==r['arrays'][role]['SHA256'];v=np.load(path,mmap_mode='r');assert v.shape==(n,478,478) and v.dtype==np.uint8;fields[stage][role]=v
    lineage=base.identities(m,paths);a.out.mkdir()
    protocol=dict(experiment='TEST_supervised_adaptation_NOT_independent_test',TEST_supervised=True,TEST_gradient=True,independent_test_samples=0,valid_for_formal_test_table=False,original_target_evaluable=False,original_development_best_replaced=False,source_git=m.git_head(),source_SHA256=m.sha(__file__),config=c,profile_SHA256=m.sha(a.profile),parent_PT=str(pt),parent_PT_SHA256=m.sha(pt),parent_report_SHA256=m.sha(source/'report.json'),data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),trainable_parameters=5140,new_inference_layers=0,SDK_used=False,GPU_used=False,new_capture=False,original_split_files_unchanged=True,gradient_origin_original_TRAIN=1600,gradient_origin_former_TEST=411)
    m.save(a.out/'protocol.json',protocol);model.cpu().eval().requires_grad_(False);cache={};prob={};maxerr=0.
    with torch.inference_mode():
        for role,n in(('train',1600),('test',411)):
            values=[];probabilities=[]
            for begin in range(0,n,8):
                images,count,end=m.padded(data,role,begin);model.measured={}
                for k,v in fields.items():
                    arr=np.asarray(v[role][begin:end]).copy()
                    if count<8:arr=np.concatenate([arr,np.zeros((8-count,478,478),arr.dtype)])
                    model.measured[k]=torch.from_numpy(arr)
                p=model(m.encode_images(images))[0][:count].numpy().copy();values.append(model.readout['features'][:count].numpy().copy());probabilities.append(p)
                if begin%160==0:m.save(a.out/'progress.json',dict(state='native_frozen_feature_cache',role_original=role,consumed=begin+count,TEST_supervised=True,independent_test_samples=0))
            cache[role]=np.concatenate(values);prob[role]=np.concatenate(probabilities)
            check=canonical_affine_logits(torch.from_numpy(cache[role]),model.linear.weight,model.linear.bias).softmax(1).numpy();err=float(np.max(abs(check-prob[role])));maxerr=max(maxerr,err)
            old=np.load(source/('best_'+role+'_predictions.npz'));assert np.array_equal(old['ids'],data[role][2]) and np.array_equal(old['labels'],data[role][1]);maxerr=max(maxerr,float(np.max(abs(old['probabilities']-prob[role]))))
    assert maxerr<=3e-6 and int((prob['test'].argmax(1)==data['test'][1]).sum())==272
    np.savez_compressed(a.out/'TEST_supervised_native_features.npz',TRAIN_features=cache['train'],TRAIN_ids=data['train'][2],TRAIN_labels=data['train'][1],former_TEST_features=cache['test'],former_TEST_ids=data['test'][2],former_TEST_labels=data['test'][1])
    m.save(a.out/'baseline.json',dict(original_development_correct=272,former_TEST_adaptation_correct_before=272,samples=411,TEST_supervised_experiment=True,independent_test_samples=0,native_cache_probability_max_error=maxerr))
    fitted=fit_head(cache['train'],data['train'][1],data['train'][2],cache['test'],data['test'][1],data['test'][2],model.linear.weight.detach().numpy(),model.linear.bias.detach().numpy(),prob['train'],c,a.out,m)
    results={}
    for label in('best','last'):
        with torch.no_grad():model.linear.weight.copy_(torch.from_numpy(fitted[label+'_weight']));model.linear.bias.copy_(torch.from_numpy(fitted[label+'_bias']))
        loop.validate_state(m,model,original,keys);file=a.out/(label+'.pt');loop.saved_model(m,model,file,protocol);native,_=m.model_for(file,m.sha(file),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(file))
        for role,display in(('train','TRAIN_original'),('test','former_TEST_adaptation')):
            m.save(a.out/'progress.json',dict(state='strict_original_CPU_adaptation_replay',checkpoint=label,role=display,TEST_supervised=True,independent_test_samples=0))
            metric,p=base.evaluate(native,data,fields,role);metric.update(TEST_supervised=True,independent_test=False);results[label][display]=metric
            np.savez_compressed(a.out/(label+'_'+display+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
    assert results['best']['former_TEST_adaptation']['correct']==fitted['selected_adaptation_correct'] and base.identities(m,paths)==lineage and m.sha(protected)==c['protected_accepted_manifest_SHA256']
    report=dict(state='complete',experiment='TEST_supervised_adaptation_NOT_independent_test',TEST_supervised=True,TEST_gradient=True,independent_test_samples=0,valid_for_formal_test_table=False,original_target_evaluable=False,official_TEST_accuracy=None,original_development_best_replaced=False,baseline_correct=272,selected_adaptation_correct=fitted['selected_adaptation_correct'],selected_E0=fitted['selected']==0,selected_update=fitted['selected'],optimizer_updates=fitted['optimizer_updates'],full_supervised_gradient_evaluations=fitted['gradient_evaluations'],rounds=fitted['rounds'],results=results,strict_original_CPU=True,full_physical_capture_prefix=True,all_non_readout_bitwise_frozen=True,original_TRAIN=1600,former_TEST=411,VALID=0,native_cache_probability_max_error=maxerr,SDK_used=False,GPU_used=False,new_capture=False,new_inference_layers=0,protocol_SHA256=m.sha(a.out/'protocol.json'))
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),all_non_readout_bitwise_frozen=True,prefix_SHA256=lineage,original_accepted_manifest_SHA256=m.sha(protected),TEST_supervised=True,independent_test_samples=0,NOTE='Independent code/state audit does NOT mean independent TEST evaluation.'))


def selftest():
    import numpy as np,torch
    c=json.loads((base.HERE/'lab_caltech20_TEST_supervised_profile_20261024.json').read_text());validate(c);torch.set_num_threads(2)
    rng=np.random.default_rng(73);x=rng.normal(size=(40,6)).astype(np.float32);ty=np.arange(40)%20;z=x[:20].copy();fy=np.arange(20);pred=fy.copy();pred[:3]=(pred[:3]+1)%20
    p,plan=supervised_pool(x,ty,np.array(['TRAIN'+str(i)for i in range(40)]),z,fy,np.array(['TEST'+str(i)for i in range(20)]),pred,c,73)
    assert plan['former_TEST_augmented']==6 and len(p['features'])==66 and set(p['source_roles'])=={'original_TRAIN','former_TEST'} and np.array_equal(p['features'][:40],x)
    assert np.isclose(p['loss_weights'][:40].sum(),.35) and np.isclose(p['loss_weights'][40:].sum(),.65)
    logits=torch.randn(66,20,dtype=torch.float64,requires_grad=True);labels=torch.tensor(p['labels']);weights=torch.tensor(p['loss_weights']);parent=torch.randn(40,20,dtype=torch.float64).softmax(1);d=torch.zeros((20,6),dtype=torch.float64,requires_grad=True);db=torch.zeros(20,dtype=torch.float64,requires_grad=True)
    objective(logits,labels,weights,parent,d,db,c).backward();assert logits.grad[40:].abs().sum()>0 and torch.isfinite(logits.grad).all()
    print(json.dumps(dict(passed=True,TEST_supervised=True,former_TEST_gradient_deliberately_nonzero=True,independent_test_samples=0,split_origin_preserved=True,original_TRAIN_unchanged=True,new_inference_layers=0,SDK_used=False,GPU_used=False)))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,default=base.HERE/'lab_caltech20_TEST_supervised_profile_20261024.json');q.add_argument('--out',type=Path);q.add_argument('--selftest',action='store_true');a=q.parse_args()
    if a.selftest:selftest()
    else:
        assert a.out is not None
        try:run(a)
        except BaseException as e:
            if a.out.exists():
                _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True,TEST_supervised=True,independent_test_samples=0))
            raise
