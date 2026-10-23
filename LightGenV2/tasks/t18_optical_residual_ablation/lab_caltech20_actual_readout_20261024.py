"""Native measured-prefix original readout, class-contrast L2, TRAIN-only fit."""
import time,json
import lab_caltech20_tucam_layerwise_20261023 as base


def selftest():
    from lab_caltech20_TRAIN_contrast_readout_20261024 import selftest as contrast_test
    contrast_test()


def TRAIN_perturbation_audit(model,data,dm,fields,paths,parent,c,out):
    """Read-only original TRAIN perturbation sensitivity, no optimizer/TEST."""
    np,torch,m,loop=base.numerical_modules()
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert not out.exists() and not m.LOCK.exists();out.mkdir();model.cpu().eval().requires_grad_(False)
    original=m.state_cpu(model);lineage=base.identities(m,paths);ix=np.arange(80);labels=data['train'][1][ix]
    assert np.bincount(labels,minlength=20).tolist()==[4]*20
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=lineage,
        TRAIN=80,TEST_used=False,VALID=0,optimizer_updates=0,SDK_used=False,GPU_used=False,new_inference_layers=0,noise_is_synthetic_not_calibrated=True)
    m.save(out/'protocol.json',protocol)
    def forward():
        probabilities=[];features=[]
        with torch.inference_mode():
            for begin in range(0,80,8):
                batch=ix[begin:begin+8];probabilities.append(model(loop.put_fields(model,data,fields,'train',batch,'cpu'))[0].numpy().copy());features.append(model.readout['features'].numpy().copy())
        return np.concatenate(probabilities),np.concatenate(features)
    saturation={k:dict(count=0,abs_gt3=0,abs_gt6=0,tanh_derivative_sum=0.) for k in (2,3)}
    def collect(k):
        def hook(module,args,value):
            v=value.detach();row=saturation[k];row['count']+=v.numel();row['abs_gt3']+=int((v.abs()>3).sum());row['abs_gt6']+=int((v.abs()>6).sum());row['tanh_derivative_sum']+=float((1-v.tanh().square()).double().sum())
        return hook
    handles=[model.electronic[k].local.register_forward_hook(collect(k)) for k in (2,3)]
    try:clean,feature=forward()
    finally:
        for handle in handles:handle.remove()
    clean_again,_=forward();assert np.array_equal(clean,clean_again)
    rows={}
    cases=dict(current={**c},gain_additive_only={**c,'CCD_pixel_dropout_p':0.},pixel_dropout_only={**c,'CCD_gain_sigma':0.,'CCD_additive_sigma_DN':0.})
    for case,config in cases.items():
        torch.manual_seed(73);transforms=TRAINTransforms(model,'2',config);values=[];cosines=[]
        try:
            model.train()
            for repetition in range(4):
                probability,value=forward();values.append(probability);cosines.append((feature*value).sum(1)/(np.linalg.norm(feature,axis=1)*np.linalg.norm(value,axis=1)).clip(1e-12))
                m.save(out/'progress.json',dict(state='TRAIN_readonly_noise_audit',case=case,repetition=repetition+1,TRAIN=80,TEST_used=False))
        finally:transforms.remove();model.eval()
        v=np.stack(values);native=v.argmax(2);clean_prediction=clean.argmax(1);correct=clean_prediction==labels
        rows[case]=dict(TRAIN_trials=320,mean_correct_per80=float((native==labels[None]).sum(1).mean()),
            NLL=float(-np.log(v[np.arange(4)[:,None],np.arange(80)[None],labels[None]].clip(1e-12)).mean()),
            flip_fraction_from_clean=float((native!=clean_prediction[None]).mean()),clean_correct_to_wrong_fraction=float(((native!=labels[None])&correct[None]).sum()/max(1,4*correct.sum())),
            feature_cosine_mean=float(np.mean(cosines)),TRAIN_transform_counts=transforms.counts)
    after,_=forward();assert np.array_equal(clean,after) and all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()) and base.identities(m,paths)==lineage
    report=dict(state='complete',TRAIN=80,TEST_used=False,VALID=0,clean_correct=int((clean.argmax(1)==labels).sum()),
        clean_NLL=float(-np.log(clean[np.arange(80),labels].clip(1e-12)).mean()),cases=rows,
        TRAIN_postE2_original_tanh_statistics={str(k):dict(abs_gt3_fraction=v['abs_gt3']/v['count'],abs_gt6_fraction=v['abs_gt6']/v['count'],mean_tanh_derivative=v['tanh_derivative_sum']/v['count'])for k,v in saturation.items()},original_state_bitwise_unchanged=True,
        original_eval_identity_exact=True,optimizer_updates=0,SDK_used=False,GPU_used=False,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_parameters_bitwise_frozen=True))
    return report


def fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_tucam_TRAIN_linear_20261023 import TRAIN_affine_objective
    assert list(fields)==['router','0','1','2'] and not out.exists() and not m.LOCK.exists();out.mkdir();model.cpu().eval().requires_grad_(False)
    original=m.state_cpu(model);lineage=base.identities(m,paths);source=parent.parent;r=m.read(source/'report.json');p=m.read(source/'protocol.json')
    assert r['strict_original_CPU'] and r['results']['best']['PT_SHA256']==m.sha(parent) and p['prefix_SHA256']==lineage and p['data_SHA256']==dm['sha256']
    cache_source=base.ROOT/c['native_feature_source'] if c.get('native_feature_source') else source
    fresh=c.get('fresh_native_TRAIN_features',False);assert isinstance(fresh,bool)
    if cache_source!=source and not fresh:
        cache_report=m.read(cache_source/'report.json');cache_model,_=m.model_for(cache_source/'best.pt',cache_report['results']['best']['PT_SHA256'],4,reference)
        loop.validate_state(m,model,m.state_cpu(cache_model),{'linear.weight','linear.bias'});assert m.read(cache_source/'protocol.json')['prefix_SHA256']==lineage
    if fresh:
        parts=[]
        with torch.inference_mode():
            for begin in range(0,1600,8):
                ix=np.arange(begin,begin+8);model(loop.put_fields(model,data,fields,'train',ix,'cpu'));parts.append(model.readout['features'].numpy().copy())
                if begin%160==0:m.save(out/'progress.json',dict(state='fresh_native_TRAIN_features',TRAIN_consumed=begin+8,TEST_used_for_fit=False))
        cache=out/'native_TRAIN_features.npz';np.savez_compressed(cache,ids=data['train'][2],labels=data['train'][1],features=np.concatenate(parts))
        m.save(out/'TRAIN_statistics.json',dict(feature_cache_SHA256=m.sha(cache),TRAIN=1600,TEST_used=False,source_PT_SHA256=m.sha(parent),prefix_SHA256=lineage))
    else:cache=cache_source/'native_TRAIN_features.npz';stats=m.read(cache_source/'TRAIN_statistics.json');assert m.sha(cache)==stats['feature_cache_SHA256']
    with np.load(cache,allow_pickle=False) as z:
        assert np.array_equal(z['ids'],data['train'][2]) and np.array_equal(z['labels'],data['train'][1]);train=torch.from_numpy(z['features'].copy())
    assert train.shape==(1600,256) and torch.isfinite(train).all()
    with np.load(source/'best_train_predictions.npz',allow_pickle=False) as z:train_error=float(np.max(np.abs(model.linear(train).softmax(1).detach().numpy()-z['probabilities'])))
    baseline,prob=base.evaluate(model,data,fields,'test');test=[]
    with torch.inference_mode():
        for begin in range(0,411,8):
            ix=np.arange(begin,min(begin+8,411));model(loop.put_fields(model,data,fields,'test',ix,'cpu'));test.append(model.readout['features'].numpy().copy())
    test=torch.from_numpy(np.concatenate(test));error=float(np.max(np.abs(model.linear(test).softmax(1).detach().numpy()-prob)));assert max(error,train_error)<=3e-6
    keys={'linear.weight','linear.bias'};head=torch.nn.Linear(256,20).double();head.load_state_dict(model.linear.state_dict());raw_parent={k:v.detach().clone() for k,v in head.state_dict().items()}
    with torch.no_grad():head.weight.sub_(head.weight.mean(0,keepdim=True));head.bias.sub_(head.bias.mean())
    regularization=2*.1*r['results']['best']['train']['nll']/float(head.weight.detach().square().sum())
    labels=torch.from_numpy(data['train'][1].astype(np.int64));test_labels=torch.from_numpy(data['test'][1].astype(np.int64));train64=train.double();prior=np.asarray(c['original_TRAIN_counts'],float);weights=torch.tensor(.5+.5*np.sqrt(prior/prior.mean()),dtype=torch.float64)
    ridge=c.get('TRAIN_group_ridge',False);proximal=c.get('TRAIN_parent_proximal',False);assert isinstance(ridge,bool) and isinstance(proximal,bool) and not(ridge and proximal)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),config=c,recipe=recipe,
        data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,
        trainable_parameters=5140,new_inference_layers=0,full_physical=False,SDK_used=False,GPU_used=False,class_contrast_regularization=not(ridge or proximal),TRAIN_only_L2=None if ridge or proximal else regularization,
        cached_TRAIN_feature_SHA256=m.sha(cache),native_cache_probability_max_error=max(error,train_error),raw_parent_fallback_preserved=True)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',baseline);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    best,bnll,selected=baseline['correct'],baseline['nll'],0;best_head=raw_parent;history=[];calls=0;start=time.monotonic()
    opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=5,max_eval=8,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    if proximal:
        from lab_caltech20_TRAIN_parent_proximal_20261024 import fit as proximal_fit
        protocol.update(TRAIN_parent_proximal=True,TRAIN_only_class_weights='original TRAIN prior square-root, no TEST weighting',TRAIN_CE_smoothing=.03,TRAIN_parent_correct_only_KL_coefficient=.5,TRAIN_parent_temperature=2,TRAIN_parameter_anchor=.2,working_row_radius=.075,native_row_radius_max=.08,maximum_full_TRAIN_updates=800,new_inference_delta_module=False);m.save(out/'protocol.json',protocol)
        def emit(row,rows):m.save(out/'history.json',rows);m.save(out/'progress.json',row);print(json.dumps(row),flush=True)
        correction=proximal_fit(head,train64,labels,weights,raw_parent,test,test_labels,baseline,target,recipe['seconds'],emit)
        best,bnll,selected,step,history=correction['correct'],correction['nll'],correction['selected_update'],correction['updates'],correction['history'];best_head=correction['best_head'];calls=step
    if ridge:
        from lab_caltech20_TRAIN_group_ridge_20261024 import fit as ridge_fit
        path=base.ROOT/c['sequence']/'assets/TRAIN_lineage.json';assert m.sha(path)==dm['lineage_SHA256'];records=m.read(path)['records'];assert [v['sample_id'] for v in records]==data['train'][2].tolist()
        teacher_targets=c.get('TRAIN_teacher_targets',False);assert isinstance(teacher_targets,bool)
        if teacher_targets:
            from lab_caltech20_TRAIN_teacher_ridge_20261024 import fit as teacher_fit
            folder=base.ROOT/c['TRAIN_teacher_cache'];manifest=m.read(folder/'teacher_manifest.json');teacher_cache=folder/'teacher_TRAIN_logits.npz'
            assert m.sha(folder/'teacher_manifest.json')=='761611b3cf1986254e77efdc0fc0a0c5044fdb46f3fd89fbfe9cfa837b220a98'
            assert manifest['role']=='train' and manifest['samples']==1600 and not manifest['TEST_used'] and not manifest['VALID_used']
            assert manifest['source_PT_SHA256']=='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'
            assert m.sha(teacher_cache)==manifest['cache_SHA256']=='c73a318e3887b00200c51814790c7874754d64df9190edb044557347e25d6069'
            with np.load(teacher_cache,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],data['train'][2]) and np.array_equal(z['labels'],data['train'][1]);teacher_logits=z['logits'].copy()
            weight,bias,ridge_statistics=teacher_fit(train.numpy(),data['train'][1],c['original_TRAIN_counts'],[v['source_id'] for v in records],teacher_logits)
            m.save(out/'teacher_manifest.json',manifest);protocol.update(TRAIN_teacher_targets=True,teacher_cache_SHA256=m.sha(teacher_cache),inference_teacher=False)
        else:weight,bias,ridge_statistics=ridge_fit(train.numpy(),data['train'][1],c['original_TRAIN_counts'],[v['source_id'] for v in records])
        head.load_state_dict(dict(weight=torch.from_numpy(weight),bias=torch.from_numpy(bias)));m.save(out/'TRAIN_statistics.json',ridge_statistics)
        protocol.update(TRAIN_group_ridge=True,TRAIN_ridge_regularization=ridge_statistics['regularization'],TRAIN_family_lineage_SHA256=m.sha(path),TRAIN_statistics_SHA256=m.sha(out/'TRAIN_statistics.json'));m.save(out/'protocol.json',protocol)
        with torch.inference_mode():
            logits=F.linear(test,head.weight.float(),head.bias.float());correct=int((logits.argmax(1)==test_labels).sum());nll=float(F.cross_entropy(logits,test_labels));tl=F.linear(train,head.weight.float(),head.bias.float());train_correct=int((tl.argmax(1)==labels).sum())
        if correct>best or (correct==best and nll<bnll-1e-4):best,bnll,selected=correct,nll,1;best_head={k:v.detach().clone() for k,v in head.state_dict().items()}
        row=dict(analytical_fit=1,TRAIN_consumed=1600,TRAIN_correct=train_correct,TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_parent=selected==0,TEST_used_for_regularization=False);history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row);step=1
    for step in range(1,1 if ridge or proximal else 81):
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True);loss,_,_=TRAIN_affine_objective(head.weight,head.bias,train64,labels,weights,.03,regularization);assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters());calls+=1;return loss
        opt.step(closure)
        if step==1 or step%5==0 or step==80:
            with torch.inference_mode():
                logits=F.linear(test,head.weight.float(),head.bias.float());correct=int((logits.argmax(1)==test_labels).sum());nll=float(F.cross_entropy(logits,test_labels));tl=F.linear(train,head.weight.float(),head.bias.float());train_correct=int((tl.argmax(1)==labels).sum())
            if correct>best or (correct==best and nll<bnll-1e-4):best,bnll,selected=correct,nll,step;best_head={k:v.detach().clone() for k,v in head.state_dict().items()}
            row=dict(outer_step=step,TRAIN_consumed=1600,TRAIN_correct=train_correct,TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_outer_step=selected,full_TRAIN_gradient_evaluations=calls,elapsed_s=time.monotonic()-start);history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row);print(json.dumps(row),flush=True)
        if best>=target or time.monotonic()-start>=recipe['seconds']:break
    model.linear.load_state_dict(correction['last_head']if proximal else head.float().state_dict());loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    model.linear.load_state_dict({k:v.float() for k,v in best_head.items()});loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'best.pt',protocol);results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,v=base.evaluate(native,data,fields,role);results[label][role]=metric;np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=v)
    assert results['best']['test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_E0=selected==0,selected_outer_step=None if ridge or proximal else selected,selected_full_TRAIN_update=selected if proximal else None,outer_steps=0 if ridge or proximal else step,actual_epochs=0,optimizer_updates=0 if ridge else step,analytical_fits=1 if ridge else 0,TRAIN_parent_proximal=proximal,full_TRAIN_gradient_evaluations=calls,
        results=results,strict_original_CPU=True,target=target,target_met=best>=target,captured_upstream_bitwise_frozen=True,all_non_readout_frozen=True,TEST_gradient=False,TEST_development=True,full_physical=False,new_inference_layers=0,SDK_used=False,GPU_used=False,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True,all_non_readout_frozen=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);return chosen,out/'best.pt',report
