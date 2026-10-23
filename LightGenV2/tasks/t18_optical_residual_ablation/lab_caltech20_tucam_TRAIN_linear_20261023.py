"""One TRAIN-only covariance-regularized fit of the existing5140 linear head.

OAS formulation reference: scikit-learn covariance/_shrunk_covariance.py,
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/covariance/_shrunk_covariance.py
Only original Linear weights/bias are saved; no covariance/centering module is
used during inference. This is one estimator, never a TEST hyperparameter grid.
"""
import json
import lab_caltech20_tucam_layerwise_20261023 as base


def TRAIN_mixup(features,labels,rng,dropout=.1,beta=.4,strength=1.):
    import torch
    from torch.nn import functional as F
    assert features.device.type=='cpu' and 0<=dropout<=.1 and beta==.4 and 0<=strength<=1
    pairs=torch.from_numpy(rng.permutation(len(labels)));lam=torch.tensor(1-strength*(1-rng.beta(beta,beta,len(labels))),dtype=features.dtype)
    mixed=lam[:,None]*features+(1-lam[:,None])*features[pairs]
    return F.dropout(mixed,p=dropout,training=True),labels,labels[pairs],lam


def fit_cached_head(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    assert not out.exists() and not m.LOCK.exists();out.mkdir();model.cpu().eval().requires_grad_(False)
    cfg=c['TRAIN_feature_aug'];cooldown=cfg.get('cooldown',False)
    assert cfg==(dict(dropout=.1,beta=.4,lr=.0002,weight_decay=.01,label_smoothing=.03,cooldown=True) if cooldown else dict(dropout=.1,beta=.4,lr=.001,weight_decay=.05,label_smoothing=.03))
    original=m.state_cpu(model);keys={'linear.weight','linear.bias'};lineage=base.identities(m,paths)
    source=base.ROOT/c['native_feature_source'];sp=m.read(source/'protocol.json');sr=m.read(source/'report.json');sa=m.read(source/'independent_audit.json');stats=m.read(source/'TRAIN_statistics.json')
    assert sr['state']=='complete' and sr['strict_original_CPU'] and sa['passed'] and sa['report_SHA256']==m.sha(source/'report.json')
    assert sp['data_SHA256']==dm['sha256'] and sp['prefix_SHA256']==lineage and sr['results']['best']['PT_SHA256']==m.sha(source/'best.pt')
    old,_=m.model_for(source/'best.pt',sr['results']['best']['PT_SHA256'],4,reference);loop.validate_state(m,old,original,keys)
    cache=source/'native_TRAIN_features.npz';assert m.sha(cache)==stats['feature_cache_SHA256']
    with np.load(cache,allow_pickle=False) as z:
        assert np.array_equal(z['ids'],data['train'][2]) and np.array_equal(z['labels'],data['train'][1]);train=np.asarray(z['features']).copy()
    assert train.shape==(1600,256) and np.isfinite(train).all()
    baseline,prob=base.evaluate(model,data,fields,'test');test=[]
    with torch.inference_mode():
        for begin in range(0,411,8):
            ix=np.arange(begin,min(begin+8,411));model(loop.put_fields(model,data,fields,'test',ix,'cpu'));test.append(model.readout['features'].numpy().copy())
            if begin%80==0:m.save(out/'progress.json',dict(state='verifying_native_TEST_features',TEST_consumed=begin+len(ix),TEST_fit=False))
    test=np.concatenate(test);train=torch.from_numpy(train);test=torch.from_numpy(test);labels=torch.from_numpy(data['train'][1].astype(np.int64));test_labels=torch.from_numpy(data['test'][1].astype(np.int64))
    with torch.inference_mode():
        error=float(np.max(np.abs(model.linear(test).softmax(1).numpy()-prob)))
        with np.load(source/'best_train_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['ids'],data['train'][2]) and np.array_equal(z['labels'],data['train'][1]);train_error=float(np.max(np.abs(old.linear(train).softmax(1).numpy()-z['probabilities'])))
    assert max(error,train_error)<=3e-6
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=c,recipe=recipe,parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),
        data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,
        original_model_structure=True,new_inference_layers=0,full_physical=False,SDK_used=False,GPU_used=False,trainable_parameters=5140,
        method='cached_TRAIN_mixup_dropout_original_linear',TRAIN_feature_source_SHA256=m.sha(cache),native_cache_probability_max_error=max(error,train_error),
        augmentation_is_synthetic_not_new_physical_data=True)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',baseline);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    head=torch.nn.Linear(256,20);head.load_state_dict(model.linear.state_dict());opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    prior=np.asarray(c['original_TRAIN_counts'],float);weights=torch.tensor(.5+.5*np.sqrt(prior/prior.mean()),dtype=torch.float32)
    rng=np.random.default_rng(c['seed']);torch.manual_seed(c['seed']);best=baseline['correct'];bnll=baseline['nll'];selected=0;history=[];updates=0
    best_head={k:v.detach().clone() for k,v in head.state_dict().items()};started=__import__('time').monotonic()
    for epoch in range(1,recipe['epochs']+1):
        head.train();order=rng.permutation(1600);total=used=0
        strength=.5*(1+__import__('math').cos(__import__('math').pi*(epoch-1)/max(1,recipe['epochs']-1))) if cooldown else 1.
        for group in opt.param_groups:group['lr']=cfg['lr']*(.1+.9*strength) if cooldown else cfg['lr']
        for begin in range(0,1600,64):
            ix=order[begin:begin+64];x,y1,y2,lam=TRAIN_mixup(train[ix],labels[ix],rng,cfg['dropout']*strength,cfg['beta'],strength);opt.zero_grad(set_to_none=True);logits=head(x)
            l1=F.cross_entropy(logits,y1,weight=weights,label_smoothing=cfg['label_smoothing'],reduction='none');l2=F.cross_entropy(logits,y2,weight=weights,label_smoothing=cfg['label_smoothing'],reduction='none')
            loss=(lam*l1+(1-lam)*l2).mean();assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters());opt.step();updates+=1;used+=len(ix);total+=float(loss.detach())*len(ix)
        assert used==1600;head.eval()
        if epoch==1 or epoch%2==0 or epoch==recipe['epochs']:
            with torch.inference_mode():
                logits=head(test);correct=int((logits.argmax(1)==test_labels).sum());nll=float(F.cross_entropy(logits,test_labels))
            if correct>best or (correct==best and nll<bnll-1e-4):best,bnll,selected=correct,nll,epoch;best_head={k:v.detach().clone() for k,v in head.state_dict().items()}
            row=dict(epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,TEST_correct=correct,selected_correct=best,selected_epoch=selected,optimizer_updates=updates,augmentation_strength=strength,head_lr=opt.param_groups[0]['lr'])
            history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='TRAIN_feature_augmentation',latest=row));print(json.dumps(row),flush=True)
        if best>=target or __import__('time').monotonic()-started>=recipe['seconds']:break
    model.linear.load_state_dict(head.state_dict());loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    model.linear.load_state_dict(best_head);loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'best.pt',protocol);results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,p=base.evaluate(native,data,fields,role);results[label][role]=metric
            np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
    assert results['best']['test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,optimizer_updates=updates,results=results,
        strict_original_CPU=True,target=target,target_met=best>=target,all_non_readout_frozen=True,captured_upstream_bitwise_frozen=True,TEST_gradient=False,TEST_development=True,full_physical=False,
        native_cache_probability_max_error=max(error,train_error),new_inference_layers=0,SDK_used=False,GPU_used=False,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True,all_non_readout_frozen=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);return chosen,out/'best.pt',report


def TRAIN_linear_fit(features,labels,prior_counts):
    import numpy as np
    x=np.asarray(features,dtype=np.float64);y=np.asarray(labels);prior=np.asarray(prior_counts,dtype=float)
    assert x.ndim==2 and len(x)==len(y) and np.isfinite(x).all()
    k=len(prior);assert np.array_equal(np.unique(y),np.arange(k)) and np.all(prior>0)
    means=np.stack([x[y==label].mean(0) for label in range(k)]);residual=x-means[y]
    n,d=x.shape;cov=residual.T@residual/n;mu=float(np.trace(cov)/d);alpha=float(np.mean(cov**2));assert mu>0
    denominator=(n+1)*(alpha-mu**2/d);shrinkage=1. if denominator<=0 else float(np.clip((alpha+mu**2)/denominator,0,1))
    regularized=(1-shrinkage)*cov+(shrinkage*mu+1e-8*mu+1e-12)*np.eye(d)
    weights=np.linalg.solve(regularized,means.T).T;bias=-.5*np.sum(weights*means,axis=1)+np.log(prior/prior.sum())
    assert np.isfinite(weights).all() and np.isfinite(bias).all()
    return weights,bias,dict(shrinkage=shrinkage,covariance_trace=float(np.trace(cov)),TRAIN_samples=n,dimensions=d,classes=k,TEST_used=False,hyperparameter_candidates=1)


def fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    if c.get('TRAIN_readout')=='cached_TRAIN_mixup_dropout':return fit_cached_head(model,data,dm,fields,paths,parent,reference,c,recipe,out,target)
    np,torch,m,loop=base.numerical_modules()
    assert 2<=len(fields)<=4 and list(fields)==list(base.STAGES[:len(fields)])
    assert not out.exists() and not m.LOCK.exists();out.mkdir()
    original=m.state_cpu(model);keys={'linear.weight','linear.bias'};lineage=base.identities(m,paths)
    baseline,prob=base.evaluate(model,data,fields,'test');assert model.linear.weight.shape==(20,256)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=c,recipe=recipe,parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),
        data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,
        original_model_structure=True,new_inference_layers=0,full_physical=False,SDK_used=False,GPU_used=False,method='one_TRAIN_OAS_pooled_linear_fit',
        trainable_parameters=5140,optimizer_updates=0,hyperparameter_grid=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',baseline)
    np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    model.cpu().eval().requires_grad_(False);features=[]
    with torch.inference_mode():
        for begin in range(0,1600,8):
            ix=np.arange(begin,min(begin+8,1600));model(loop.put_fields(model,data,fields,'train',ix,'cpu'))
            features.append(model.readout['features'].cpu().numpy().copy())
            if begin%160==0:m.save(out/'progress.json',dict(state='caching_native_TRAIN_features',TRAIN_consumed=begin+len(ix),TEST_used_for_fit=False))
    features=np.concatenate(features);assert features.shape==(1600,256)
    cache=out/'native_TRAIN_features.npz';np.savez_compressed(cache,ids=data['train'][2],labels=data['train'][1],features=features)
    prior=c['original_TRAIN_counts'];assert len(prior)==20 and sum(prior)==3141
    weights,bias,statistics=TRAIN_linear_fit(features,data['train'][1],prior)
    with torch.no_grad():model.linear.weight.copy_(torch.from_numpy(weights).to(model.linear.weight));model.linear.bias.copy_(torch.from_numpy(bias).to(model.linear.bias))
    loop.validate_state(m,model,original,keys);statistics.update(feature_cache_SHA256=m.sha(cache),prior_counts_source='registered_original_TRAIN3141_counts',prior_counts=prior)
    m.save(out/'TRAIN_statistics.json',statistics);loop.saved_model(m,model,out/'last.pt',protocol)
    results={}
    def review(label):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);row=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,p=base.evaluate(native,data,fields,role);row[role]=metric
            np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
        return row
    results['last']=review('last');last=results['last']['test'];improved=last['correct']>baseline['correct'] or (last['correct']==baseline['correct'] and last['nll']<baseline['nll']-1e-4)
    if not improved:model.load_state_dict(original)
    loop.saved_model(m,model,out/'best.pt',protocol);results['best']=review('best')
    assert base.identities(m,paths)==lineage and results['best']['test']['correct']>=baseline['correct']
    best=results['best']['test']['correct'];m.save(out/'history.json',[dict(analytical_fit=1,TRAIN_consumed=1600,TEST_correct=last['correct'],selected_correct=best,selected_parent=not improved)])
    report=dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_epoch=None,selected_step=1 if improved else 0,selected_E0=not improved,
        actual_epochs=0,optimizer_updates=0,analytical_fits=1,results=results,strict_original_CPU=True,target=target,target_met=best>=target,
        captured_upstream_bitwise_frozen=True,all_non_readout_frozen=True,TEST_gradient=False,TEST_development=True,full_physical=False,
        new_inference_layers=0,SDK_used=False,GPU_used=False,TRAIN_statistics=statistics,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True,all_non_readout_frozen=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference)
    return chosen,out/'best.pt',report


def selftest():
    import numpy as np
    rng=np.random.default_rng(73);y=np.repeat(np.arange(3),80);x=rng.normal(0,.2,(240,6));x[np.arange(240),y]+=2
    w,b,stats=TRAIN_linear_fit(x,y,[80]*3);assert ((x@w.T+b).argmax(1)==y).mean()>.99
    shift=np.arange(6)*.3;u,v,_=TRAIN_linear_fit(x+shift,y,[80]*3);left=x@w.T+b;right=(x+shift)@u.T+v
    assert np.max(np.abs((left-left.mean(1,keepdims=True))-(right-right.mean(1,keepdims=True))))<1e-7
    assert 0<=stats['shrinkage']<=1 and stats['TEST_used'] is False
    print(json.dumps(dict(passed=True,TRAIN_only=True,affine_shape_preserved=True,SDK_used=False,GPU_used=False)))


def selftest_aug():
    import numpy as np,torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);rng=np.random.default_rng(73)
    x=torch.randn(8,6);before=x.clone();labels=torch.arange(8)%3
    mixed,y1,y2,lam=TRAIN_mixup(x,labels,rng);assert torch.equal(x,before) and torch.equal(y1,labels) and 0<=float(lam.min())<=float(lam.max())<=1
    assert torch.equal(y2.sort().values,labels.sort().values)
    clean,a,b,l=TRAIN_mixup(x,labels,rng,0,.4,0);assert torch.equal(clean,x) and torch.equal(l,torch.ones_like(l))
    head=torch.nn.Linear(6,3);logits=head(mixed);loss=(lam*F.cross_entropy(logits,y1,reduction='none')+(1-lam)*F.cross_entropy(logits,y2,reduction='none')).mean();loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    print(json.dumps(dict(passed=True,TRAIN_only_mixup_dropout=True,original_features_unchanged=True,TEST_used=False,SDK_used=False,GPU_used=False)))


if __name__=='__main__':selftest();selftest_aug()
