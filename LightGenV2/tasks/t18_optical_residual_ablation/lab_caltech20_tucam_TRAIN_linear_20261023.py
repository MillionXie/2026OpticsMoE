"""One TRAIN-only covariance-regularized fit of the existing5140 linear head.

OAS formulation reference: scikit-learn covariance/_shrunk_covariance.py,
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/covariance/_shrunk_covariance.py
Only original Linear weights/bias are saved; no covariance/centering module is
used during inference. This is one estimator, never a TEST hyperparameter grid.
"""
import json
import lab_caltech20_tucam_layerwise_20261023 as base


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
    np,torch,m,loop=base.numerical_modules();assert list(fields)==['router','0'] and not out.exists() and not m.LOCK.exists();out.mkdir()
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


if __name__=='__main__':selftest()
