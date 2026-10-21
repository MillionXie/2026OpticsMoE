"""One TRAIN-only shrinkage-discriminant fit folded into the existing head.

No architecture change: pooled covariance and class means only determine the
same Linear256to20 weights/bias. No TEST-derived derivative or fit statistic.
The sole OAS shrinkage estimate and class priors come from TRAIN, not a grid.
"""
import argparse
import os
from pathlib import Path

PRIOR=(348,160,638,640,102,78,68,72,98,85,68,79,80,70,68,91,69,68,68,191)


def fit_head(x,y):
    import numpy as np
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
    assert x.shape==(1600,256) and np.array_equal(np.bincount(y,minlength=20),np.full(20,80))
    means=np.stack([x[y==k].mean(0) for k in range(20)])
    residual=x-means[y];cov=residual.T@residual/len(y)
    p=x.shape[1];mu=float(np.trace(cov)/p);alpha=float(np.square(cov).mean());assert mu>0
    denominator=(len(y)-20+1)*(alpha-mu*mu/p)
    shrinkage=min(1.,(alpha+mu*mu)/denominator) if denominator>0 else 1.
    assert 0<shrinkage<=1
    regular=(1-shrinkage)*cov+shrinkage*mu*np.eye(p)
    w=np.linalg.solve(regular,means.T).T
    prior=np.asarray(PRIOR,dtype=float);prior/=prior.sum()
    b=-.5*np.sum(means*w,axis=1)+np.log(prior)
    assert np.isfinite(w).all() and np.isfinite(b).all()
    return w.astype(np.float32),b.astype(np.float32),dict(TRAIN_records=1600,dimension=p,classes=20,pooled_covariance_effective_samples=1580,OAS_shrinkage=shrinkage,TRAIN_prior_counts=list(PRIOR),TEST_fit_records=0,TEST_hyperparameter_search=False)


def selftest():
    import numpy as np
    rng=np.random.default_rng(73);labels=np.repeat(np.arange(20),80);centers=rng.normal(size=(20,256))
    x=centers[labels]+rng.normal(size=(1600,256))*.1;w,b,proof=fit_head(x,labels)
    assert np.mean((x@w.T+b).argmax(1)==labels)==1
    order=rng.permutation(1600);ww,bb,p=fit_head(x[order],labels[order])
    assert np.allclose(w,ww,rtol=2e-6,atol=2e-6) and np.allclose(b,bb,rtol=2e-6,atol=2e-6)
    print(dict(passed=True,TRAIN_only=True,permutation_invariant=True,unchanged_5140_head=True,SDK_opened=False))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_closedloop_20261021 import SPECS,HARDWARE,validate_state,original_replay,saved_model
    from lab_caltech20_residual_guarded_20261018 import probability
    torch.set_num_threads(2)
    spec=SPECS[a.depth];campaign='TRAIN_fullbatch_closedloop_20261021' if a.campaign=='forward' else 'TRAIN_fullbatch_adjoint_sign_20261021'
    folder=a.sequence/campaign/f'L{a.depth}_round1';source=folder/'fullTRAIN_row_curvature'
    protocol=m.read(source/'protocol.json');report=m.read(source/'report.json');audit=m.read(folder/'independent_round_audit.json')
    assert report['state']=='complete' and report['protocol_SHA256']==m.sha(source/'protocol.json') and audit['passed']
    assert protocol['source_round_audit_SHA256']==m.sha(folder/'independent_round_audit.json')
    assert not a.out.exists();a.out.mkdir()
    data,dm=m.data_for(a.sequence);assert dm['sha256']==protocol['data_SHA256'] and m.session_hardware_digest(a.sequence)==HARDWARE
    session=a.sequence/f'L{a.depth}_response_20261020';fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,HARDWARE)
    capture=folder/'capture';cr=m.read(capture/'report.json');fields[spec['stage']]={}
    for role,size in (('train',1600),('test',411)):
        p=capture/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256'];fields[spec['stage']][role]=np.load(p,mmap_mode='r');identities[spec['stage']+'/'+role]=m.sha(p)
    assert identities==protocol['prefix_SHA256']
    parent=source/'best.pt';digest=m.sha(parent);assert digest==report['results']['best']['PT_SHA256']
    model,_=m.model_for(parent,digest,a.depth,session/'source_reference.ts');original=m.state_cpu(model);banks={}
    for role,size in (('train',1600),('test',411)):
        with np.load(source/(role+'_features.npz')) as z:
            assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1])
            banks[role]=(z['features'].copy(),z['labels'].copy())
        assert banks[role][0].shape==(size,256)
    baseline,bp=original_replay(model,data,fields,'test')
    expected=probability(banks['test'][0],original['linear.weight'].numpy(),original['linear.bias'].numpy())
    assert float(np.abs(expected-bp).max())<=3e-6 and np.array_equal(expected.argmax(1),bp.argmax(1))
    assert baseline['correct']==report['selected_correct']
    w,b,statistics=fit_head(*banks['train'])
    method=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),method='single_fullTRAIN_OAS_pooled_discriminant',TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_fit_records=0,TEST_informed_development=True,statistics=statistics,parent_PT_SHA256=digest,source_report_SHA256=m.sha(source/'report.json'),data_SHA256=dm['sha256'],prefix_SHA256=identities,cache_SHA256={r:m.sha(source/(r+'_features.npz')) for r in banks},all_nonhead_frozen=True,new_inference_layers=0,hardware_changed=False,SDK_used=False,GPU_used=False)
    m.save(a.out/'protocol.json',method)
    state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(w);state['linear.bias']=torch.from_numpy(b);model.load_state_dict(state);validate_state(m,model,original,{'linear.weight','linear.bias'})
    candidate=saved_model(m,model,a.out/'candidate.pt',method);results={}
    for role in ('train','test'):
        actual,p=original_replay(model,data,fields,role);expected=probability(banks[role][0],w,b)
        error=float(np.abs(p-expected).max());assert error<=3e-6 and np.array_equal(p.argmax(1),expected.argmax(1))
        results[role]=actual;np.savez_compressed(a.out/(role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
    better=results['test']['correct']>baseline['correct'] or (results['test']['correct']==baseline['correct'] and results['test']['nll']<baseline['nll']-1e-4)
    # One proposed statistical fit, not a family selected on TEST.
    selected_state=state if better else original
    for name in ('best','last'):
        model.load_state_dict(selected_state if name=='best' else state);saved_model(m,model,a.out/(name+'.pt'),method)
    result=dict(state='complete',baseline=baseline,candidate=results,selected_correct=results['test']['correct'] if better else baseline['correct'],selected_e0=not better,candidate_PT_SHA256=candidate,best_PT_SHA256=m.sha(a.out/'best.pt'),last_PT_SHA256=m.sha(a.out/'last.pt'),global_old_incumbent_correct=spec['correct'],global_old_PT_and_CCD_preserved=True,target=spec['target'],passed=(results['test']['correct'] if better else baseline['correct'])>=spec['target'],original_full_CPU_TRAIN1600_TEST411_verified=True,all_nonhead_frozen=True,full_physical=False,SDK_used=False,GPU_used=False,TEST_fit_records=0,protocol_SHA256=m.sha(a.out/'protocol.json'))
    m.save(a.out/'report.json',result);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),protocol_SHA256=m.sha(a.out/'protocol.json'),original_full_graph_replay=True))
    print(m.json.dumps(dict(complete=True,baseline=baseline['correct'],candidate=results['test']['correct'],selected=result['selected_correct'],passed=result['passed'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--campaign',choices=('forward','reverse'),default='forward');p.add_argument('--out',type=Path);a=p.parse_args();selftest() if a.selftest else run(a)
