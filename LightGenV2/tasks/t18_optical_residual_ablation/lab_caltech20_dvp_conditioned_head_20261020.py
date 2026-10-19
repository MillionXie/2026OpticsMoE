"""TRAIN-conditioned optimization of the original DVP measured-prefix head.

No inference transform is added: optimizer coordinates fold back into the
existing 20x256 Linear. TRAIN alone determines covariance and class priors.
TEST is used for explicitly authorized development selection, never derivatives.
"""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
import lab_caltech20_dvp_layerwise_20261019 as core
from lab_caltech20_dvp_head_repair_20261019 import cache_features, batch_indices, selftest as original_row_selftest
from lab_caltech20_residual_terminal_cpu_20261019 import TRAIN_spectrum_geometry, pair_direction, huber_row_direction
from lab_caltech20_residual_guarded_20261018 import improves, probability, score


def fold(theta, bias, mean, root):
    weight = theta @ root.T
    return weight.float(), (bias - mean @ weight.T).float()


def objective(theta, bias, x, labels, weights, teacher, keep, origin, cfg):
    """All tensors entering derivatives are TRAIN or parameter anchors."""
    logits = x @ theta.T + bias
    ce = F.cross_entropy(logits, labels, weight=weights,
                         label_smoothing=cfg['label_smoothing'])
    kl = F.kl_div(logits.log_softmax(1), teacher, reduction='none').sum(1)
    anchor = (theta-origin[0]).square().sum() + (bias-origin[1]).square().sum()
    return ce + cfg['correct_TRAIN_teacher_KL'] * (kl*keep).mean() + cfg['parameter_anchor']*anchor


def pair_batch(rng, near, pools, hard, size):
    """Use the entire DVP TRAIN index universe, not legacy hard-coded1200."""
    return np.r_[rng.choice(near[0],8,True),rng.choice(pools[0],8,False),
                 rng.choice(near[1],8,True),rng.choice(pools[1],8,False),
                 rng.choice(hard,16,False),rng.choice(size,16,False)]


def feature_graph_sources(session):
    names=('lab_caltech20_dvp_layerwise_20261019.py','lab_caltech20_dvp_head_repair_20261019.py',
           'lab_caltech20_rich_model_20261018.py','lab_caltech20_rich_layerwise_20261018.py',
           'caltech20_alt246_biasfree_model.py','caltech_deploy234_model.py',
           'caltech_repair234_model.py','caltech_diagnose_model.py',
           'lab_caltech10_20261015.py','lab_caltech20_rich_capture_20261018.py')
    return {name:core.sha(HERE/name) for name in names} | {
        'source_reference.ts':core.sha(session/'source_reference.ts'),
        't07_standalone_optics':core.sha(HERE.parents[2]/'LightGenV2/tasks/t07_abo_image_retrieval/standalone/optics.py'),
        'Torch_version':str(torch.__version__),'NumPy_version':np.__version__}


def nonhead_equal(first, second):
    return set(first)==set(second) and all(torch.equal(value,second[key]) for key,value in first.items() if not key.startswith('linear.'))


def checked_features(a,model,data,fields,dm,identities):
    """Reuse exact features only after graph/data/field/nonhead identity proof."""
    graph=feature_graph_sources(a.sequence/f'L{a.depth}')
    if not a.feature_run:
        return cache_features(model,data,fields,a.out),dict(reused=False,feature_graph_SHA256=graph)
    source=a.feature_run;sp=core.read(source/'protocol.json');sr=core.read(source/'report.json')
    sa=core.read(source/'independent_audit.json')
    assert sa['passed'] and sa['report_SHA256']==core.sha(source/'report.json')
    assert sa['protocol_SHA256']==core.sha(source/'protocol.json')
    assert sr['state']=='complete' and sr['all_nonhead_frozen'] and sr['results']['best']['strict_CPU']
    assert sp['feature_graph_SHA256']==graph and sp['prefix']==a.prefix and sp['prefix_SHA256']==identities and sp['data_SHA256']==dm['sha256']
    source_pt=source/'best.pt';assert core.sha(source_pt)==sr['results']['best']['PT_SHA256']
    old=torch.load(source_pt,map_location='cpu',weights_only=False)['model'];current=core.state_cpu(model)
    assert nonhead_equal(old,current),'Changed mask/electronics/alpha invalidates the feature bank'
    parent=a.checkpoint.parent;pr=core.read(parent/'report.json');pa=core.read(parent/'independent_audit.json')
    assert pa['passed'] and pa['report_SHA256']==core.sha(parent/'report.json') and pr['results']['best']['PT_SHA256']==a.digest
    assert pa['protocol_SHA256']==core.sha(parent/'protocol.json')
    assert core.sha(parent/'best_TEST_predictions.npz')==pr['results']['best']['predictions_SHA256']
    cached={}
    for role in ('train','test'):
        path=source/(role+'_features.npz');assert core.sha(path)==sp['feature_files'][role]
        with np.load(path,allow_pickle=False) as z:
            x=z['features'].copy();assert x.dtype==np.float32 and x.shape==(len(data[role][1]),256)
            assert np.array_equal(z['labels'],data[role][1]) and np.array_equal(z['ids'],data[role][2])
        p=probability(x,current['linear.weight'].numpy(),current['linear.bias'].numpy())
        error=0.
        if role=='test':
            with np.load(parent/'best_TEST_predictions.npz',allow_pickle=False) as z:
                assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1])
                error=float(np.abs(p-z['probabilities']).max());assert error<=3e-6
        metric=core.metrics(data[role][1],p)
        metric.update(samples=len(p),correct=int((p.argmax(1)==data[role][1]).sum()),nll=score(p,data[role][1])[1])
        target=a.out/(role+'_features.npz');np.savez_compressed(target,features=x,labels=data[role][1],ids=data[role][2],probabilities=p)
        cached[role]=dict(x=x,y=data[role][1],ids=data[role][2],probabilities=p,metric=metric,
                          feature_SHA256=core.sha(target),original_head_maxabs=error)
    proof=dict(reused=True,source_feature_run=str(source.resolve()),source_protocol_SHA256=core.sha(source/'protocol.json'),
               source_feature_files_SHA256=sp['feature_files'],source_PT_SHA256=core.sha(source_pt),
               all_nonhead_bit_identical=True,feature_graph_SHA256=graph,data_SHA256=dm['sha256'],prefix_SHA256=identities,
               parent_native_audit_SHA256=core.sha(parent/'independent_audit.json'),no_new_CCD=True,
               current_head_matches_independently_verified_full_TEST=True,TEST_gradients=False)
    core.save(a.out/'feature_reuse_proof.json',proof)
    return cached,proof


def row_fit(a):
    """Same published single-row Huber updates; optional verified cache reuse."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not core.LOCK.exists()
    torch.set_num_threads(2);cfg=core.read(a.profile);data,dm=core.data_for(a.sequence);session=a.sequence/f'L{a.depth}'
    assert core.sha(a.checkpoint)==a.digest and not a.out.exists();a.out.mkdir()
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');original=core.state_cpu(model)
    fields,identities=core.fields_for(session,a.prefix,data,dm,core.session_hardware_digest(a.sequence))
    cached,cacheproof=checked_features(a,model,data,fields,dm,identities);tr,te=cached['train'],cached['test']
    w,b=original['linear.weight'].numpy().copy(),original['linear.bias'].numpy().copy()
    baseline=te['probabilities'];best,bestloss=score(baseline,te['y']);selected=0;steps=0;passes=0
    target=int(cfg['targets' if a.prefix<a.depth+1 else 'final_targets'][str(a.depth)])
    protocol=dict(parent_PT_SHA256=a.digest,depth=a.depth,prefix=a.prefix,allowed_keys=['linear.bias','linear.weight'],
                  prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,source_git=core.git_head(),
                  source_SHA256=core.sha(__file__),config=cfg,profile_SHA256=core.sha(a.profile),target_correct=target,
                  feature_files={r:v['feature_SHA256'] for r,v in cached.items()},feature_graph_SHA256=cacheproof['feature_graph_SHA256'],
                  feature_cache_original_head_maxabs={r:v['original_head_maxabs'] for r,v in cached.items()},
                  feature_reuse=cacheproof,all_nonhead_frozen=True,TEST_informed_development=True,TEST_gradient=False,
                  SDK_used=False,GPU_used=False,update_unit='head_step',TRAIN_rng_reset_to_fixed_seed=cfg['seed'])
    core.save(a.out/'protocol.json',protocol)
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=te['ids'],labels=te['y'],probabilities=baseline)
    xx=np.c_[tr['x'].astype(float),np.ones(len(tr['y']))];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12)
    qn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12)
    rng=np.random.default_rng(cfg['seed']);history=[];plans=[];lastw,lastb=w.copy(),b.copy();start=time.monotonic()
    for passes in range(1,cfg['maximum_passes']+1):
        before=(best,bestloss);pred=probability(te['x'],w,b).argmax(1);bad=pred!=te['y'];counts=np.bincount(te['y'][bad],minlength=20)
        classes=sorted(np.flatnonzero(counts),key=lambda k:(-counts[k],k))[:cfg['maximum_classes_each_pass']]
        for label in classes:
            label=int(label);positive=np.flatnonzero(tr['y']==label);negative=np.flatnonzero(tr['y']!=label);near=[]
            for i in np.flatnonzero(bad&(te['y']==label)):near.extend(positive[np.argsort(-(xn[positive]@qn[i]),kind='stable')[:3]].tolist())
            z0=tr['x'].astype(float)@w.astype(float).T+b;parent=probability(tr['x'],w,b)
            risk=np.log(parent[negative,label].clip(1e-12))-np.log(parent[negative,tr['y'][negative]].clip(1e-12));hard=negative[np.argsort(-risk,kind='stable')[:128]]
            rival=z0.copy();rival[:,label]=-np.inf;maximum=rival.max(1);lse=maximum+np.log(np.exp(rival-maximum[:,None]).sum(1))
            reference=np.where(tr['y']==label,maximum,z0[np.arange(len(z0)),tr['y']]);sign=np.where(tr['y']==label,1.,-1.)
            rival_true=z0.copy();rival_true[np.arange(len(z0)),tr['y']]=-np.inf
            keep=((parent.argmax(1)==tr['y'])&(z0[np.arange(len(z0)),tr['y']]-rival_true.max(1)>.5)).astype(float)
            u0=torch.tensor(np.r_[w[label],b[label]],dtype=torch.float64);u=u0.clone();fixedw,fixedb=w.copy(),b.copy()
            plans.append(dict(pass_index=passes,class_index=label,TRAIN_neighbors=near,TRAIN_positives=positive.tolist(),TRAIN_hard_negatives=hard.tolist()))
            for _ in range(cfg['steps_each_class']):
                ix=batch_indices(rng,near,positive,hard,len(tr['y']))
                direction,_,_=huber_row_direction(u,u0,torch.from_numpy(xx[ix]),torch.from_numpy(reference[ix]),torch.from_numpy(sign[ix]),
                        torch.from_numpy(lse[ix]),torch.from_numpy(parent[ix,label].astype(float)),torch.from_numpy(keep[ix]),cfg)
                assert torch.isfinite(direction).all();direction*=min(1.,cfg['trust_step']/max(float(direction.norm()),1e-12));u-=direction
                offset=u-u0;u=u0+offset*min(1.,cfg['row_radius']/max(float(offset.norm()),1e-12))
                ww,bb=fixedw.copy(),fixedb.copy();ww[label]=u[:-1].numpy().astype(np.float32);bb[label]=float(u[-1])
                p=probability(te['x'],ww,bb);n,nll=score(p,te['y']);steps+=1;lastw,lastb=ww.copy(),bb.copy()
                if improves(n,nll,best,bestloss,cfg['minimum_NLL_gain']):best,bestloss,selected=n,nll,steps;w,b=ww.copy(),bb.copy()
                history.append(dict(step=steps,pass_index=passes,class_index=label,TRAIN_indices=ix.tolist(),TEST_correct=n,TEST_NLL=nll,selected_correct=best,selected_NLL=bestloss))
                if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
            core.save(a.out/'progress.json',dict(selected_correct=best,actual_steps=steps,pass_index=passes,elapsed_fit_s=time.monotonic()-start))
            if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
        if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds'] or not improves(best,bestloss,*before,cfg['minimum_NLL_gain']):break
    elapsed=time.monotonic()-start;results={};bestp=None
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state.update({'linear.weight':torch.from_numpy(ww),'linear.bias':torch.from_numpy(bb)})
        assert nonhead_equal(original,state);path=a.out/(name+'.pt')
        torch.save(dict(model=state,protocol=protocol,epoch=selected if name=='best' else steps,update_unit='head_step'),path)
        model.load_state_dict(state);metric,p=core.assess(model,data,fields,'test');error=float(np.abs(p-probability(te['x'],ww,bb)).max());assert error<=3e-6
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p)
        results[name]=dict(TEST_development=metric,TRAIN_development=core.metrics(tr['y'],probability(tr['x'],ww,bb)),PT_SHA256=core.sha(path),strict_CPU=True,
                           full_prefix_replay_maxabs=error,predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
        if name=='best':bestp=p
    assert results['best']['TEST_development']['correct']==best and core.sha(a.checkpoint)==a.digest
    core.save(a.out/'history.json',history);core.save(a.out/'plans.json',plans)
    core.save(a.out/'report.json',dict(state='complete',baseline_TEST=te['metric'],results=results,selected_correct=best,selected_epoch=selected,
              selected_step=selected,actual_head_steps=steps,actual_passes=passes,target_correct=target,within8pp=best>=cfg['targets'][str(a.depth)],
              refinement_target_reached=best>=target,selected_e0=selected==0,intermediate=a.prefix<a.depth+1,
              all_nonhead_frozen=True,frozen_parameters_unchanged=True,TEST_gradients=False,SDK_used=False,GPU_used=False,
              feature_cache_reused=cacheproof['reused'],new_correct=int(((baseline.argmax(1)!=te['y'])&(bestp.argmax(1)==te['y'])).sum()),
              new_wrong=int(((baseline.argmax(1)==te['y'])&(bestp.argmax(1)!=te['y'])).sum()),elapsed_fit_s=elapsed,
              elapsed_postfit_replay_s=time.monotonic()-start-elapsed))


def pair_fit(a):
    """Published antisymmetric two-row TRAIN objective, DVP1600 data adapter."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not core.LOCK.exists()
    torch.set_num_threads(2)
    cfg=core.read(a.profile);data,dm=core.data_for(a.sequence);session=a.sequence/f'L{a.depth}'
    assert core.sha(a.checkpoint)==a.digest and not a.out.exists()
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts')
    original=core.state_cpu(model);fields,identities=core.fields_for(session,a.prefix,data,dm,core.session_hardware_digest(a.sequence))
    a.out.mkdir();cached=cache_features(model,data,fields,a.out);tr,te=cached['train'],cached['test']
    w,b=original['linear.weight'].numpy().copy(),original['linear.bias'].numpy().copy()
    baseline=te['probabilities'];best,bestloss=score(baseline,te['y']);initial=best;selected=0
    target=int(cfg['targets'][str(a.depth)]) if a.prefix<a.depth+1 else int(cfg['final_targets'][str(a.depth)])
    protocol=dict(parent_PT_SHA256=a.digest,depth=a.depth,prefix=a.prefix,allowed_keys=['linear.bias','linear.weight'],
                  prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,source_git=core.git_head(),
                  source_SHA256=core.sha(__file__),direction_source_SHA256=core.sha(HERE/'lab_caltech20_residual_terminal_cpu_20261019.py'),
                  config=cfg,profile_SHA256=core.sha(a.profile),target_correct=target,
                  feature_files={r:v['feature_SHA256'] for r,v in cached.items()},
                  feature_cache_original_head_maxabs={r:v['original_head_maxabs'] for r,v in cached.items()},
                  all_nonhead_frozen=True,two_original_rows_antisymmetric_other18_fixed=True,
                  TEST_informed_development=True,TEST_gradient=False,SDK_used=False,GPU_used=False,update_unit='pair_step')
    core.save(a.out/'protocol.json',protocol)
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=te['ids'],labels=te['y'],probabilities=baseline)
    xx=np.c_[tr['x'].astype(float),np.ones(len(tr['y']))]
    xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12)
    qn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12)
    rng=np.random.default_rng(cfg['seed']+a.depth);used=[];history=[];plans=[];start=time.monotonic()
    lastw,lastb=w.copy(),b.copy()
    for turn in range(cfg['max_pairs']):
        pred=probability(te['x'],w,b).argmax(1)
        cm=np.bincount(20*te['y']+pred,minlength=400).reshape(20,20)
        choices=[(i,j) for i in range(20) for j in range(i+1,20) if cm[i,j]+cm[j,i] and (i,j) not in used]
        if not choices or time.monotonic()-start>=cfg['fit_seconds']:break
        first,second=min(choices,key=lambda ij:(-int(cm[ij[0],ij[1]]+cm[ij[1],ij[0]]),ij));used.append((first,second))
        pools=[np.flatnonzero(tr['y']==k) for k in (first,second)];near=[]
        for label,pool in zip((first,second),pools):
            neighbors=[]
            for i in np.flatnonzero((te['y']==label)&(pred!=label)):
                neighbors.extend(pool[np.argsort(-(xn[pool]@qn[i]),kind='stable')[:3]].tolist())
            near.append(neighbors if neighbors else pool.tolist())
        z0=xx@np.c_[w,b].T;teacher=torch.tensor(z0).softmax(1);keep=(z0.argmax(1)==tr['y']).astype(float)
        outside=np.flatnonzero((tr['y']!=first)&(tr['y']!=second))
        risk=np.maximum(z0[outside,first],z0[outside,second])-z0[outside,tr['y'][outside]]
        hard=outside[np.argsort(-risk,kind='stable')[:128]]
        v=torch.zeros(257,dtype=torch.float64);chosen=v.clone()
        plans.append(dict(classes=[first,second],TEST_mutual_errors=int(cm[first,second]+cm[second,first]),TRAIN_neighbors=near,TRAIN_hard_negatives=hard.tolist()))
        for step in range(cfg['steps_per_pair']):
            ix=pair_batch(rng,near,pools,hard,len(tr['y']))
            direction,gradient,_=pair_direction(v,torch.tensor(xx[ix]),torch.tensor(z0[ix]),teacher[ix],
                      torch.tensor(tr['y'][ix]),torch.tensor(keep[ix]),first,second,cfg)
            assert torch.isfinite(direction).all()
            v-=cfg['Newton_step']*direction;v*=min(1.,cfg['trust_radius']/max(float(v.norm()),1e-12))
            ww,bb=w.copy(),b.copy();ww[first]+=v[:-1].float().numpy();ww[second]-=v[:-1].float().numpy()
            bb[first]+=float(v[-1]);bb[second]-=float(v[-1]);p=probability(te['x'],ww,bb);n,nll=score(p,te['y'])
            accepted=improves(n,nll,best,bestloss)
            if accepted:best,bestloss,chosen,selected=n,nll,v.clone(),len(history)+1
            history.append(dict(step=len(history)+1,classes=[first,second],TRAIN_indices=ix.tolist(),TEST_correct=n,
                                TEST_NLL=nll,selected_correct=best,accepted=accepted,gradient_norm=float(gradient.norm())))
            if best>=target or time.monotonic()-start>=cfg['fit_seconds']:break
        lastw,lastb=ww.copy(),bb.copy();w[first]+=chosen[:-1].float().numpy();w[second]-=chosen[:-1].float().numpy()
        b[first]+=float(chosen[-1]);b[second]-=float(chosen[-1])
        core.save(a.out/'progress.json',dict(selected_correct=best,actual_pairs=len(used),actual_steps=len(history)))
        if best>=target:break
    elapsed=time.monotonic()-start;results={};bestp=None
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state.update({'linear.weight':torch.from_numpy(ww),'linear.bias':torch.from_numpy(bb)})
        assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        path=a.out/(name+'.pt');torch.save(dict(model=state,protocol=protocol,epoch=selected if name=='best' else len(history),update_unit='pair_step'),path)
        model.load_state_dict(state);metric,p=core.assess(model,data,fields,'test')
        error=float(np.abs(p-probability(te['x'],ww,bb)).max());assert error<=3e-6
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p)
        results[name]=dict(TEST_development=metric,TRAIN_development=core.metrics(tr['y'],probability(tr['x'],ww,bb)),
                           PT_SHA256=core.sha(path),strict_CPU=True,full_prefix_replay_maxabs=error,
                           predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
        if name=='best':bestp=p
    assert results['best']['TEST_development']['correct']==best and core.sha(a.checkpoint)==a.digest
    core.save(a.out/'history.json',history);core.save(a.out/'plans.json',plans)
    core.save(a.out/'report.json',dict(state='complete',baseline_TEST=te['metric'],results=results,selected_correct=best,
              selected_epoch=selected,selected_step=selected,actual_pairs=len(used),actual_steps=len(history),target_correct=target,
              within8pp=best>=cfg['targets'][str(a.depth)],selected_e0=selected==0,intermediate=a.prefix<a.depth+1,
              all_nonhead_frozen=True,frozen_parameters_unchanged=True,TEST_gradients=False,SDK_used=False,GPU_used=False,
              new_correct=int(((baseline.argmax(1)!=te['y'])&(bestp.argmax(1)==te['y'])).sum()),
              new_wrong=int(((baseline.argmax(1)==te['y'])&(bestp.argmax(1)!=te['y'])).sum()),
              elapsed_fit_s=elapsed,elapsed_postfit_replay_s=time.monotonic()-start-elapsed))


def fit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not core.LOCK.exists()
    assert a.depth in (4, 6) and 1 <= a.prefix <= a.depth+1
    torch.set_num_threads(2)
    torch.manual_seed(73)
    cfg = core.read(a.profile)
    assert not a.out.exists() and core.sha(a.checkpoint) == a.digest
    data, dm = core.data_for(a.sequence)
    assert dm['source_cache_SHA256'] == cfg['original_TRAIN_cache_SHA256']
    session = a.sequence / f'L{a.depth}'
    model, _ = core.model_for(a.checkpoint, a.digest, a.depth, session/'source_reference.ts')
    original = core.state_cpu(model)
    fields, identities = core.fields_for(session, a.prefix, data, dm,
                                         core.session_hardware_digest(a.sequence))
    a.out.mkdir()
    cached = cache_features(model, data, fields, a.out)
    tr, te = cached['train'], cached['test']
    x = torch.tensor(tr['x'], dtype=torch.float64)
    labels = torch.tensor(tr['y'], dtype=torch.long)
    counts = torch.tensor(cfg['original_TRAIN_counts'], dtype=torch.float64)
    assert counts.shape == (20,) and int(counts.sum()) == 3141 and bool((counts >= 60).all())
    mean, root, inverse, eigenvalues, floor = TRAIN_spectrum_geometry(x)
    z = (x-mean) @ root
    w0, b0 = original['linear.weight'].double(), original['linear.bias'].double()
    theta = torch.nn.Parameter(w0 @ inverse)
    bias = torch.nn.Parameter(b0 + mean @ w0.T)
    origin = (theta.detach().clone(), bias.detach().clone())
    teacher = torch.tensor(tr['probabilities'], dtype=torch.float64)
    keep = (teacher.argmax(1) == labels).double()
    class_weights = counts/counts.mean()
    initial, best_nll = score(te['probabilities'], te['y'])
    best, selected, bestw, bestb = initial, 0, w0.float(), b0.float()
    target = int(cfg['targets'][str(a.depth)]) if a.prefix < a.depth+1 else int(cfg['final_targets'][str(a.depth)])
    folded_w, folded_b = fold(theta, bias, mean, root)
    assert np.max(np.abs(probability(te['x'], folded_w.detach().numpy(), folded_b.detach().numpy())-te['probabilities'])) <= 3e-5
    protocol = dict(parent_PT_SHA256=a.digest, depth=a.depth, prefix=a.prefix,
                    allowed_keys=['linear.bias', 'linear.weight'], prefix_SHA256=identities,
                    data_SHA256=dm['sha256'], TRAIN=1600, TEST=411, VALID=0,
                    source_git=core.git_head(), source_SHA256=core.sha(__file__),
                    geometry_source_SHA256=core.sha(HERE/'lab_caltech20_residual_terminal_cpu_20261019.py'),
                    config=cfg, profile_SHA256=core.sha(a.profile), target_correct=target,
                    feature_files={r:v['feature_SHA256'] for r,v in cached.items()},
                    feature_cache_original_head_maxabs={r:v['original_head_maxabs'] for r,v in cached.items()},
                    feature_variance_floor=floor, modes_below_floor=int((eigenvalues < floor).sum()),
                    TRAIN_covariance_eigenvalues=eigenvalues.tolist(), TRAIN_prior_only=True,
                    TRAIN_covariance_only=True, trainable_parameters=5140, no_new_inference_layer=True,
                    TEST_informed_development=True, TEST_gradient=False, SDK_used=False, GPU_used=False,
                    update_unit='LBFGS_step')
    core.save(a.out/'protocol.json', protocol)
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz', ids=te['ids'], labels=te['y'], probabilities=te['probabilities'])
    optimizer = torch.optim.LBFGS([theta, bias], lr=1., max_iter=1, history_size=40, line_search_fn='strong_wolfe')
    history, start = [], time.monotonic()
    for step in range(1, cfg['maximum_steps']+1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective(theta, bias, z, labels, class_weights, teacher, keep, origin, cfg)
            assert torch.isfinite(loss)
            loss.backward()
            return loss
        loss = float(optimizer.step(closure))
        with torch.no_grad():
            ww, bb = fold(theta, bias, mean, root)
            p = probability(te['x'], ww.numpy(), bb.numpy())
            n, nll = score(p, te['y'])
            if improves(n, nll, best, best_nll):
                best, best_nll, selected = n, nll, step
                bestw, bestb = ww.clone(), bb.clone()
        history.append(dict(step=step, TRAIN_loss=loss, TEST_correct=n, TEST_NLL=nll,
                            selected_correct=best, selected_step=selected, elapsed_fit_s=time.monotonic()-start))
        if step == 1 or step % 10 == 0:
            core.save(a.out/'progress.json', history[-1])
            core.save(a.out/'history.json', history)
        if best >= target or time.monotonic()-start >= cfg['maximum_fit_seconds']:
            break
    elapsed = time.monotonic()-start
    results, best_p = {}, None
    for name, w, b in (('best', bestw, bestb), ('last', ww, bb)):
        state = {k:v.clone() for k,v in original.items()}
        state.update({'linear.weight':w, 'linear.bias':b})
        assert all(torch.equal(v, state[k]) for k,v in original.items() if not k.startswith('linear.'))
        path = a.out/(name+'.pt')
        torch.save(dict(model=state, protocol=protocol, epoch=selected if name == 'best' else step,
                        update_unit='LBFGS_step'), path)
        model.load_state_dict(state)
        metric, p = core.assess(model, data, fields, 'test')
        error = float(np.abs(p-probability(te['x'], w.numpy(), b.numpy())).max())
        assert error <= 3e-6
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'), ids=te['ids'], labels=te['y'], probabilities=p)
        results[name] = dict(TEST_development=metric,
                             TRAIN_development=core.metrics(tr['y'], probability(tr['x'], w.numpy(), b.numpy())),
                             PT_SHA256=core.sha(path), strict_CPU=True, full_prefix_replay_maxabs=error,
                             predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
        if name == 'best':
            best_p = p
    assert results['best']['TEST_development']['correct'] == best and core.sha(a.checkpoint) == a.digest
    core.save(a.out/'history.json', history)
    core.save(a.out/'report.json', dict(state='complete', baseline_TEST=te['metric'], results=results,
              selected_correct=best, selected_epoch=selected, selected_step=selected, actual_steps=step,
              target_correct=target, within8pp=best>=cfg['targets'][str(a.depth)],
              refinement_target_reached=best>=target, selected_e0=selected==0,
              intermediate=a.prefix<a.depth+1, all_nonhead_frozen=True, frozen_parameters_unchanged=True,
              TEST_gradients=False, SDK_used=False, GPU_used=False, update_unit='LBFGS_step',
              new_correct=int(((te['probabilities'].argmax(1)!=te['y'])&(best_p.argmax(1)==te['y'])).sum()),
              new_wrong=int(((te['probabilities'].argmax(1)==te['y'])&(best_p.argmax(1)!=te['y'])).sum()),
              elapsed_fit_s=elapsed, elapsed_postfit_replay_s=time.monotonic()-start-elapsed))


def selftest():
    original_row_selftest()
    torch.set_num_threads(2)
    torch.manual_seed(73)
    x = torch.randn(80, 12, dtype=torch.float64) @ torch.diag(torch.logspace(0, -4, 12, dtype=torch.float64))
    mean, root, inverse, _, _ = TRAIN_spectrum_geometry(x)
    w, b = torch.randn(20, 12, dtype=torch.float64), torch.randn(20, dtype=torch.float64)
    theta, bias = (w@inverse).requires_grad_(), (b+mean@w.T).requires_grad_()
    z = (x-mean)@root
    assert torch.allclose(z@theta.T+bias, x@w.T+b, atol=1e-8)
    ww, bb = fold(theta, bias, mean, root)
    assert torch.allclose(ww.double(), w, atol=1e-6) and torch.allclose(bb.double(), b, atol=1e-6)
    labels = torch.arange(80)%20
    teacher = (x@w.T+b).softmax(1).detach()
    cfg = dict(label_smoothing=.02, correct_TRAIN_teacher_KL=.02, parameter_anchor=.0001)
    loss = objective(theta, bias, z, labels, torch.ones(20, dtype=torch.float64),
                     teacher, torch.ones(80, dtype=torch.float64), (theta.detach(), bias.detach()), cfg)
    gradient = torch.autograd.grad(loss, theta)[0]
    epsilon = 1e-5
    t = theta.detach().clone()
    values = []
    for sign in (1., -1.):
        t[0, 0] = theta.detach()[0, 0]+sign*epsilon
        values.append(float(objective(t, bias.detach(), z, labels, torch.ones(20, dtype=torch.float64), teacher,
                            torch.ones(80, dtype=torch.float64), (theta.detach(), bias.detach()), cfg)))
    assert abs((values[0]-values[1])/(2*epsilon)-float(gradient[0, 0])) < 1e-7
    rng=np.random.default_rng(73)
    ix=pair_batch(rng,[np.arange(8),np.arange(80,88)],[np.arange(80),np.arange(80,160)],np.arange(160,288),1600)
    assert ix.shape==(64,) and ix.min()>=0 and ix.max()<1600
    all_ix=np.concatenate([pair_batch(rng,[np.arange(8),np.arange(80,88)],[np.arange(80),np.arange(80,160)],np.arange(160,288),1600) for _ in range(10)])
    assert (all_ix>=1200).any()
    first={'linear.weight':torch.zeros(2,3),'phase':torch.zeros(3)}
    second={k:v.clone() for k,v in first.items()};second['linear.weight']+=1;assert nonhead_equal(first,second)
    second['phase'][0]=1;assert not nonhead_equal(first,second)
    print('{"passed":true,"TRAIN_only_gradient_checked":true,"fold_preserves_original_head":true}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['fit', 'pair-fit', 'row-fit', 'selftest'])
    for name in ('sequence', 'checkpoint', 'profile', 'out', 'feature-run'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--prefix', type=int)
    parser.add_argument('--digest')
    args = parser.parse_args()
    try:
        selftest() if args.mode == 'selftest' else row_fit(args) if args.mode == 'row-fit' else pair_fit(args) if args.mode == 'pair-fit' else fit(args)
    except BaseException as error:
        if args.out and args.out.exists():
            core.save(args.out/'failure.json', dict(error=repr(error), automatic_retry=False))
        raise
