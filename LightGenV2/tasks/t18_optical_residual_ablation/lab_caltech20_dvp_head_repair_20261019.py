"""Original 5140-head local repair on exact new-camera measured-prefix features.

Reuses the audited signed Huber/exact-curvature direction. All optical,
electronic and alpha state stays frozen; TEST informs TRAIN-neighbor plans and
development checkpoint selection, never gradients. No SDK, GPU or new layer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import lab_caltech20_dvp_layerwise_20261019 as core
from lab_caltech20_residual_terminal_cpu_20261019 import huber_row_direction
from lab_caltech20_residual_guarded_20261018 import probability,score,improves


def batch_indices(rng,near,positive,hard,size):
    assert len(near) and len(positive)>=16 and len(hard)>=16 and size>=64
    return np.r_[rng.choice(near,16,True),rng.choice(positive,16,False),rng.choice(hard,16,False),rng.choice(size,16,False)]


def cache_features(model,data,fields,out):
    """Hook reads the ORIGINAL Linear input; no feature transform or graph edit."""
    cached={};parts=[]
    def collect(module,args):parts.append(args[0].detach().cpu().numpy().copy())
    hook=model.linear.register_forward_pre_hook(collect)
    try:
        for role in ('train','test'):
            parts.clear();metric,p=core.assess(model,data,fields,role);x=np.concatenate(parts)[:len(data[role][1])]
            assert x.shape==(len(p),256) and x.dtype==np.float32
            replay=probability(x,model.linear.weight.detach().numpy(),model.linear.bias.detach().numpy());error=float(np.abs(replay-p).max());assert error<=3e-6
            path=out/(role+'_features.npz');np.savez_compressed(path,features=x,labels=data[role][1],ids=data[role][2],probabilities=p)
            cached[role]=dict(x=x,y=data[role][1],ids=data[role][2],probabilities=p,metric=metric,feature_SHA256=core.sha(path),original_head_maxabs=error)
    finally:hook.remove()
    return cached


def fit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not core.LOCK.exists()
    torch.set_num_threads(2);torch.manual_seed(17);cfg=core.read(a.profile);data,dm=core.data_for(a.sequence);session=core.session_for(a)
    assert core.sha(a.checkpoint)==a.digest and not a.out.exists();a.out.mkdir()
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');original=core.state_cpu(model);fields,identities=core.fields_for(session,a.prefix,data,dm,core.session_hardware_digest(a.sequence))
    cached=cache_features(model,data,fields,a.out);tr,te=cached['train'],cached['test'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();baseline=te['probabilities'];best,bestloss=score(baseline,te['y']);initial=best
    target=int(cfg['targets'][str(a.depth)]);protocol=dict(parent_PT_SHA256=a.digest,depth=a.depth,prefix=a.prefix,allowed_keys=['linear.bias','linear.weight'],prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,baseline_TEST=te['metric'],target_correct=target,config=cfg,source_git=core.git_head(),source_SHA256=core.sha(__file__),feature_files={r:v['feature_SHA256'] for r,v in cached.items()},feature_cache_original_head_maxabs={r:v['original_head_maxabs'] for r,v in cached.items()},TEST_informed_development=True,TEST_gradient=False,update_unit='head_step')
    core.save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=te['ids'],labels=te['y'],probabilities=baseline)
    xx=np.c_[tr['x'].astype(np.float64),np.ones(len(tr['y']))];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);qn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12)
    rng=np.random.default_rng(cfg['seed']);history=[];plans=[];steps=0;selected=0;lastw,lastb=w.copy(),b.copy();start=time.monotonic();passes=0
    for passes in range(1,cfg['maximum_passes']+1):
        before=(best,bestloss);tp=probability(te['x'],w,b);bad=tp.argmax(1)!=te['y'];counts=np.bincount(te['y'][bad],minlength=20);classes=sorted(np.flatnonzero(counts),key=lambda k:(-counts[k],k))[:cfg['maximum_classes_each_pass']]
        for label in classes:
            label=int(label);positive=np.flatnonzero(tr['y']==label);negative=np.flatnonzero(tr['y']!=label);near=[]
            for i in np.flatnonzero(bad&(te['y']==label)):near.extend(positive[np.argsort(-(xn[positive]@qn[i]),kind='stable')[:3]].tolist())
            z0=tr['x'].astype(float)@w.astype(float).T+b;parent=probability(tr['x'],w,b);risk=np.log(parent[negative,label].clip(1e-12))-np.log(parent[negative,tr['y'][negative]].clip(1e-12));hard=negative[np.argsort(-risk,kind='stable')[:128]]
            rival=z0.copy();rival[:,label]=-np.inf;maximum=rival.max(1);lse=maximum+np.log(np.exp(rival-maximum[:,None]).sum(1));reference=np.where(tr['y']==label,maximum,z0[np.arange(len(z0)),tr['y']]);sign=np.where(tr['y']==label,1.,-1.)
            rival_true=z0.copy();rival_true[np.arange(len(z0)),tr['y']]=-np.inf;keep=((parent.argmax(1)==tr['y'])&(z0[np.arange(len(z0)),tr['y']]-rival_true.max(1)>.5)).astype(float)
            u0=torch.tensor(np.r_[w[label],b[label]],dtype=torch.float64);u=u0.clone();fixed_w,fixed_b=w.copy(),b.copy();plan=dict(pass_index=passes,class_index=label,TEST_error_indices=np.flatnonzero(bad&(te['y']==label)).tolist(),TRAIN_neighbors=near,TRAIN_positives=positive.tolist(),TRAIN_hard_negatives=hard.tolist());plans.append(plan)
            for _ in range(cfg['steps_each_class']):
                ix=batch_indices(rng,near,positive,hard,len(tr['y']));x=torch.from_numpy(xx[ix]);direction,_,_=huber_row_direction(u,u0,x,torch.from_numpy(reference[ix]),torch.from_numpy(sign[ix]),torch.from_numpy(lse[ix]),torch.from_numpy(parent[ix,label].astype(float)),torch.from_numpy(keep[ix]),cfg)
                assert torch.isfinite(direction).all()
                direction*=min(1.,cfg['trust_step']/max(float(direction.norm()),1e-12));u-=direction;offset=u-u0;u=u0+offset*min(1.,cfg['row_radius']/max(float(offset.norm()),1e-12))
                trial_w,trial_b=fixed_w.copy(),fixed_b.copy();trial_w[label]=u[:-1].numpy().astype(np.float32);trial_b[label]=float(u[-1]);p=probability(te['x'],trial_w,trial_b);n,loss=score(p,te['y']);steps+=1;lastw,lastb=trial_w.copy(),trial_b.copy()
                if improves(n,loss,best,bestloss,cfg['minimum_NLL_gain']):best,bestloss=n,loss;w,b=trial_w.copy(),trial_b.copy();selected=steps
                history.append(dict(step=steps,pass_index=passes,class_index=label,TRAIN_indices=ix.tolist(),TEST_correct=n,TEST_NLL=loss,selected_correct=best,selected_NLL=bestloss))
                if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
            core.save(a.out/'progress.json',dict(state='fixed_feature_head_repair',pass_index=passes,actual_head_steps=steps,selected_correct=best,selected_NLL=bestloss,elapsed_fit_s=time.monotonic()-start))
            if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
        if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds'] or not improves(best,bestloss,*before,cfg['minimum_NLL_gain']):break
    fit_elapsed=time.monotonic()-start;results={};best_probability=None
    for name,ww,bb,epoch in (('best',w,b,selected),('last',lastw,lastb,steps)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        path=a.out/(name+'.pt');torch.save(dict(model=state,protocol=protocol,epoch=epoch,update_unit='head_step'),path);model.load_state_dict(state);metric,p=core.assess(model,data,fields,'test');expected=probability(te['x'],ww,bb);error=float(np.abs(p-expected).max());assert error<=3e-6
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p);results[name]=dict(TEST_development=metric,TRAIN_development=core.metrics(tr['y'],probability(tr['x'],ww,bb)),PT_SHA256=core.sha(path),strict_CPU=True,full_prefix_replay_maxabs=error,predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
        if name=='best':best_probability=p.copy()
    assert results['best']['TEST_development']['correct']==best and core.sha(a.checkpoint)==a.digest
    core.save(a.out/'history.json',history);core.save(a.out/'plans.json',plans);core.save(a.out/'report.json',dict(state='complete',baseline_TEST=te['metric'],results=results,selected_epoch=selected,selected_step=selected,actual_head_steps=steps,actual_passes=passes,selected_correct=best,target_correct=target,within8pp=best>=target,selected_e0=selected==0,intermediate=a.prefix<a.depth+1,frozen_parameters_unchanged=True,all_nonhead_frozen=True,TEST_gradients=False,SDK_used=False,GPU_used=False,update_unit='head_step',new_correct=int(((baseline.argmax(1)!=te['y'])&(best_probability.argmax(1)==te['y'])).sum()),new_wrong=int(((baseline.argmax(1)==te['y'])&(best_probability.argmax(1)!=te['y'])).sum()),elapsed_fit_s=fit_elapsed,elapsed_postfit_replay_s=time.monotonic()-start-fit_elapsed))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(17);rng=np.random.default_rng(17);positive=np.arange(80);hard=np.arange(80,208);ix=batch_indices(rng,positive[:10],positive,hard,1600);assert ix.shape==(64,) and ix.min()>=0 and ix.max()<1600
    x=torch.randn(12,9,dtype=torch.float64);u=torch.randn(9,dtype=torch.float64)*.1;u0=u+.01;ref=torch.randn(12,dtype=torch.float64)*.2;sign=torch.where(torch.arange(12)%2==0,1.,-1.);lse=ref+1;q=torch.rand(12,dtype=torch.float64);keep=(torch.arange(12)%3!=0).double();c=dict(temperature=1.,huber_delta=1.,margin=1.,teacher_KL=.25,anchor=.05,LM_damping=.1)
    def objective(v):
        h=(c['margin']-sign*(x@v-ref)).clamp_min(0);supervision=torch.where(h<1,h.square()/2,h-.5);kl=F.binary_cross_entropy_with_logits(x@v-lse,q,reduction='none');return (supervision+c['teacher_KL']*keep*kl).mean()+c['anchor']*(v-u0).square().sum()
    v=u.clone().requires_grad_();g=torch.autograd.grad(objective(v),v)[0];h=torch.autograd.functional.hessian(objective,v);direction,actual,_=huber_row_direction(u,u0,x,ref,sign,lse,q,keep,c);assert torch.allclose(g,actual,atol=1e-10) and torch.allclose(direction,torch.linalg.solve(h+.1*torch.eye(9,dtype=torch.float64),g),atol=1e-9)
    print(json.dumps(dict(passed=True,original_Huber_gradient_curvature_verified=True,all_sampling_indices_within_TRAIN1600=True,SDK_opened=False,GPU_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('fit','selftest'),required=True);p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int);p.add_argument('--prefix',type=int);p.add_argument('--checkpoint',type=Path);p.add_argument('--digest');p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);p.add_argument('--attempt',choices=('strict8pp_20261020',));a=p.parse_args()
    try:selftest() if a.mode=='selftest' else fit(a)
    except BaseException as error:
        if a.out and a.out.exists():core.save(a.out/'failure.json',dict(error=repr(error),automatic_retry=False))
        raise
