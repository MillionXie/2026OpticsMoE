"""Full1600-TRAIN row-curvature refinement of an audited new-mask suffix.

Every update includes ALL TRAIN records, not a selected small training set.
TEST errors select the row and TRAIN-neighbor emphasis, never derivatives.
Only original Linear rows change; no SDK/GPU/new inference capacity.
"""
import argparse
import math
import os
from pathlib import Path
import time


def direction(u,u0,x,reference,sign,lse,q,keep,c,sample_weight,binary=False):
    """Primal257 system; equivalent to published Woodbury curvature at uniform weights."""
    import torch
    t=c['temperature'];z=x@u;p=((z-lse)/t).sigmoid()
    if binary:
        target=torch.where(sign>0,1-c['binary_label_smoothing']/2,c['binary_label_smoothing']/2)
        supervised=(p-target)/t;curvature=p*(1-p)/t**2
    else:
        violation=(c['margin']-sign*(z-reference)).clamp_min(0)
        supervised=-sign*violation.clamp_max(c['huber_delta'])/c['huber_delta']
        curvature=((violation>0)&(violation<c['huber_delta'])).double()/c['huber_delta']
    w=sample_weight/sample_weight.sum()
    gradient=x.T@(w*(supervised+c['teacher_KL']*keep*(p-q)/t))+2*c['anchor']*(u-u0)
    d=w*(curvature+c['teacher_KL']*keep*p*(1-p)/t**2)
    hessian=x.T@(d[:,None]*x)+(2*c['anchor']+c['LM_damping'])*torch.eye(x.shape[1],dtype=x.dtype)
    return torch.linalg.solve(hessian,gradient),gradient,d


def selftest():
    import torch
    from lab_caltech20_residual_terminal_cpu_20261019 import binary_row_direction,huber_row_direction
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(64,17,dtype=torch.float64);u=torch.randn(17,dtype=torch.float64)*.05;u0=torch.zeros_like(u)
    reference=torch.rand(64,dtype=x.dtype);sign=torch.where(torch.arange(64)%3==0,1.,-1.).double();lse=torch.randn(64,dtype=x.dtype);keep=(torch.arange(64)%2).double()
    c=dict(temperature=1.,margin=1.,huber_delta=1.,teacher_KL=.25,anchor=.05,LM_damping=.1,binary_label_smoothing=.02)
    q=torch.sigmoid(x@u-lse);rows=[]
    for binary in (False,True):
        c['temperature']=2. if binary else 1.;q=torch.sigmoid((x@u-lse)/c['temperature'])
        actual,g,d=direction(u,u0,x,reference,sign,lse,q,keep,c,torch.ones(64,dtype=x.dtype),binary)
        if binary:
            target=torch.where(sign>0,.99,.01);expected,eg,ed=binary_row_direction(u,u0,x,lse,target,q,keep,c)
        else:expected,eg,ed=huber_row_direction(u,u0,x,reference,sign,lse,q,keep,c)
        error=float((actual-expected).abs().max());assert error<1e-10 and torch.allclose(g,eg,atol=1e-12) and torch.allclose(d,ed,atol=1e-12)
        rows.append(dict(binary=binary,direction_maxabs=error))
    # Full physical TRAIN dimension, all weights strictly positive.
    xx=torch.randn(1600,257,dtype=torch.float64);v=torch.zeros(257,dtype=torch.float64);s=torch.where(torch.arange(1600)<80,1.,-1.).double()
    step,g,d=direction(v,v,xx,torch.zeros(1600),s,torch.zeros(1600),torch.full((1600,),.5),torch.zeros(1600),c,torch.ones(1600,dtype=torch.float64),True)
    assert torch.isfinite(step).all() and float(step.abs().max())>0
    print(dict(passed=True,published_curvature_equivalence=rows,full_TRAIN1600_primal257_finite=True,SDK_opened=False))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_closedloop_20261021 import SPECS,HARDWARE,put_fields,validate_state,original_replay,saved_model
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    torch.set_num_threads(2);torch.manual_seed(73)
    spec=SPECS[a.depth];root=a.sequence/'TRAIN_fullbatch_closedloop_20261021';folder=root/f'L{a.depth}_round{a.round}'
    assert not a.out.exists();audit=m.read(folder/'independent_round_audit.json');report=m.read(folder/'suffix/report.json')
    assert audit['passed'] and audit['source_fit_report_SHA256']==m.sha(folder/'suffix/report.json')
    assert audit['source_capture_report_SHA256']==m.sha(folder/'capture/report.json') and not report['passed']
    data,dm=m.data_for(a.sequence);assert dm['sha256']==audit['data_SHA256'] and m.session_hardware_digest(a.sequence)==HARDWARE
    session=a.sequence/f'L{a.depth}_response_20261020';fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,HARDWARE)
    cap=folder/'capture';cr=m.read(cap/'report.json');fields[spec['stage']]={}
    for role in ('train','test'):
        p=cap/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256'];fields[spec['stage']][role]=np.load(p,mmap_mode='r');identities[spec['stage']+'/'+role]=m.sha(p)
    parent=folder/'suffix/best.pt';digest=m.sha(parent);assert digest==audit['results']['best']['PT_SHA256']
    model,_=m.model_for(parent,digest,a.depth,session/'source_reference.ts');original=m.state_cpu(model)
    banks={};a.out.mkdir()
    with torch.no_grad():
        for role,size in (('train',1600),('test',411)):
            features=[];probs=[]
            for begin in range(0,size,8):
                ix=np.arange(begin,min(size,begin+8));n=len(ix);x,n,end=m.padded(data,role,begin)
                model.measured={k:torch.from_numpy(np.asarray(v[role][begin:end]).copy()) for k,v in fields.items()}
                if n<8:model.measured={k:torch.cat([v,torch.zeros((8-n,478,478),dtype=v.dtype)]) for k,v in model.measured.items()}
                p=model(m.encode_images(x))[0][:n];features.append(model.readout['features'][:n].numpy().copy());probs.append(p.numpy().copy())
                m.save(a.out/'progress.json',dict(state='original_feature_cache',role=role,completed=end,total=size))
            z=np.concatenate(features);p=np.concatenate(probs);head=probability(z,original['linear.weight'].numpy(),original['linear.bias'].numpy());error=float(np.abs(head-p).max());assert error<=3e-6
            assert np.array_equal(head.argmax(1),p.argmax(1));banks[role]=dict(x=z,y=data[role][1],ids=data[role][2],p=p,parity=error)
            np.savez_compressed(a.out/(role+'_features.npz'),features=z,labels=data[role][1],ids=data[role][2],probabilities=p)
    tr,te=banks['train'],banks['test'];train_accuracy=float((tr['p'].argmax(1)==tr['y']).mean());binary=train_accuracy>=.8
    base=m.HERE/('lab_caltech20_DVP_binary_head_profile_20261020.json' if binary else 'lab_caltech20_DVP_head_local_profile_20261019.json');c=m.read(base)
    profile_path=m.HERE/'lab_caltech20_DVP_closedloop_head_profile_20261021.json';profile=m.read(profile_path)
    assert profile['schema']=='audited_closedloop_fullTRAIN_row_v1';c.update(profile['overrides'])
    target=spec['target'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();best,bnll=score(te['p'],te['y']);selected=0;history=[];plans=[];steps=0;passes=0
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=digest,source_round_audit_SHA256=m.sha(folder/'independent_round_audit.json'),
        data_SHA256=dm['sha256'],prefix_SHA256=identities,allowed_keys=['linear.weight','linear.bias'],TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,
        full_TRAIN_each_update=True,gradient_batch=1600,solver='primal257 equivalent published row curvature; positive TRAIN-neighbor emphasis and full replay',
        original_TRAIN_accuracy=train_accuracy,objective='binary_row' if binary else 'huber_margin',published_profile_SHA256=m.sha(base),config=c,new_profile_SHA256=m.sha(profile_path),new_profile=profile,
        all_nonhead_frozen=True,feature_parity={k:v['parity'] for k,v in banks.items()},SDK_used=False,GPU_used=False,TEST_informed_development=True,
        target_correct=target,global_old_incumbent_not_replaced_without_improvement=True)
    m.save(a.out/'protocol.json',protocol);bestw,bestb=w.copy(),b.copy();lastw,lastb=w.copy(),b.copy();start=time.monotonic()
    xx=torch.tensor(np.c_[tr['x'].astype(float),np.ones(1600)],dtype=torch.float64);xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);qn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12)
    for passes in range(1,c['maximum_passes']+1):
        prior_best=(best,bnll);tp=probability(te['x'],w,b);bad=tp.argmax(1)!=te['y'];counts=np.bincount(te['y'][bad],minlength=20)
        for label in sorted(np.flatnonzero(counts),key=lambda k:(-counts[k],k)):
            label=int(label);positive=np.flatnonzero(tr['y']==label);negative=np.flatnonzero(tr['y']!=label);hits=np.zeros(1600)
            for index in np.flatnonzero(bad&(te['y']==label)):
                chosen=positive[np.argsort(-(xn[positive]@qn[index]),kind='stable')[:3]];np.add.at(hits,chosen,1)
            sample=1+profile['TRAIN_neighbor_boost']*hits/max(float(hits.max()),1.);sample[positive]*=profile['positive_mass']/sample[positive].sum();sample[negative]*=(1-profile['positive_mass'])/sample[negative].sum();assert np.all(sample>0) and abs(sample.sum()-1)<1e-10
            z0=tr['x'].astype(float)@w.astype(float).T+b;parent_prob=probability(tr['x'],w,b);rival=z0.copy();rival[:,label]=-np.inf;maximum=rival.max(1);lse=maximum+np.log(np.exp(rival-maximum[:,None]).sum(1))
            reference=np.where(tr['y']==label,maximum,z0[np.arange(1600),tr['y']]);sign=np.where(tr['y']==label,1.,-1.)
            other=z0.copy();other[np.arange(1600),tr['y']]=-np.inf;keep=((parent_prob.argmax(1)==tr['y'])&(z0[np.arange(1600),tr['y']]-other.max(1)>.5)).astype(float)
            q=1/(1+np.exp(-np.clip((z0[:,label]-lse)/c['temperature'],-60,60)))
            u0=torch.tensor(np.r_[w[label],b[label]],dtype=torch.float64);u=u0.clone();fixedw,fixedb=w.copy(),b.copy();plans.append(dict(pass_index=passes,class_index=label,TRAIN_weights=sample.tolist(),all_TRAIN_records_used=1600))
            args=[torch.from_numpy(v) for v in (reference,sign,lse,q,keep)]
            for _ in range(c['steps_each_class']):
                step,_,_=direction(u,u0,xx,*args,c,torch.from_numpy(sample),binary);assert torch.isfinite(step).all();step*=min(1.,c['trust_step']/max(float(step.norm()),1e-12));u-=step
                offset=u-u0;u=u0+offset*min(1.,c['row_radius']/max(float(offset.norm()),1e-12))
                ww,bb=fixedw.copy(),fixedb.copy();ww[label]=u[:-1].numpy().astype(np.float32);bb[label]=float(u[-1]);lastw,lastb=ww.copy(),bb.copy();p=probability(te['x'],ww,bb);n,nll=score(p,te['y']);steps+=1
                if improves(n,nll,best,bnll,c['minimum_NLL_gain']):best,bnll,selected=n,nll,steps;w,b,bestw,bestb=ww.copy(),bb.copy(),ww.copy(),bb.copy()
                history.append(dict(step=steps,pass_index=passes,class_index=label,TRAIN_gradient_records=1600,TEST_correct=n,selected_correct=best,selected_NLL=bnll))
                if best>=target or time.monotonic()-start>=c['maximum_fit_seconds']:break
            m.save(a.out/'progress.json',dict(state='full_TRAIN_row_curvature',selected_correct=best,actual_steps=steps,pass_index=passes,elapsed_fit_s=time.monotonic()-start))
            if best>=target or time.monotonic()-start>=c['maximum_fit_seconds']:break
        if best>=target or time.monotonic()-start>=c['maximum_fit_seconds'] or not improves(best,bnll,*prior_best,c['minimum_NLL_gain']):break
    fit_elapsed=time.monotonic()-start;m.save(a.out/'history.json',history);m.save(a.out/'plans.json',plans);results={}
    for name,ww,bb in (('best',bestw,bestb),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);model.load_state_dict(state);validate_state(m,model,original,{'linear.weight','linear.bias'})
        pt=a.out/(name+'.pt');saved_model(m,model,pt,protocol);metric,p=original_replay(model,data,fields,'test');expected=probability(te['x'],ww,bb);error=float(np.abs(p-expected).max());assert error<=3e-6 and np.array_equal(p.argmax(1),expected.argmax(1))
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p)
        train=m.metrics(tr['y'],probability(tr['x'],ww,bb));results[name]=dict(TEST=metric,TRAIN=train,PT_SHA256=m.sha(pt),full_CPU_replay_maxabs=error,all_nonhead_frozen=True)
    assert results['best']['TEST']['correct']==best
    m.save(a.out/'report.json',dict(state='complete',selected_correct=best,selected_e0=selected==0,selected_step=selected,actual_steps=steps,actual_passes=passes,target=target,passed=best>=target,results=results,baseline_correct=score(te['p'],te['y'])[0],full_TRAIN_each_update=True,fit_elapsed_s=fit_elapsed,TEST_gradient=False,SDK_used=False,GPU_used=False,full_physical=False,protocol_SHA256=m.sha(a.out/'protocol.json')))
    print(m.json.dumps(dict(complete=True,selected_correct=best,selected_e0=selected==0,target=target,passed=best>=target)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--round',type=int,choices=(1,2));p.add_argument('--out',type=Path);a=p.parse_args();selftest() if a.selftest else run(a)
