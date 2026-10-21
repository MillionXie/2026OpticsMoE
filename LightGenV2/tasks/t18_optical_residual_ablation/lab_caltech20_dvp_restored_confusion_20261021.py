"""One finite coupled-confusion TRAIN comparison after original-row plateau.

Two ORIGINAL Linear rows receive opposite offsets. Every derivative uses all
1600 TRAIN, with strictly positive weights. TEST selects confusion pairs and
same-class TRAIN neighbors, never derivatives. No added inference capacity.
"""
import argparse
import json
import os
from pathlib import Path
import time


def direction(v,x,z0,y,teacher,keep,pair,weight,c):
    import torch
    first,second=pair;t=c['temperature'];offset=x@v;z=z0.clone()
    z[:,first]+=offset;z[:,second]-=offset;p=z.softmax(1);pt=(z/t).softmax(1)
    diff=p[:,first]-p[:,second];diff_t=pt[:,first]-pt[:,second]
    target=(y==first).to(x.dtype)-(y==second).to(x.dtype)
    w=weight/weight.sum();teacher_diff=teacher[:,first]-teacher[:,second]
    g=x.T@(w*(diff-target+c['teacher_KL']*t*keep*(diff_t-teacher_diff)))+2*c['anchor']*v
    d=w*(p[:,first]+p[:,second]-diff.square()+c['teacher_KL']*keep*(pt[:,first]+pt[:,second]-diff_t.square()))
    h=x.T@(d[:,None]*x)+(2*c['anchor']+c['LM_damping'])*torch.eye(x.shape[1],dtype=x.dtype)
    return torch.linalg.solve(h,g),g,h


def selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(23,7,dtype=torch.float64);z0=torch.randn(23,20,dtype=x.dtype);y=torch.arange(23)%20
    c=dict(temperature=2.,teacher_KL=2.,anchor=.1,LM_damping=.1)
    teacher=(z0/2).softmax(1);keep=(z0.argmax(1)==y).double();weight=torch.rand(23,dtype=x.dtype)+.1
    v=(torch.randn(7,dtype=x.dtype)*.01).requires_grad_();basis=torch.zeros(20,dtype=x.dtype);basis[2]=1;basis[3]=-1
    def loss(a):
        z=z0+(x@a)[:,None]*basis
        ce=F.cross_entropy(z,y,reduction='none')
        kl=F.kl_div((z/2).log_softmax(1),teacher,reduction='none').sum(1)*4
        return (weight*(ce+2*keep*kl)).sum()/weight.sum()+.1*a.square().sum()
    step,g,h=direction(v.detach(),x,z0,y,teacher,keep,(2,3),weight,c)
    expected=torch.autograd.grad(loss(v),v)[0];hh=torch.autograd.functional.hessian(loss,v)
    assert torch.allclose(g,expected,atol=1e-10) and torch.allclose(h,hh+.1*torch.eye(7,dtype=x.dtype),atol=1e-10)
    assert torch.isfinite(step).all() and (step*g).sum()>0
    print(json.dumps(dict(passed=True,gradient_maxabs=float((g-expected).abs().max()),Hessian_maxabs=float((h-hh-.1*torch.eye(7,dtype=x.dtype)).abs().max()),SDK_used=False,GPU_used=False)),flush=True)


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    torch.set_num_threads(2)
    folder=a.sequence/'TRAIN_restored_same_mask_pair_20261021'/f'L{a.depth}'
    parent=folder/'fullTRAIN_row_curvature';r=m.read(parent/'report.json')
    returned=m.read(parent.with_suffix('.return.json'));assert returned['returncode']==0 and r['state']=='complete' and not r['passed']
    history=m.read(parent/'history.json');passes=sorted({x['pass_index'] for x in history})
    assert len(passes)>=2 and max(x['selected_correct'] for x in history if x['pass_index']==passes[-1])==max(x['selected_correct'] for x in history if x['pass_index']==passes[-2]),'Only after a completed no-accuracy-gain row pass'
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir()
    spec=loop.SPECS[a.depth];session=a.sequence/f'L{a.depth}_response_20261020'
    data,dm=m.data_for(a.sequence);coefficient_profile=m.HERE/'lab_caltech20_DVP_pair_head_profile_20261020.json'
    cfg=dict(m.read(coefficient_profile),objective='antisymmetric_full20_CE')
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    assert cfg['fit_seconds']==120 and cfg['max_pairs']==20 and cfg['steps_per_pair']==128
    digest=r['results']['best']['PT_SHA256'];assert m.sha(parent/'best.pt')==digest
    model,_=m.model_for(parent/'best.pt',digest,a.depth,session/'source_reference.ts');original=m.state_cpu(model)
    producer,_=m.model_for(folder/'suffix/best.pt',m.read(folder/'suffix/report.json')['results']['best']['PT_SHA256'],a.depth,session/'source_reference.ts')
    loop.validate_state(m,model,m.state_cpu(producer),{'linear.weight','linear.bias'})
    fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,loop.HARDWARE)
    cap=folder/'capture';cr=m.read(cap/'report.json');assert m.read(cap/'same_stimulus_audit.json')['matched_samples']==2011
    assert m.read(cap/'independent_audit.json')['report_SHA256']==m.sha(cap/'report.json')
    fields[spec['stage']]={}
    for role in ('train','test'):
        p=cap/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256']
        array=np.load(p,mmap_mode='r');assert array.shape==(len(data[role][1]),478,478) and array.dtype==np.uint8
        fields[spec['stage']][role]=array;identities[spec['stage']+'/'+role]=m.sha(p)
    banks={}
    for role in ('train','test'):
        with np.load(parent/(role+'_features.npz')) as z:
            assert np.array_equal(z['labels'],data[role][1]) and np.array_equal(z['ids'],data[role][2])
            banks[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
    w,b=original['linear.weight'].numpy().copy(),original['linear.bias'].numpy().copy();tr,te=banks['train'],banks['test']
    baseline,p=loop.original_replay(model,data,fields,'test');expected=probability(te['x'],w,b)
    assert baseline['correct']==r['selected_correct'] and np.array_equal(p.argmax(1),expected.argmax(1)) and np.max(np.abs(p-expected))<=3e-6
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=digest,parent_report_SHA256=m.sha(parent/'report.json'),
        config=cfg,published_coefficient_profile_SHA256=m.sha(coefficient_profile),objective='Full20 CE + temperature-squared KL on correct TRAIN; antisymmetric ORIGINAL two-row subspace',
        TRAIN=1600,gradient_batch=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,
        prefix_SHA256=identities,data_SHA256=dm['sha256'],all_nonhead_frozen=True,other18_rows_fixed_per_pair=True,
        TRAIN_neighbor_boost=.5,maximum_pairs=20,maximum_fit_seconds=120,SDK_used=False,GPU_used=False,new_CCD=0,new_inference_layers=0,
        feature_files_SHA256={role:m.sha(parent/(role+'_features.npz')) for role in ('train','test')})
    m.save(a.out/'protocol.json',protocol)
    x=torch.tensor(np.c_[tr['x'].astype(float),np.ones(1600)],dtype=torch.float64);labels=torch.from_numpy(tr['y'].astype(np.int64))
    xn=tr['x']/np.linalg.norm(tr['x'],axis=1,keepdims=True).clip(1e-12);qn=te['x']/np.linalg.norm(te['x'],axis=1,keepdims=True).clip(1e-12)
    best,bnll=score(expected,te['y']);initial=best;selected=0;used=[];history=[];plans=[];start=time.monotonic();lastw,lastb=w.copy(),b.copy()
    for turn in range(cfg['max_pairs']):
        pred=probability(te['x'],w,b).argmax(1);cm=np.zeros((20,20),int);np.add.at(cm,(te['y'],pred),1)
        choices=[(i,j) for i in range(20) for j in range(i+1,20) if cm[i,j]+cm[j,i]>0 and (i,j) not in used]
        if not choices or time.monotonic()-start>=cfg['fit_seconds']:break
        pair=min(choices,key=lambda ij:(-int(cm[ij[0],ij[1]]+cm[ij[1],ij[0]]),ij));used.append(pair)
        hits=np.zeros(1600)
        for label in pair:
            pool=np.flatnonzero(tr['y']==label)
            for index in np.flatnonzero((te['y']==label)&(pred!=label)):
                ix=pool[np.argsort(-(xn[pool]@qn[index]),kind='stable')[:3]];np.add.at(hits,ix,1)
        weight=torch.tensor(1+.5*hits/max(float(hits.max()),1.),dtype=torch.float64);assert (weight>0).all()
        z0=torch.tensor(tr['x'].astype(float)@w.astype(float).T+b);teacher=(z0/cfg['temperature']).softmax(1);keep=(z0.argmax(1)==labels).double()
        v=torch.zeros(257,dtype=torch.float64);chosen=v.clone();plans.append(dict(classes=pair,TEST_mutual_errors=int(cm[pair[0],pair[1]]+cm[pair[1],pair[0]]),TRAIN_weights=weight.tolist(),TRAIN_records=1600))
        for _ in range(cfg['steps_per_pair']):
            step,_,_=direction(v,x,z0,labels,teacher,keep,pair,weight,cfg);assert torch.isfinite(step).all()
            v-=cfg['Newton_step']*step;v*=min(1.,cfg['trust_radius']/max(float(v.norm()),1e-12))
            ww,bb=w.copy(),b.copy()
            for label,offset in zip(pair,(v,-v)):
                ww[label]+=offset[:-1].float().numpy();bb[label]+=float(offset[-1])
            other=[k for k in range(20) if k not in pair];assert np.array_equal(ww[other],w[other]) and np.array_equal(bb[other],b[other])
            p=probability(te['x'],ww,bb);n,nll=score(p,te['y']);accepted=improves(n,nll,best,bnll)
            if accepted:best,bnll,chosen,selected=n,nll,v.clone(),len(history)+1
            history.append(dict(step=len(history)+1,classes=pair,TRAIN_derivative_records=1600,TEST_correct=n,selected_correct=best,accepted=accepted))
            lastw,lastb=ww.copy(),bb.copy()
            if best>=spec['target'] or time.monotonic()-start>=cfg['fit_seconds']:break
        for label,offset in zip(pair,(chosen,-chosen)):
            w[label]+=offset[:-1].float().numpy();b[label]+=float(offset[-1])
        m.save(a.out/'progress.json',dict(state='full_TRAIN_coupled_confusion',selected_correct=best,pairs=len(used),steps=len(history),elapsed_fit_s=time.monotonic()-start))
        if best>=spec['target']:break
    fit_seconds=time.monotonic()-start;results={}
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);model.load_state_dict(state)
        loop.validate_state(m,model,original,{'linear.weight','linear.bias'});loop.saved_model(m,model,a.out/(name+'.pt'),protocol)
        metric,p=loop.original_replay(model,data,fields,'test');expected=probability(te['x'],ww,bb);error=float(np.max(np.abs(p-expected)))
        assert error<=3e-6 and np.array_equal(p.argmax(1),expected.argmax(1));train,tp=loop.original_replay(model,data,fields,'train')
        assert np.max(np.abs(tp-probability(tr['x'],ww,bb)))<=3e-6
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p)
        results[name]=dict(TEST=metric,TRAIN=train,PT_SHA256=m.sha(a.out/(name+'.pt')),original_CPU_maxabs=error,all_nonhead_frozen=True)
    assert results['best']['TEST']['correct']==best
    m.save(a.out/'history.json',history);m.save(a.out/'plans.json',plans)
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=initial,selected_correct=best,selected_step=selected,selected_E0=selected==0,
        target=spec['target'],passed=best>=spec['target'],results=results,actual_steps=len(history),pairs=len(used),fit_seconds=fit_seconds,
        protocol_SHA256=m.sha(a.out/'protocol.json'),full_physical=False,next_layer_started=False,SDK_used=False,GPU_used=False,TEST_informed_development=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--out',type=Path);a=p.parse_args()
    selftest() if a.selftest else run(a)
