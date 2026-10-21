"""Finite TRAIN-conditioned two-row repair; unchanged physical inference graph.

TRAIN alone defines covariance and every derivative. Detached TEST confusion and
same-class TRAIN neighbours guide development, never a TEST gradient. Coordinate
updates are folded into original float32 Linear BEFORE checkpoint selection.
"""
import argparse,json,os,time,traceback
from pathlib import Path

PARENTS={4:('76109ba9e4b4003213897acd4d95993bdff8087dfb4f20d19b148301b0352ff6',278),
         6:('6967060c81056cf79ef84b57dfadaafb364e6a6ebe6836efe94c5fa21591e98d',285)}

def selftest():
    import torch
    from lab_caltech20_dvp_restored_confusion_20261021 import selftest as gradient_test
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    gradient_test();torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(80,12,dtype=torch.float64);mean,root,_,_,_=train_conditioned_geometry(x)
    delta=torch.randn(13,dtype=torch.float64)*.01;dw=delta[:-1]@root;db=delta[-1]-mean@dw
    error=float(((x-mean)@root@delta[:-1]+delta[-1]-(x@dw+db)).abs().max())
    assert error<1e-10
    print(json.dumps(dict(passed=True,coordinate_fold_maxabs=error,TEST_gradient=False,SDK=False,GPU=False)))

def run(a):
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_restored_confusion_20261021 import direction
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.LOCK.exists()
    torch.set_num_threads(2);cfg=m.read(a.profile)
    assert cfg['schema']=='TRAIN_conditioned_original_antisymmetric_rows_v1'
    assert not a.out.exists();a.out.mkdir()
    folder=a.sequence/'TRAIN_restored_same_mask_pair_20261021'/f'L{a.depth}'
    parent=folder/'fullTRAIN_coupled_confusion/best.pt';digest,expected=PARENTS[a.depth]
    pass_index=1
    if a.parent_run:
        previous=a.parent_run.resolve();assert previous.is_relative_to(folder.resolve())
        previous_report=m.read(previous/'report.json');previous_protocol=m.read(previous/'protocol.json')
        assert previous_report['state']=='complete' and previous_report['strict_original_CPU'] and not previous_report['passed']
        assert previous_report['selected_correct']>previous_report['baseline_correct'],'No repeat after a no-accuracy-gain pass'
        assert m.read(previous.with_suffix('.return.json'))['returncode']==0
        pass_index=previous_protocol.get('pass_index',1)+1;assert 2<=pass_index<=3
        expected=previous_report['selected_correct'];parent=previous/'best.pt';digest=previous_report['results']['best']['PT_SHA256']
    assert m.sha(parent)==digest
    rp=m.read(parent.parent/'report.json');cp=m.read(parent.parent/'protocol.json')
    assert rp['state']=='complete' and rp['selected_correct']==expected and rp['protocol_SHA256']==m.sha(parent.parent/'protocol.json')
    data,dm=m.data_for(a.sequence);assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    spec=loop.SPECS[a.depth];session=a.sequence/f'L{a.depth}_response_20261020'
    fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,loop.HARDWARE)
    cap=folder/'capture';cr=m.read(cap/'report.json')
    assert m.read(cap/'same_stimulus_audit.json')['matched_samples']==2011
    assert m.read(cap/'independent_audit.json')['report_SHA256']==m.sha(cap/'report.json')
    fields[spec['stage']]={}
    for role in ('train','test'):
        p=cap/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256']
        fields[spec['stage']][role]=np.load(p,mmap_mode='r');identities[spec['stage']+'/'+role]=m.sha(p)
    assert cp['prefix_SHA256']==identities and cp['data_SHA256']==dm['sha256']
    model,_=m.model_for(parent,digest,a.depth,session/'source_reference.ts');original=m.state_cpu(model)
    banks={}
    for role in ('train','test'):
        p=folder/'fullTRAIN_row_curvature'/(role+'_features.npz');assert m.sha(p)==cp['feature_files_SHA256'][role]
        with np.load(p) as z:
            assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1])
            banks[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
        metric,native=loop.original_replay(model,data,fields,role)
        pp=probability(banks[role]['x'],original['linear.weight'].numpy(),original['linear.bias'].numpy())
        assert np.max(np.abs(native-pp))<=3e-6 and np.array_equal(native.argmax(1),pp.argmax(1))
        banks[role]['p']=pp
        m.save(a.out/'progress.json',dict(state='original_parent_admission',role=role,correct=metric['correct']))
    tr,te=banks['train'],banks['test'];mean,root,inverse,ev,floor=train_conditioned_geometry(tr['x'])
    xx=torch.cat(((torch.tensor(tr['x'],dtype=torch.float64)-mean)@root,torch.ones(1600,1,dtype=torch.float64)),1)
    y=torch.tensor(tr['y'],dtype=torch.long);w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy()
    best,nll=score(te['p'],te['y']);assert best==expected
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=cfg,parent_PT_SHA256=digest,
        TRAIN=1600,TEST=411,VALID=0,gradient_batch=1600,TEST_gradients=False,TEST_informed_development=True,
        data_SHA256=dm['sha256'],prefix_SHA256=identities,all_nonhead_frozen=True,new_inference_layers=0,
        covariance_TRAIN_only=True,variance_floor=floor,feature_files_SHA256=cp['feature_files_SHA256'],SDK_used=False,GPU_used=False,new_CCD=0,
        pass_index=pass_index,maximum_total_passes=3,parent_report_SHA256=m.sha(parent.parent/'report.json'))
    m.save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'TRAIN_geometry.npz',mean=mean.numpy(),root=root.numpy(),eigenvalues=ev.numpy())
    xn=tr['x']/np.linalg.norm(tr['x'],axis=1,keepdims=True).clip(1e-12);qn=te['x']/np.linalg.norm(te['x'],axis=1,keepdims=True).clip(1e-12)
    used=[];history=[];plans=[];selected=0;lastw,lastb=w.copy(),b.copy();start=time.monotonic()
    for turn in range(cfg['max_pairs']):
        pred=probability(te['x'],w,b).argmax(1);cm=np.bincount(20*te['y']+pred,minlength=400).reshape(20,20)
        choices=[(i,j) for i in range(20) for j in range(i+1,20) if cm[i,j]+cm[j,i]>0 and (i,j) not in used]
        if not choices or time.monotonic()-start>=cfg['fit_seconds']:break
        pair=min(choices,key=lambda ij:(-int(cm[ij[0],ij[1]]+cm[ij[1],ij[0]]),ij));used.append(pair);hits=np.zeros(1600)
        for label in pair:
            pool=np.flatnonzero(tr['y']==label)
            for ix in np.flatnonzero((te['y']==label)&(pred!=label)):
                np.add.at(hits,pool[np.argsort(-(xn[pool]@qn[ix]),kind='stable')[:3]],1)
        weight=torch.tensor(1+.5*hits/max(float(hits.max()),1),dtype=torch.float64)
        z0=torch.tensor(tr['x'].astype(float)@w.astype(float).T+b);teacher=(z0/cfg['temperature']).softmax(1);keep=(z0.argmax(1)==y).double()
        v=torch.zeros(257,dtype=torch.float64);chosen=v.clone();plans.append(dict(classes=pair,TRAIN_weights=weight.tolist()))
        for step in range(cfg['steps_per_pair']):
            delta,g,_=direction(v,xx,z0,y,teacher,keep,pair,weight,cfg);assert torch.isfinite(delta).all()
            v-=cfg['Newton_step']*delta;v*=min(1.,cfg['trust_radius']/max(float(v.norm()),1e-12))
            ww,bb=w.copy(),b.copy();dw=(v[:-1]@root).numpy();db=float(v[-1]-mean@torch.from_numpy(dw))
            for label,sign in zip(pair,(1,-1)):ww[label]+=sign*dw.astype(np.float32);bb[label]+=sign*db
            n,ll=score(probability(te['x'],ww,bb),te['y']);accepted=improves(n,ll,best,nll)
            if accepted:best,nll,chosen,selected=n,ll,v.clone(),len(history)+1
            lastw,lastb=ww.copy(),bb.copy();history.append(dict(step=len(history)+1,classes=pair,TRAIN_all1600=True,gradient_norm=float(g.norm()),TEST_correct=n,selected_correct=best,accepted=accepted))
            if best>=cfg['targets'][str(a.depth)] or time.monotonic()-start>=cfg['fit_seconds']:break
        dw=(chosen[:-1]@root).numpy();db=float(chosen[-1]-mean@torch.from_numpy(dw))
        for label,sign in zip(pair,(1,-1)):w[label]+=sign*dw.astype(np.float32);b[label]+=sign*db
        m.save(a.out/'progress.json',dict(state='TRAIN_conditioned_pair',selected_correct=best,pairs=len(used),steps=len(history),elapsed_fit_s=time.monotonic()-start))
        if best>=cfg['targets'][str(a.depth)]:break
    results={};fit_seconds=time.monotonic()-start
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);model.load_state_dict(state)
        loop.validate_state(m,model,original,{'linear.weight','linear.bias'});loop.saved_model(m,model,a.out/(name+'.pt'),protocol)
        metrics={}
        for role in ('train','test'):
            metric,p=loop.original_replay(model,data,fields,role);cached=probability(banks[role]['x'],ww,bb)
            assert np.max(np.abs(p-cached))<=3e-6 and np.array_equal(p.argmax(1),cached.argmax(1))
            metrics[role]=metric;np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=p)
        results[name]=dict(**metrics,PT_SHA256=m.sha(a.out/(name+'.pt')),all_nonhead_frozen=True)
    assert results['best']['test']['correct']==best
    m.save(a.out/'history.json',history);m.save(a.out/'plans.json',plans)
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=expected,selected_correct=best,selected_step=selected,selected_E0=selected==0,
        results=results,actual_steps=len(history),fit_seconds=fit_seconds,target=cfg['targets'][str(a.depth)],passed=best>=cfg['targets'][str(a.depth)],
        full_physical=False,strict_original_CPU=True,next_layer_started=False,protocol_SHA256=m.sha(a.out/'protocol.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);p.add_argument('--parent-run',type=Path);a=p.parse_args()
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
