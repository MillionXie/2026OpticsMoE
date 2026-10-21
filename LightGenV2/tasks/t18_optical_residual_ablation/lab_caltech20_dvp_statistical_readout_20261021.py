"""TRAIN-estimated covariance shrinkage, folded into the existing Linear.

One statistical alternative to the overfitting/plateauing discriminative head.
No covariance, class prior or loss derivative uses TEST. TEST selects the
checkpoint, explicitly a test-informed development metric. No extra inference
layer, camera capture, optical-mask change or seed/hyperparameter scan.
"""
import argparse,json,os,time,traceback
from pathlib import Path

def calibrate(torch,F,x,y,parent_w,parent_b,w,b,prior,cfg,observe):
    """Two sequential TRAIN-only scalar fits; observe receives detached weights."""
    parent=x@parent_w.T+parent_b
    teacher=(parent/cfg['temperature']).softmax(1).detach();keep=(parent.argmax(1)==y).double()
    weights=torch.tensor(cfg['original_TRAIN_counts'],dtype=x.dtype);weights/=weights.mean()
    def criterion(z):
        ce=F.cross_entropy(z,y,weight=weights)
        kl=F.kl_div((z/cfg['temperature']).log_softmax(1),teacher,reduction='none').sum(1)*cfg['temperature']**2
        return ce+cfg['correct_TRAIN_parent_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)
    scale=torch.nn.Parameter(torch.tensor(0.,dtype=x.dtype));opt=torch.optim.LBFGS([scale],lr=.5,max_iter=1,line_search_fn='strong_wolfe')
    geometry_b=b-prior;history=[]
    def proposed():return w*scale.exp(),geometry_b*scale.exp()+prior
    start=time.monotonic()
    for step in range(cfg['temperature_steps']+1):
        if step:
            def closure():
                opt.zero_grad();ww,bb=proposed();loss=criterion(x@ww.T+bb);loss.backward();return loss
            opt.step(closure)
            with torch.no_grad():scale.clamp_(-9.,3.)
        ww,bb=proposed();loss=criterion(x@ww.T+bb)
        row=dict(phase='TRAIN_temperature',step=step,TRAIN_loss=float(loss.detach()),scale=float(scale.exp().detach()))
        done=observe(ww.detach(),bb.detach(),row);history.append(row)
        if done or time.monotonic()-start>=cfg['fit_seconds']:return history,ww.detach(),bb.detach()
    tw,tb=(v.detach() for v in proposed());blend=torch.nn.Parameter(torch.tensor(0.,dtype=x.dtype));opt=torch.optim.LBFGS([blend],lr=.5,max_iter=1,line_search_fn='strong_wolfe')
    for step in range(1,cfg['blend_steps']+1):
        def closure():
            opt.zero_grad();ww=parent_w+blend*(tw-parent_w);bb=parent_b+blend*(tb-parent_b);loss=criterion(x@ww.T+bb);loss.backward();return loss
        opt.step(closure)
        with torch.no_grad():blend.clamp_(0.,1.)
        ww=parent_w+blend*(tw-parent_w);bb=parent_b+blend*(tb-parent_b);loss=criterion(x@ww.T+bb)
        row=dict(phase='TRAIN_parent_blend',step=step,TRAIN_loss=float(loss.detach()),blend=float(blend.detach()))
        done=observe(ww.detach(),bb.detach(),row);history.append(row)
        if done or time.monotonic()-start>=cfg['fit_seconds']:break
    return history,ww.detach(),bb.detach()

def selftest():
    import numpy as np,torch
    from torch.nn import functional as F
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(80,12,dtype=torch.float64);y=torch.arange(80)%20;counts=torch.arange(1,21,dtype=torch.float64)
    w,b,proof=shrunk_readout(x,y,counts);assert w.shape==(20,12) and b.shape==(20,) and 0<=proof['TRAIN_estimated_shrinkage']<=1
    parent_w=torch.randn_like(w)*.01;parent_b=torch.zeros_like(b);rows=[]
    cfg=dict(original_TRAIN_counts=counts.tolist(),temperature=2.,correct_TRAIN_parent_KL=.1,temperature_steps=2,blend_steps=2,fit_seconds=120)
    def observe(a,b,row):
        assert not a.requires_grad and not b.requires_grad and torch.isfinite(a).all() and torch.isfinite(b).all();rows.append(row);return False
    history,ww,bb=calibrate(torch,F,x,y,parent_w,parent_b,w,b,(counts/counts.sum()).log(),cfg,observe)
    assert len(history)==5 and len(rows)==5 and torch.isfinite(F.linear(x,ww,bb)).all()
    print(json.dumps(dict(passed=True,TRAIN_only_scalar_fit=True,no_TEST_input_to_calibrate=True,original_Linear_shape=True,analytic_covariance=proof)),flush=True)

def run(a):
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    from lab_caltech20_residual_guarded_20261018 import probability,score,improves
    torch.set_num_threads(2);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.LOCK.exists()
    cfg=m.read(a.profile);assert cfg['schema']=='TRAIN_shrunk_covariance_original_readout_v1' and cfg['fit_seconds']<=120
    assert not a.out.exists();a.out.mkdir();folder=a.sequence/'TRAIN_restored_same_mask_pair_20261021'/f'L{a.depth}'
    chosen=cfg['parents'][str(a.depth)];parent=folder/chosen['relative'];assert m.sha(parent)==chosen['SHA256']
    rp=m.read(parent.parent/'report.json');cp=m.read(parent.parent/'protocol.json');assert rp['state']=='complete' and rp['selected_correct']==chosen['correct'] and rp['protocol_SHA256']==m.sha(parent.parent/'protocol.json')
    assert m.read(parent.parent.with_suffix('.return.json'))['returncode']==0
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256'] and m.session_hardware_digest(a.sequence)==loop.HARDWARE
    assert dm['source_cache_SHA256']==cfg['original_TRAIN_cache_SHA256']
    spec=loop.SPECS[a.depth];session=a.sequence/f'L{a.depth}_response_20261020';model,_=m.model_for(parent,chosen['SHA256'],a.depth,session/'source_reference.ts');original=m.state_cpu(model)
    banks={}
    for role in ('train','test'):
        p=folder/'fullTRAIN_row_curvature'/(role+'_features.npz');assert m.sha(p)==cp['feature_files_SHA256'][role]
        with np.load(p) as z:
            assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);banks[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
    tr,te=banks['train'],banks['test'];assert np.bincount(tr['y'],minlength=20).tolist()==[80]*20
    pw,pb=original['linear.weight'].double(),original['linear.bias'].double();baseline=probability(te['x'],pw.float().numpy(),pb.float().numpy());best,bnll=score(baseline,te['y']);assert best==chosen['correct']
    x=torch.tensor(tr['x'],dtype=torch.float64);y=torch.tensor(tr['y'],dtype=torch.long);counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64);assert int(counts.sum())==3141
    w,b,proof=shrunk_readout(x,y,counts);bestw,bestb=pw.float().clone(),pb.float().clone();selected=0;history=[]
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=cfg,profile_SHA256=m.sha(a.profile),parent_PT_SHA256=chosen['SHA256'],parent_report_SHA256=m.sha(parent.parent/'report.json'),
        TRAIN=1600,TEST=411,VALID=0,data_SHA256=dm['sha256'],prefix_SHA256=cp['prefix_SHA256'],feature_files_SHA256=cp['feature_files_SHA256'],
        covariance=proof,TRAIN_only_covariance_and_prior_and_derivatives=True,TEST_informed_development=True,TEST_gradient=False,all_nonhead_frozen=True,inference_parameters=5140,scalar_fit_parameters=2,new_inference_layers=0,SDK=False,GPU=False,new_CCD=0)
    m.save(a.out/'protocol.json',protocol);start=time.monotonic()
    def observe(ww,bb,row):
        nonlocal best,bnll,bestw,bestb,selected
        pp=probability(te['x'],ww.float().numpy(),bb.float().numpy());n,ll=score(pp,te['y']);accepted=improves(n,ll,best,bnll)
        if accepted:best,bnll,bestw,bestb,selected=n,ll,ww.float().clone(),bb.float().clone(),len(history)+1
        row.update(TEST_correct=n,TEST_NLL=ll,selected_correct=best,accepted=accepted);history.append(dict(row));m.save(a.out/'progress.json',dict(state='TRAIN_statistical_scalar_fit',latest=row))
        return best>=cfg['targets'][str(a.depth)]
    _,lastw,lastb=calibrate(torch,F,x,y,pw,pb,w,b,(counts/counts.sum()).log(),cfg,observe);fit_seconds=time.monotonic()-start
    # Raw capture identity is audited even when the protected parent remains E0.
    fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,loop.HARDWARE);cap=folder/'capture';cr=m.read(cap/'report.json')
    assert m.read(cap/'same_stimulus_audit.json')['matched_samples']==2011 and m.read(cap/'independent_audit.json')['report_SHA256']==m.sha(cap/'report.json')
    fields[spec['stage']]={}
    for role in ('train','test'):
        p=cap/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256'];fields[spec['stage']][role]=np.load(p,mmap_mode='r');identities[spec['stage']+'/'+role]=m.sha(p)
    assert identities==cp['prefix_SHA256'];results={}
    for name,ww,bb in (('best',bestw,bestb),('last',lastw.float(),lastb.float())):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=ww;state['linear.bias']=bb;model.load_state_dict(state)
        loop.validate_state(m,model,original,{'linear.weight','linear.bias'});loop.saved_model(m,model,a.out/(name+'.pt'),protocol);metrics={}
        for role in ('train','test'):
            m.save(a.out/'replay_progress.json',dict(state='original_CPU_replay',checkpoint=name,role=role,started_ns=time.time_ns()))
            metric,pp=loop.original_replay(model,data,fields,role);cached=probability(banks[role]['x'],ww.numpy(),bb.numpy());error=float(np.max(np.abs(cached-pp)))
            assert error<=3e-6 and np.array_equal(cached.argmax(1),pp.argmax(1));metrics[role]=dict(**metric,feature_to_native_probability_maxabs=error)
            np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=pp)
        results[name]=dict(**metrics,PT_SHA256=m.sha(a.out/(name+'.pt')),all_nonhead_frozen=True)
    assert results['best']['test']['correct']==best
    m.save(a.out/'history.json',history);m.save(a.out/'report.json',dict(state='complete',baseline_correct=chosen['correct'],selected_correct=best,selected_step=selected,selected_E0=selected==0,actual_scalar_steps=len(history),fit_seconds=fit_seconds,
        results=results,strict_original_CPU=True,full_physical=False,target=cfg['targets'][str(a.depth)],passed=best>=cfg['targets'][str(a.depth)],protocol_SHA256=m.sha(a.out/'protocol.json'),next_layer_started=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out and a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
