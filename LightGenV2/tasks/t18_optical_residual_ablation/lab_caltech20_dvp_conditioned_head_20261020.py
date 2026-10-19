"""TRAIN-conditioned optimization of the original DVP measured-prefix head.

No inference transform is added: optimizer coordinates fold back into the
existing 20x256 Linear. TRAIN alone determines covariance and class priors.
TEST is used for explicitly authorized development selection, never derivatives.
"""
import argparse
import copy
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
from lab_caltech20_residual_terminal_cpu_20261019 import TRAIN_spectrum_geometry, pair_direction, huber_row_direction, binary_row_direction, shrunk_readout
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


def cpu_fit_admission(a):
    """A complete frozen L4 head may fit while this run captures L6.

    No SDK import/open/close and no permission for same-depth incomplete fields.
    Field SHA/receipt validation still occurs in fields_for before computation.
    """
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    proof=dict(SDK_used=False,GPU_used=False,parallel_capture=False)
    if core.LOCK.exists():
        import psutil
        lock=core.read(core.LOCK)
        assert a.depth==4 and a.prefix==5
        assert lock.get('camera')=='DVP' and Path(lock['session']).resolve()==(a.sequence/'L6').resolve()
        args=psutil.Process(lock['pid']).cmdline()
        assert any(Path(v).name=='lab_caltech20_dvp_layerwise_20261019.py' for v in args)
        assert args[args.index('--mode')+1]=='capture' and args[args.index('--depth')+1]=='6'
        assert Path(args[args.index('--sequence')+1]).resolve()==a.sequence.resolve()
        reports={}
        for stage in ('router','0','1','2','3'):
            p=a.sequence/'L4/capture'/stage/'report.json';r=core.read(p)
            assert r['state']=='complete' and r['TRAIN']==1600 and r['TEST']==411 and r['sdk_released']
            audit=p.with_name('independent_capture_audit.json');assert core.read(audit)['passed']
            reports[stage]=dict(report_SHA256=core.sha(p),audit_SHA256=core.sha(audit),arrays=r['arrays'])
        proof.update(parallel_capture=True,capture_lock_snapshot=lock,complete_L4_capture_proofs=reports,
                     capture_process_create_time=psutil.Process(lock['pid']).create_time(),other_depth_only=True)
    a.cpu_fit_admission_proof=proof
    return proof


def original_alpha_state(original, allowed, value, key='fusion_logits.2'):
    """One declared post-E2 or final-L4 coordinate, not an alpha/seed grid."""
    assert key in ('fusion_logits.2','fusion_logits.3')
    assert key in allowed and value==.4001001
    assert original[key].dtype==torch.float64 and .4001<float(original[key])<=.95
    state={k:v.clone() for k,v in original.items()}
    state[key].fill_(value)
    assert all(v.dtype==state[k].dtype and torch.equal(v,state[k]) for k,v in original.items() if k!=key)
    return state


def terminal10_objective(logits, labels, weights, teacher, keep, conv, alpha, origin, cfg):
    """Original-head-fixed TRAIN objective; no TEST tensor enters derivatives."""
    ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=cfg['label_smoothing'])
    kl=F.kl_div(logits.log_softmax(1),teacher,reduction='none').sum(1)
    anchor=(conv-origin[0]).square().sum()+(alpha-origin[1]).square()
    return ce+cfg['teacher_KL']*(kl*keep).mean()+cfg['anchor']*anchor


def native_joint_objective(logits, labels, weights, teacher, keep, conv, alpha, delta, origin, cfg):
    """Published joint native-RMS loss, on TRAIN and original parameters only."""
    ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=cfg['label_smoothing'])
    temperature=cfg['temperature']
    kl=F.kl_div((logits/temperature).log_softmax(1),teacher,reduction='none').sum(1)
    anchor=(conv-origin[0]).square().sum()+(alpha-origin[1]).square()
    return ce+cfg['teacher_KL']*temperature**2*(kl*keep).mean()+cfg['terminal_anchor']*anchor+cfg['head_coordinate_anchor']*delta.square().sum()


def terminal10_fit(a):
    """DVP1600 adapter for published exact native-RMS terminal methods.

    Validate complete original forward against the cached original tail BEFORE
    optimization. Original10 parameters or original5150 with a folded head train;
    true RMS derivatives change backward, never inference or measured upstream.
    """
    import psutil
    from lab_caltech20_native_terminal_20261019 import native_features,folded
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    cpu_fit_admission(a);torch.set_num_threads(2);torch.manual_seed(73)
    assert a.depth==4 and a.prefix==5 and not a.feature_run and not a.out.exists()
    cfg=core.read(a.profile);assert cfg['epochs']<=60 and cfg['fit_seconds']<=600 and cfg['batch']==8
    joint=cfg.get('joint_original_head',False)
    assert joint==(a.mode=='native5150-fit')
    if joint:assert cfg['epochs']<=12 and cfg['fit_seconds']<=360 and not cfg.get('replay_first_batches_from')
    assert core.sha(HERE/'lab_caltech20_native_terminal_20261019.py')==cfg['native_tail_source_SHA256']
    data,dm=core.data_for(a.sequence);assert dm['source_cache_SHA256']==cfg['original_TRAIN_cache_SHA256']
    assert psutil.virtual_memory().available>sum(len(data[r][1]) for r in ('train','test'))*2*478*478*4+(2<<30)
    session=a.sequence/'L4';assert core.sha(a.checkpoint)==a.digest
    pr=core.read(a.checkpoint.parent/'report.json');pa=core.read(a.checkpoint.parent/'independent_audit.json')
    assert pa['passed'] and pa['report_SHA256']==core.sha(a.checkpoint.parent/'report.json') and pr['results']['best']['PT_SHA256']==a.digest
    model,_=core.model_for(a.checkpoint,a.digest,4,session/'source_reference.ts');original=core.state_cpu(model)
    fields,identities=core.fields_for(session,5,data,dm,core.session_hardware_digest(a.sequence))
    pp=core.read(a.checkpoint.parent/'protocol.json');assert pp['prefix']==5 and pp['prefix_SHA256']==identities and pp['data_SHA256']==dm['sha256']
    trainable={'electronic.3.local.weight','fusion_logits.3'};assert trainable<=core.allowed_keys(model,5)
    a.out.mkdir();core.save(a.out/'CPU_capture_admission.json',a.cpu_fit_admission_proof)
    cache={};cache_errors=[];held={};active={};counts={}
    pre=model.electronic[3].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()))
    def collect(module,args,output):
        role=active['role'];d=cache[role];begin=counts[role];n=min(len(output[0]),len(d['y'])-begin);end=begin+n
        x=held['x'].clone();o=model.nonlinear(model.readout['ccd'],'global').clone()
        f=native_features(model.electronic[3],model.fusion_logits[3],x,o)
        q=model.linear(f).softmax(1);ferr=float((f-model.readout['features']).abs().max());perr=float((q-output[0]).abs().max())
        cache_errors.append(max(ferr,perr));core.save(a.out/'progress.json',dict(state='native_terminal_cache_parity',role=role,completed=end,total=len(d['y']),feature_error=ferr,probability_error=perr))
        assert ferr<=3e-6 and perr<=3e-6,'Reject approximate terminal cache before any training'
        d['x'][begin:end].copy_(x[:n]);d['o'][begin:end].copy_(o[:n]);d['p'][begin:end]=output[0][:n].numpy();d['features'][begin:end]=f[:n].numpy();counts[role]=end
    hook=model.register_forward_hook(collect)
    try:
        for role in ('train','test'):
            n=len(data[role][1]);cache[role]=dict(x=torch.empty(n,478,478),o=torch.empty(n,478,478),y=torch.from_numpy(data[role][1].astype(np.int64)),p=np.empty((n,20),np.float32),features=np.empty((n,256),np.float32),ids=data[role][2]);counts[role]=0;active['role']=role
            metric,p=core.assess(model,data,fields,role);assert counts[role]==n and np.array_equal(p,cache[role]['p']);cache[role]['metric']=metric
    finally:hook.remove();pre.remove()
    assert max(cache_errors)<=3e-6 and cache['test']['metric']['correct']==pr['selected_correct']
    proof=dict(passed=True,TRAIN=1600,TEST=411,VALID=0,maxabs=max(cache_errors),original_forward_exact_before_training=True,original_head_frozen=not joint,cache_in_memory_only=True,no_new_CCD=True,native_tail_source_SHA256=cfg['native_tail_source_SHA256'])
    core.save(a.out/'terminal_cache_parity.json',proof)
    electronic=copy.deepcopy(model.electronic[3]);electronic.requires_grad_(True);head=copy.deepcopy(model.linear);head.requires_grad_(False)
    alpha=torch.nn.Parameter(original['fusion_logits.3'].clone());params=[*electronic.parameters(),alpha];assert sum(v.numel() for v in params)==10
    counts_prior=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float32);assert counts_prior.shape==(20,) and int(counts_prior.sum())==3141
    weights=counts_prior/counts_prior.mean();tr,te=cache['train'],cache['test'];teacher=torch.from_numpy(tr['p']);keep=teacher.argmax(1)==tr['y']
    delta=None
    if joint:
        trainable|={'linear.weight','linear.bias'};mean,root,_,ev,floor=train_conditioned_geometry(tr['features'])
        delta=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64));params.append(delta)
        assert sum(v.numel() for v in params)==5150
        teacher=teacher.clamp_min(1e-12).pow(1/cfg['temperature']);teacher/=teacher.sum(1,keepdim=True)
        np.savez_compressed(a.out/'TRAIN_geometry.npz',mean=mean.numpy(),root=root.numpy(),eigenvalues=ev.numpy(),floor=floor)
    def head_logits(features):
        if not joint:return head(features)
        w,b=folded(head.weight,head.bias,delta,mean,root);return F.linear(features,w,b)
    origin=(electronic.local.weight.detach().clone(),alpha.detach().clone());baseline=te['p'];best,bestloss=score(baseline,te['y'].numpy());selected=0
    protocol=dict(parent_PT_SHA256=a.digest,depth=4,prefix=5,allowed_keys=sorted(trainable),head_frozen=not joint,gradient_parameters=5150 if joint else 10,native_RMS_backward_only=True,
                  prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,config=cfg,profile_SHA256=core.sha(a.profile),
                  source_git=core.git_head(),source_SHA256=core.sha(__file__),native_tail_source_SHA256=cfg['native_tail_source_SHA256'],cache_parity=proof,
                  original_TRAIN_prior_only=True,correct_TRAIN_teacher_only=True,TEST_informed_development=True,TEST_gradient=False,target_correct=342,
                  CPU_threads=2,SDK_used=False,GPU_used=False,no_new_inference_layer=True)
    if joint:protocol.update(TRAIN_geometry_SHA256=core.sha(a.out/'TRAIN_geometry.npz'),head_coordinates_folded_into_original_5140=True)
    core.save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=te['ids'],labels=te['y'].numpy(),probabilities=baseline)
    def snapshot(name,epoch):
        state={k:v.clone() for k,v in original.items()};state['electronic.3.local.weight']=electronic.local.weight.detach().clone();state['fusion_logits.3']=alpha.detach().clone()
        if joint:
            w,b=folded(head.weight,head.bias,delta.detach(),mean,root);state['linear.weight']=w.detach().clone();state['linear.bias']=b.detach().clone()
        assert all(v.dtype==state[k].dtype and torch.equal(v,state[k]) for k,v in original.items() if k not in trainable)
        torch.save(dict(model=state,protocol=protocol,epoch=epoch),a.out/(name+'.pt'))
    def evaluate(role):
        d=cache[role];parts=[]
        with torch.no_grad():
            for begin in range(0,len(d['y']),8):
                x,o=d['x'][begin:begin+8],d['o'][begin:begin+8];n=len(x)
                if n<8:x=torch.cat([x,torch.zeros(8-n,478,478)]);o=torch.cat([o,torch.zeros(8-n,478,478)])
                parts.append(head_logits(native_features(electronic,alpha,x,o)).softmax(1)[:n].numpy())
        p=np.concatenate(parts);return p,score(p,d['y'].numpy())
    p,metric=evaluate('test');assert np.abs(p-baseline).max()<=3e-6 and np.array_equal(p.argmax(1),baseline.argmax(1))
    snapshot('best',0);opt=torch.optim.AdamW([dict(params=electronic.parameters(),lr=cfg['conv_lr'],weight_decay=cfg['weight_decay']),dict(params=[alpha],lr=cfg['alpha_lr'],weight_decay=0)])
    if joint:opt.add_param_group(dict(params=[delta],lr=cfg['head_coordinate_lr'],weight_decay=0))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=1e-5);history=[];start=time.monotonic();steps=0
    for epoch in range(1,cfg['epochs']+1):
        order=torch.randperm(1600);loss_sum=0.;used=0
        for begin in range(0,1600,8):
            ix=order[begin:begin+8];opt.zero_grad(set_to_none=True);z=head_logits(native_features(electronic,alpha,tr['x'][ix],tr['o'][ix]))
            loss=native_joint_objective(z,tr['y'][ix],weights,teacher[ix],keep[ix],electronic.local.weight,alpha,delta,origin,cfg) if joint else terminal10_objective(z,tr['y'][ix],weights,teacher[ix],keep[ix],electronic.local.weight,alpha,origin,cfg)
            assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            with torch.no_grad():
                alpha.clamp_(.4001001,.95)
                if joint:delta.mul_(min(1.,np.sqrt(20)*cfg['per_row_head_radius']/max(float(delta.norm()),1e-12)))
            loss_sum+=float(loss.detach())*len(ix);used+=len(ix);steps+=1
        scheduler.step();assert used==1600;p,(n,nll)=evaluate('test')
        if improves(n,nll,best,bestloss):best,bestloss,selected=n,nll,epoch;snapshot('best',epoch)
        row=dict(epoch=epoch,TRAIN_loss=loss_sum/1600,actual_TRAIN_samples=used,actual_TRAIN_steps=steps,TEST_correct=n,TEST_NLL=nll,selected_correct=best,selected_epoch=selected,alpha=float(alpha.detach()),elapsed_s=time.monotonic()-start)
        history.append(row);core.save(a.out/'history.json',history);core.save(a.out/'progress.json',dict(state='native_RMS_terminal5150_CPU' if joint else 'native_RMS_terminal10_CPU',**row));print(core.json.dumps(row),flush=True)
        if best>=342 or time.monotonic()-start>=cfg['fit_seconds']:break
    snapshot('last',epoch);results={}
    for name in ('best','last'):
        path=a.out/(name+'.pt');state=torch.load(path,map_location='cpu',weights_only=False)['model'];model.load_state_dict(state);electronic.load_state_dict(model.electronic[3].state_dict());alpha.data.copy_(state['fusion_logits.3'])
        if joint:head.load_state_dict(model.linear.state_dict());delta.data.zero_()
        assert all(v.dtype==state[k].dtype and torch.equal(v,state[k]) for k,v in original.items() if k not in trainable)
        native,p=core.assess(model,data,fields,'test');expected,_=evaluate('test');error=float(np.abs(p-expected).max())
        assert error<=3e-6 and np.array_equal(p.argmax(1),expected.argmax(1)),'Reject terminal result if original inference parity fails'
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'].numpy(),probabilities=p)
        results[name]=dict(TEST_development=native,PT_SHA256=core.sha(path),strict_CPU=True,native_tail_probability_maxabs=error,predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
    assert results['best']['TEST_development']['correct']==best and core.sha(a.checkpoint)==a.digest
    core.save(a.out/'report.json',dict(state='complete',baseline_TEST=te['metric'],results=results,selected_correct=best,selected_epoch=selected,selected_e0=selected==0,actual_epochs=epoch,
                  actual_TRAIN_steps=steps,target_correct=342,refinement_target_reached=best>=342,within8pp=best>=314,intermediate=False,head_frozen=not joint,gradient_parameters=5150 if joint else 10,
                  native_RMS_backward_only=True,frozen_parameters_unchanged=True,all_nonhead_frozen=False,SDK_used=False,GPU_used=False,TEST_gradients=False,elapsed_fit_s=time.monotonic()-start))


def admit_chain(a):
    """Package an audited multi-step optimizer chain, without another update.

    The workflow parent is the actual global incumbent. Component best/last
    roles and every intermediate digest remain explicit, not rewritten.
    """
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not core.LOCK.exists()
    assert a.depth==4 and a.prefix==4 and a.parent_run and not a.feature_run
    torch.set_num_threads(2);session=a.sequence/'L4';fits=session/'fits'
    source=a.checkpoint.parent;parent=a.parent_run
    assert source.resolve().is_relative_to(fits.resolve()) and parent.resolve().is_relative_to(fits.resolve())
    assert a.checkpoint.name=='best.pt' and core.sha(a.checkpoint)==a.digest and not a.out.exists()
    pr=core.read(parent/'report.json');pa=core.read(parent/'independent_audit.json')
    assert pr['state']=='complete' and pa['passed'] and pa['report_SHA256']==core.sha(parent/'report.json')
    assert pa['protocol_SHA256']==core.sha(parent/'protocol.json')
    parent_pt=parent/'best.pt';parent_digest=core.sha(parent_pt);assert parent_digest==pr['results']['best']['PT_SHA256']
    data,dm=core.data_for(a.sequence);fields,identities=core.fields_for(session,4,data,dm,core.session_hardware_digest(a.sequence))
    initial=torch.load(parent_pt,map_location='cpu',weights_only=False)['model']
    allowed={'linear.weight','linear.bias','fusion_logits.2'};nodes=[];current=source;role='best';digest=a.digest;seen=set()
    for _ in range(8):
        assert digest not in seen;seen.add(digest)
        r=core.read(current/'report.json');p=core.read(current/'protocol.json');audit=core.read(current/'independent_audit.json')
        assert r['state']=='complete' and r['frozen_parameters_unchanged'] and r['results'][role]['strict_CPU'] and audit['passed']
        assert audit['report_SHA256']==core.sha(current/'report.json') and audit['protocol_SHA256']==core.sha(current/'protocol.json')
        assert core.sha(current/(role+'.pt'))==r['results'][role]['PT_SHA256']==digest
        assert p['depth']==4 and p['prefix']==4 and p['prefix_SHA256']==identities and p['data_SHA256']==dm['sha256']
        assert set(p['allowed_keys'])<=allowed
        nodes.append(dict(fit=current.name,checkpoint_role=role,PT_SHA256=digest,parent_PT_SHA256=p['parent_PT_SHA256'],
                          report_SHA256=core.sha(current/'report.json'),protocol_SHA256=core.sha(current/'protocol.json'),
                          audit_SHA256=core.sha(current/'independent_audit.json')))
        if p['parent_PT_SHA256']==parent_digest:break
        matches=[]
        for folder in fits.iterdir():
            if folder.is_dir() and (folder/'report.json').exists():
                q=core.read(folder/'report.json')
                for candidate_role in ('best','last'):
                    if q.get('results',{}).get(candidate_role,{}).get('PT_SHA256')==p['parent_PT_SHA256']:
                        matches.append((folder,candidate_role))
        assert len(matches)==1;current,role=matches[0];digest=p['parent_PT_SHA256']
    else:raise AssertionError('Compound adaptation ancestry exceeds bounded8nodes')
    sr=core.read(source/'report.json');sp=core.read(source/'protocol.json')
    assert sr['selected_correct']>pr['selected_correct'] and not sr['selected_e0']
    assert core.sha(parent/'best_TEST_predictions.npz')==pr['results']['best']['predictions_SHA256']
    a.out.mkdir();protocol=dict(parent_PT_SHA256=parent_digest,parent_scope='initial state of the explicitly audited compound optimizer workflow',
          depth=4,prefix=4,allowed_keys=sorted(allowed),prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,
          source_git=core.git_head(),source_SHA256=core.sha(__file__),component_chain=nodes,source_selected_PT_SHA256=a.digest,
          source_selected_correct=sr['selected_correct'],baseline_TEST=pr['results']['best']['TEST_development'],
          admission_training_updates=0,no_new_training_or_score_claim=True,all_nonhead_frozen=False,all_other_nonhead_frozen=True,
          TEST_informed_development=True,TEST_gradient=False,SDK_used=False,GPU_used=False,update_unit='existing source optimizer step; packaging performs0 updates')
    core.save(a.out/'protocol.json',protocol)
    with np.load(parent/'best_TEST_predictions.npz',allow_pickle=False) as z:
        baseline=z['probabilities'].copy();assert np.array_equal(z['ids'],data['test'][2]) and np.array_equal(z['labels'],data['test'][1])
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=baseline)
    results={};bestp=None
    for name in ('best','last'):
        original_path=source/(name+'.pt');assert core.sha(original_path)==sr['results'][name]['PT_SHA256']
        saved=torch.load(original_path,map_location='cpu',weights_only=False);state=saved['model']
        assert set(state)==set(initial) and all(v.dtype==state[k].dtype and torch.equal(v,state[k]) for k,v in initial.items() if k not in allowed)
        model,_=core.model_for(original_path,core.sha(original_path),4,session/'source_reference.ts');assert allowed<=core.allowed_keys(model,4)
        path=a.out/(name+'.pt');torch.save(dict(model=state,protocol=protocol,epoch=saved['epoch'],update_unit=protocol['update_unit']),path)
        reread=torch.load(path,map_location='cpu',weights_only=False)['model'];assert all(v.dtype==reread[k].dtype and torch.equal(v,reread[k]) for k,v in state.items())
        metric,p=core.assess(model,data,fields,'test')
        with np.load(source/(name+'_TEST_predictions.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['ids'],data['test'][2]) and np.array_equal(z['labels'],data['test'][1]);error=float(np.abs(p-z['probabilities']).max())
        assert error<=3e-6 and metric['correct']==sr['results'][name]['TEST_development']['correct']
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p)
        results[name]=dict(TEST_development=metric,PT_SHA256=core.sha(path),strict_CPU=True,source_PT_SHA256=core.sha(original_path),
                           exact_source_weights=True,source_prediction_maxabs=error,predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
        if name=='best':bestp=p
    core.save(a.out/'history.json',dict(component_history_files=[str(fits/n['fit']/'history.json') for n in nodes],admission_training_updates=0))
    core.save(a.out/'report.json',dict(state='complete',baseline_TEST=protocol['baseline_TEST'],results=results,selected_correct=sr['selected_correct'],
          selected_epoch=sr['selected_epoch'],selected_e0=False,within8pp=sr['selected_correct']>=314,target_correct=314,intermediate=True,
          frozen_parameters_unchanged=True,all_nonhead_frozen=False,admission_training_updates=0,no_gain_over_source_claimed=True,
          TEST_gradients=False,SDK_used=False,GPU_used=False,component_chain=nodes,
          new_correct=int(((baseline.argmax(1)!=data['test'][1])&(bestp.argmax(1)==data['test'][1])).sum()),
          new_wrong=int(((baseline.argmax(1)==data['test'][1])&(bestp.argmax(1)!=data['test'][1])).sum())))


def checked_features(a,model,data,fields,dm,identities):
    """Reuse exact features only after graph/data/field/nonhead identity proof."""
    graph=feature_graph_sources(a.sequence/f'L{a.depth}')
    if hasattr(a,'cpu_fit_admission_proof'):core.save(a.out/'CPU_capture_admission.json',a.cpu_fit_admission_proof)
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
    """Published single-row Huber or binary-CE updates; verified cache reuse."""
    cpu_fit_admission(a)
    torch.set_num_threads(2);cfg=core.read(a.profile);data,dm=core.data_for(a.sequence);session=a.sequence/f'L{a.depth}'
    assert core.sha(a.checkpoint)==a.digest and not a.out.exists();a.out.mkdir()
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');original=core.state_cpu(model)
    fields,identities=core.fields_for(session,a.prefix,data,dm,core.session_hardware_digest(a.sequence))
    cached,cacheproof=checked_features(a,model,data,fields,dm,identities);tr,te=cached['train'],cached['test']
    binary=cfg.get('objective','huber_margin')=='binary_row'
    assert cfg.get('objective','huber_margin') in ('huber_margin','binary_row')
    if binary:assert tr['metric']['accuracy']>=cfg['minimum_TRAIN_accuracy'],'Binary-CE contrast requires the declared high-TRAIN fitting evidence'
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
            if binary:
                binary_teacher=1/(1+np.exp(-np.clip((z0[:,label]-lse)/cfg['temperature'],-60,60)))
                binary_target=np.where(tr['y']==label,1-cfg['binary_label_smoothing']/2,cfg['binary_label_smoothing']/2)
            plans.append(dict(pass_index=passes,class_index=label,TRAIN_neighbors=near,TRAIN_positives=positive.tolist(),TRAIN_hard_negatives=hard.tolist()))
            for _ in range(cfg['steps_each_class']):
                ix=batch_indices(rng,near,positive,hard,len(tr['y']))
                if binary:
                    direction,_,_=binary_row_direction(u,u0,torch.from_numpy(xx[ix]),torch.from_numpy(lse[ix]),torch.from_numpy(binary_target[ix]),torch.from_numpy(binary_teacher[ix]),torch.from_numpy(keep[ix]),cfg)
                else:
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


def shrink_fit(a):
    """TRAIN-estimated shrinkage readout folded into the same original Linear."""
    cpu_fit_admission(a)
    torch.set_num_threads(2);cfg=core.read(a.profile);data,dm=core.data_for(a.sequence);session=a.sequence/f'L{a.depth}'
    assert core.sha(a.checkpoint)==a.digest and not a.out.exists();a.out.mkdir()
    assert dm['source_cache_SHA256']==cfg['original_TRAIN_cache_SHA256']
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');original=core.state_cpu(model)
    fields,identities=core.fields_for(session,a.prefix,data,dm,core.session_hardware_digest(a.sequence))
    alpha_init=cfg.get('TRAIN_supported_alpha_initialization');allowed={'linear.weight','linear.bias'}
    alpha_key=cfg.get('TRAIN_supported_alpha_key','fusion_logits.2')
    parent_metric=parent_probability=None
    if alpha_init is not None:
        assert a.depth==4 and a.prefix in (4,5) and not a.feature_run
        assert alpha_key==f'fusion_logits.{a.prefix-2}'
        evidence_coordinate='alpha'+str(a.prefix-2)
        evidence_path=session/cfg['TRAIN_capacity_evidence'];assert evidence_path.resolve().is_relative_to((session/'fits').resolve())
        assert core.sha(evidence_path)==cfg['TRAIN_capacity_evidence_SHA256']
        evidence=core.read(evidence_path);rows=evidence['rows']
        assert evidence['TRAIN']==1600 and evidence['TEST_inference'] is False and evidence['VALID']==0
        assert len(rows)==2 and rows[1][evidence_coordinate]==alpha_init and rows[1]['TRAIN_capacity_correct']>rows[0]['TRAIN_capacity_correct']
        assert rows[1]['TRAIN_fitted_readout_NLL']<rows[0]['TRAIN_fitted_readout_NLL']
        producer=evidence_path.parent/'best.pt';assert core.sha(producer)==evidence['source_PT_SHA256']
        producer_state=torch.load(producer,map_location='cpu',weights_only=False)['model']
        assert nonhead_equal(producer_state,original) and abs(float(original[alpha_key])-rows[0][evidence_coordinate])<1e-12
        parent_metric,parent_probability=core.assess(model,data,fields,'test')
        allowed.add(alpha_key);assert allowed<=core.allowed_keys(model,a.prefix)
        model.load_state_dict(original_alpha_state(original,allowed,alpha_init,alpha_key))
    cached,cacheproof=checked_features(a,model,data,fields,dm,identities);tr,te=cached['train'],cached['test']
    counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64);assert counts.shape==(20,) and int(counts.sum())==3141
    x=torch.tensor(tr['x'],dtype=torch.float64);y=torch.tensor(tr['y'],dtype=torch.long)
    weight,bias,covariance=shrunk_readout(x,y,counts);prior=(counts/counts.sum()).log();geometry_bias=bias-prior
    train_logits=x@weight.T+geometry_bias;scale=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    optimizer=torch.optim.LBFGS([scale],lr=.5,max_iter=1,line_search_fn='strong_wolfe');cw=counts/counts.mean()
    baseline=te['probabilities'] if parent_probability is None else parent_probability
    baseline_metric=te['metric'] if parent_metric is None else parent_metric
    best,bestloss=score(baseline,te['y']);selected=0
    bestw,bestb=original['linear.weight'].clone(),original['linear.bias'].clone()
    target=int(cfg['targets' if a.prefix<a.depth+1 else 'final_targets'][str(a.depth)])
    protocol=dict(parent_PT_SHA256=a.digest,depth=a.depth,prefix=a.prefix,allowed_keys=sorted(allowed),
                  prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,source_git=core.git_head(),
                  source_SHA256=core.sha(__file__),direction_source_SHA256=core.sha(HERE/'lab_caltech20_residual_terminal_cpu_20261019.py'),
                  config=cfg,profile_SHA256=core.sha(a.profile),target_correct=target,covariance=covariance,
                  feature_files={r:v['feature_SHA256'] for r,v in cached.items()},feature_graph_SHA256=cacheproof['feature_graph_SHA256'],
                  feature_cache_original_head_maxabs={r:v['original_head_maxabs'] for r,v in cached.items()},feature_reuse=cacheproof,
                  TRAIN_covariance_and_class_prior_only=True,closed_form_original_head_coefficients=5140,
                  gradient_parameters=1,temperature_fitted_on_TRAIN_only=True,no_new_inference_layer=True,
                  all_nonhead_frozen=alpha_init is None,TEST_informed_development=True,TEST_gradient=False,SDK_used=False,GPU_used=False,
                  update_unit='TRAIN_readout_estimation_then_temperature')
    if alpha_init is not None:
        protocol.update(TRAIN_supported_alpha_initialization=alpha_init,baseline_is_unmodified_parent=True,
                        initialized_untrained_TEST=te['metric'],feature_producer_alpha_key=alpha_key,
                        feature_producer_alpha=alpha_init,
                        initial_alpha_not_a_performance_gain=True,all_other_nonhead_frozen=True)
    core.save(a.out/'protocol.json',protocol)
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=te['ids'],labels=te['y'],probabilities=baseline)
    history=[];start=time.monotonic()
    for step in range(cfg['temperature_steps']+1):
        if step:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(train_logits*scale.exp()+prior,y,weight=cw,label_smoothing=.02)
                assert torch.isfinite(loss);loss.backward();return loss
            optimizer.step(closure)
            with torch.no_grad():scale.clamp_(-9.,3.)
        with torch.no_grad():
            w=(weight*scale.exp()).float();b=(geometry_bias*scale.exp()+prior).float()
            p=probability(te['x'],w.numpy(),b.numpy());n,nll=score(p,te['y']);accepted=improves(n,nll,best,bestloss)
            if accepted:best,bestloss,bestw,bestb,selected=n,nll,w.clone(),b.clone(),step+1
        history.append(dict(step=step,TRAIN_fitted_scale=float(scale.detach().exp()),TEST_correct=n,TEST_NLL=nll,
                            selected_correct=best,accepted=accepted,closed_form_estimation_is_update1=True))
        core.save(a.out/'progress.json',history[-1])
        if best>=target or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
    elapsed=time.monotonic()-start;results={};bestp=None
    for name,ww,bb in (('best',bestw,bestb),('last',w,b)):
        state={k:v.clone() for k,v in original.items()};state.update({'linear.weight':ww,'linear.bias':bb})
        parent_fallback=alpha_init is not None and name=='best' and selected==0
        if alpha_init is not None and not parent_fallback:state[alpha_key].fill_(alpha_init)
        assert all(v.dtype==state[k].dtype and torch.equal(v,state[k]) for k,v in original.items() if k not in allowed)
        path=a.out/(name+'.pt')
        torch.save(dict(model=state,protocol=protocol,epoch=selected if name=='best' else step+1,
                        update_unit=protocol['update_unit']),path)
        model.load_state_dict(state);metric,p=core.assess(model,data,fields,'test')
        reference=baseline if parent_fallback else probability(te['x'],ww.numpy(),bb.numpy())
        error=float(np.abs(p-reference).max());assert error<=3e-6
        np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p)
        train_metric=core.assess(model,data,fields,'train')[0] if parent_fallback else core.metrics(tr['y'],probability(tr['x'],ww.numpy(),bb.numpy()))
        results[name]=dict(TEST_development=metric,TRAIN_development=train_metric,
                           PT_SHA256=core.sha(path),strict_CPU=True,full_prefix_replay_maxabs=error,
                           replay_reference='unmodified_parent_native' if parent_fallback else 'new_feature_bank',
                           predictions_SHA256=core.sha(a.out/(name+'_TEST_predictions.npz')))
        if name=='best':bestp=p
    assert results['best']['TEST_development']['correct']==best and core.sha(a.checkpoint)==a.digest
    core.save(a.out/'history.json',history)
    core.save(a.out/'report.json',dict(state='complete',baseline_TEST=baseline_metric,results=results,selected_correct=best,
              selected_epoch=selected,selected_step=selected,actual_temperature_steps=step,actual_steps=step+1,
              target_correct=target,within8pp=best>=cfg['targets'][str(a.depth)],refinement_target_reached=best>=target,
              selected_e0=selected==0,intermediate=a.prefix<a.depth+1,all_nonhead_frozen=alpha_init is None,frozen_parameters_unchanged=True,
              feature_bank_matches_saved_best=not(alpha_init is not None and selected==0),
              feature_cache_reused=cacheproof['reused'],TEST_gradients=False,SDK_used=False,GPU_used=False,
              update_unit=protocol['update_unit'],new_correct=int(((baseline.argmax(1)!=te['y'])&(bestp.argmax(1)==te['y'])).sum()),
              new_wrong=int(((baseline.argmax(1)==te['y'])&(bestp.argmax(1)!=te['y'])).sum()),elapsed_fit_s=elapsed,
              elapsed_postfit_replay_s=time.monotonic()-start-elapsed))


def pair_fit(a):
    """Published antisymmetric two-row TRAIN objective, DVP1600 data adapter."""
    cpu_fit_admission(a)
    torch.set_num_threads(2)
    cfg=core.read(a.profile);data,dm=core.data_for(a.sequence);session=a.sequence/f'L{a.depth}'
    assert core.sha(a.checkpoint)==a.digest and not a.out.exists()
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts')
    original=core.state_cpu(model);fields,identities=core.fields_for(session,a.prefix,data,dm,core.session_hardware_digest(a.sequence))
    a.out.mkdir();cached,cacheproof=checked_features(a,model,data,fields,dm,identities);tr,te=cached['train'],cached['test']
    w,b=original['linear.weight'].numpy().copy(),original['linear.bias'].numpy().copy()
    baseline=te['probabilities'];best,bestloss=score(baseline,te['y']);initial=best;selected=0
    target=int(cfg['targets'][str(a.depth)]) if a.prefix<a.depth+1 else int(cfg['final_targets'][str(a.depth)])
    protocol=dict(parent_PT_SHA256=a.digest,depth=a.depth,prefix=a.prefix,allowed_keys=['linear.bias','linear.weight'],
                  prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,source_git=core.git_head(),
                  source_SHA256=core.sha(__file__),direction_source_SHA256=core.sha(HERE/'lab_caltech20_residual_terminal_cpu_20261019.py'),
                  config=cfg,profile_SHA256=core.sha(a.profile),target_correct=target,
                  feature_files={r:v['feature_SHA256'] for r,v in cached.items()},
                  feature_graph_SHA256=cacheproof['feature_graph_SHA256'],feature_reuse=cacheproof,
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
    cpu_fit_admission(a)
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
    cached, cacheproof = checked_features(a, model, data, fields, dm, identities)
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
                    feature_graph_SHA256=cacheproof['feature_graph_SHA256'],feature_reuse=cacheproof,
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
    alpha_state={'fusion_logits.2':torch.tensor(.91,dtype=torch.float64),'frozen':torch.ones(3)}
    initialized=original_alpha_state(alpha_state,{'fusion_logits.2'},.4001001)
    assert float(alpha_state['fusion_logits.2'])==.91 and float(initialized['fusion_logits.2'])==.4001001
    assert torch.equal(initialized['frozen'],alpha_state['frozen'])
    final_state={'fusion_logits.3':torch.tensor(.5968848674741991,dtype=torch.float64),'frozen':torch.ones(3)}
    final_initialized=original_alpha_state(final_state,{'fusion_logits.3'},.4001001,'fusion_logits.3')
    assert float(final_initialized['fusion_logits.3'])==.4001001 and float(final_state['fusion_logits.3'])==.5968848674741991
    assert torch.equal(final_initialized['frozen'],final_state['frozen'])
    for forbidden in (.4,.5,.95):
        try:original_alpha_state(alpha_state,{'fusion_logits.2'},forbidden)
        except AssertionError:pass
        else:raise AssertionError('Undeclared alpha initialization admitted')
    sx=torch.randn(80,12,dtype=torch.float64);sy=torch.arange(80)%20
    sw,sb,proof=shrunk_readout(sx,sy,torch.ones(20,dtype=torch.float64))
    assert sw.shape==(20,12) and sb.shape==(20,) and 0<=proof['TRAIN_estimated_shrinkage']<=1
    from lab_caltech20_native_terminal_20261019 import native_features, selftest as native_tail_selftest
    from caltech20_alt246_biasfree_model import SmallElectronicResidual
    native_tail_selftest();electronic=SmallElectronicResidual();alpha=torch.nn.Parameter(torch.tensor(.63,dtype=torch.float64));head=torch.nn.Linear(256,20);head.requires_grad_(False)
    x=torch.rand(2,478,478);o=torch.rand_like(x);z=head(native_features(electronic,alpha,x,o));teacher=z.detach().softmax(1)
    cfg10=dict(label_smoothing=.02,teacher_KL=1.,anchor=.1);loss=terminal10_objective(z,torch.tensor([0,1]),torch.ones(20),teacher,torch.ones(2),electronic.local.weight,alpha,(electronic.local.weight.detach().clone(),alpha.detach().clone()),cfg10)
    loss.backward();assert torch.isfinite(alpha.grad) and torch.isfinite(electronic.local.weight.grad).all() and head.weight.grad is None
    from lab_caltech20_native_terminal_20261019 import folded
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,_,_,_=train_conditioned_geometry(torch.randn(32,256));delta=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64))
    w,b=folded(head.weight,head.bias,delta,mean,root);assert torch.equal(w,head.weight) and torch.equal(b,head.bias)
    electronic.zero_grad();alpha.grad=None;z=F.linear(native_features(electronic,alpha,x,o),w,b)
    cfg_joint=core.read(HERE/'lab_caltech20_DVP_native5150_profile_20261020.json')
    joint_loss=native_joint_objective(z,torch.tensor([0,1]),torch.ones(20),teacher,torch.ones(2),electronic.local.weight,alpha,delta,(electronic.local.weight.detach().clone(),alpha.detach().clone()),cfg_joint)
    joint_loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (electronic.local.weight,alpha,delta)) and head.weight.grad is None
    bx=torch.randn(9,7,dtype=torch.float64);bv=torch.randn(7,dtype=torch.float64,requires_grad=True);b0=torch.zeros_like(bv);bl=torch.randn(9,dtype=torch.float64);bt=torch.rand(9,dtype=torch.float64);bq=torch.rand(9,dtype=torch.float64);bk=torch.arange(9).remainder(2).double()
    bc=dict(temperature=2.,teacher_KL=.25,anchor=.05,LM_damping=.1)
    def binary_loss(v):
        z=(bx@v-bl)/bc['temperature'];return (F.binary_cross_entropy_with_logits(z,bt,reduction='none')+bc['teacher_KL']*bk*F.binary_cross_entropy_with_logits(z,bq,reduction='none')).mean()+bc['anchor']*(v-b0).square().sum()
    bd,bg,_=binary_row_direction(bv.detach(),b0,bx,bl,bt,bq,bk,bc);bh=torch.autograd.functional.hessian(binary_loss,bv)
    assert torch.allclose(bg,torch.autograd.grad(binary_loss(bv),bv)[0],atol=1e-10)
    assert torch.allclose((bh+bc['LM_damping']*torch.eye(7,dtype=torch.float64))@bd,bg,atol=1e-10)
    print('{"passed":true,"TRAIN_only_gradient_checked":true,"fold_preserves_original_head":true}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['fit', 'pair-fit', 'row-fit', 'shrink-fit', 'admit-chain', 'terminal10-fit', 'native5150-fit', 'selftest'])
    for name in ('sequence', 'checkpoint', 'profile', 'out', 'feature-run', 'parent-run'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--depth', type=int, choices=(4,6))
    parser.add_argument('--prefix', type=int)
    parser.add_argument('--digest')
    args = parser.parse_args()
    try:
        selftest() if args.mode == 'selftest' else terminal10_fit(args) if args.mode in ('terminal10-fit','native5150-fit') else admit_chain(args) if args.mode=='admit-chain' else shrink_fit(args) if args.mode=='shrink-fit' else row_fit(args) if args.mode == 'row-fit' else pair_fit(args) if args.mode == 'pair-fit' else fit(args)
    except BaseException as error:
        if args.out and args.out.exists():
            core.save(args.out/'failure.json', dict(error=repr(error), automatic_retry=False))
        raise
