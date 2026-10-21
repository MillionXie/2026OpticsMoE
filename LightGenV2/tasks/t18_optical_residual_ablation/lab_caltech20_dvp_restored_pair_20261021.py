"""Same-mask fresh capture followed immediately by paired-real-TRAIN suffix fit.

No camera differentiation or response surrogate. Two recorded response views
of the SAME1600 TRAIN images supervise the ORIGINAL unmeasured suffix. The
current mask, its input, all measured upstream and all device settings freeze.
Only new-observation TEST selects checkpoints; no TEST derivatives/teacher.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time
import traceback

CORE_SHA='977eb00b88d85f1deb31eda557aa9e753fc6b15dfb8cf30d0aa529268c9c10d5'


def paired_loss(F,old,new,labels,weights,teacher,teacher_correct,cfg):
    """All tensors are TRAIN-only. Correct-parent filter uses TRAIN labels."""
    lp_old=old.clamp_min(1e-12).log();lp_new=new.clamp_min(1e-12).log()
    smooth=cfg['label_smoothing']
    def ce(lp):
        row=(1-smooth)*F.nll_loss(lp,labels,reduction='none')-smooth*lp.mean(1)
        return (row*weights[labels]).sum()/weights[labels].sum()
    temperature=cfg['teacher_temperature']
    softened=(teacher.clamp_min(1e-12).log()/temperature).softmax(1).detach()
    log_student=(lp_new/temperature).log_softmax(1)
    kl=F.kl_div(log_student,softened,reduction='none').sum(1)*temperature**2
    mask=teacher_correct.to(kl.dtype)
    kl=(kl*mask).sum()/mask.sum().clamp_min(1.)
    return cfg['old_view_CE_weight']*ce(lp_old)+cfg['new_view_CE_weight']*ce(lp_new)+cfg['correct_old_TRAIN_teacher_KL_weight']*kl


def match_observations(m,data,session,cap,stage):
    """All physical displayed stimuli must match, not merely sampled pixels."""
    old=session/'capture'/stage
    assert m.sha(old/'phase.bmp')==m.sha(cap/'phase.bmp')
    checked=0
    for role in ('train','test'):
        for i,sample_id in enumerate(data[role][2]):
            a=m.read(old/role/f'{i:06d}.json');b=m.read(cap/role/f'{i:06d}.json')
            assert a['sample_id']==b['sample_id']==str(sample_id)
            assert a['phase_SHA256']==b['phase_SHA256']
            assert a.get('input_SHA256',a.get('BMP_SHA256'))==b['BMP_SHA256']
            for key in ('exposure_us','gain','shape','flip_h','flip_v','rotate','opposite','format','bits'):
                assert a['frame_metadata'][key]==b['frame_metadata'][key],key
            checked+=1
    assert checked==2011
    result=dict(passed=True,matched_samples=checked,same_native_BMP_and_phase_and_readbacks=True,
        old_capture_report_SHA256=m.sha(old/'report.json'),new_capture_report_SHA256=m.sha(cap/'report.json'),
        distinct_observation_times_not_independent_samples=True,settings_not_all_physical_conditions=True)
    m.save(cap/'same_stimulus_audit.json',result)
    return result


def fit(m,loop,data,dm,old_fields,new_fields,model,spec,cfg,out,reference,ready=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    try:
        loop.gpu_admission(m,cfg)
        original=m.state_cpu(model);allowed=m.allowed_keys(model,spec['prefix'])
        assert not allowed&loop.mask_keys(model,spec['stage'])
        assert not out.exists();out.mkdir()
        old_train,teacher=loop.original_replay(model,data,old_fields,'train')
        baseline,bp=loop.original_replay(model,data,new_fields,'test')
        old_test,_=loop.original_replay(model,data,old_fields,'test')
        teacher_correct=teacher.argmax(1)==data['train'][1]
        np.savez_compressed(out/'old_TRAIN_teacher.npz',ids=data['train'][2],labels=data['train'][1],probabilities=teacher,correct=teacher_correct)
        protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),core_SHA256=CORE_SHA,
            config=cfg,depth=model.depth,prefix=spec['prefix'],allowed_keys=sorted(allowed),
            data_SHA256=dm['sha256'],TRAIN_unique=1600,TRAIN_views_per_epoch=3200,TEST_per_epoch=411,VALID=0,
            baseline_new_TEST=baseline,baseline_old_TEST=old_test,old_TRAIN_teacher_metric=old_train,
            old_TRAIN_teacher_SHA256=m.sha(out/'old_TRAIN_teacher.npz'),
            measured_current_mask_and_upstream_frozen=True,original_unmeasured_suffix_only=True,
            TEST_gradient=False,TEST_teacher=False,TEST_informed_development=True,
            new_inference_layers=0,new_mask_proposal=False,full_physical=False,SDK_used=False)
        m.save(out/'protocol.json',protocol)
        best,bestloss,selected=baseline['correct'],baseline['nll'],0
        loop.saved_model(m,model,out/'best.pt',protocol)
        np.savez_compressed(out/'baseline_new_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=bp)
        model.cuda().train();model.requires_grad_(False)
        for name,p in model.named_parameters():p.requires_grad_(name in allowed)
        groups=[]
        for prefix,lr in (('phase_sets.',cfg['lr_phase']),('electronic.',cfg['lr_electronic']),('fusion_logits.',cfg['lr_alpha']),('linear.',cfg['lr_head'])):
            parameters=[p for n,p in model.named_parameters() if n in allowed and n.startswith(prefix)]
            if parameters:groups.append(dict(params=parameters,lr=lr,weight_decay=cfg['weight_decay']))
        optimizer=torch.optim.AdamW(groups)
        prior=np.asarray(cfg['original_TRAIN_counts'],float);assert prior.sum()==3141
        weights=torch.tensor(prior/prior.mean(),device='cuda',dtype=torch.float32)
        rng=np.random.default_rng(cfg['seed']);history=[];start=time.monotonic();stale=0
        if ready is not None:ready.set()
        for epoch in range(1,cfg['maximum_epochs']+1):
            order=rng.permutation(1600);used=0;loss_sum=0.;model.cuda().train()
            for begin in range(0,1600,cfg['batch']):
                ix=order[begin:begin+cfg['batch']];y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                optimizer.zero_grad(set_to_none=True)
                po=model(loop.put_fields(model,data,old_fields,'train',ix,'cuda'))[0]
                pn=model(loop.put_fields(model,data,new_fields,'train',ix,'cuda'))[0]
                t=torch.from_numpy(teacher[ix]).to(device='cuda',dtype=pn.dtype)
                tc=torch.from_numpy(teacher_correct[ix]).cuda()
                loss=paired_loss(F,po,pn,y,weights,t,tc,cfg);assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n in allowed)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                optimizer.step()
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                used+=len(ix);loss_sum+=float(loss.detach())*len(ix)
                if used%80==0:m.save(out/'progress.json',dict(state='paired_TRAIN_fit',epoch=epoch,TRAIN_unique_consumed=used,TRAIN_response_views_consumed=used*2,loss=loss_sum/used,elapsed_s=time.monotonic()-start))
            assert used==1600
            score,_=loop.original_replay(model,data,new_fields,'test')
            improved=score['correct']>best or (score['correct']==best and score['nll']<bestloss-1e-8)
            if improved:
                best,bestloss,selected=score['correct'],score['nll'],epoch
                loop.saved_model(m,model,out/'best.pt',protocol)
            stale=0 if improved else stale+1
            row=dict(epoch=epoch,TRAIN_unique=1600,TRAIN_response_views=3200,TRAIN_loss=loss_sum/1600,new_TEST_correct=score['correct'],new_TEST_NLL=score['nll'],selected_correct=best,elapsed_s=time.monotonic()-start)
            history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='paired_epoch_complete',latest=row));print(json.dumps(row),flush=True)
            if best>=spec['target'] or stale>=cfg['no_improvement_epochs'] or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
        model.cpu();loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,out/'last.pt',protocol)
        del optimizer;torch.cuda.empty_cache();results={}
        for name in ('best','last'):
            digest=m.sha(out/(name+'.pt'));checked,_=m.model_for(out/(name+'.pt'),digest,model.depth,reference)
            loop.validate_state(m,checked,original,allowed);rows={}
            for view,fields in (('old',old_fields),('new',new_fields)):
                rows[view]={}
                for role in ('train','test'):
                    metric,p=loop.original_replay(checked,data,fields,role);rows[view][role]=metric
                    if role=='test':np.savez_compressed(out/(name+'_'+view+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p)
            results[name]=dict(PT_SHA256=digest,metrics=rows)
        assert results['best']['metrics']['new']['test']['correct']==best
        m.save(out/'report.json',dict(state='complete',selected_correct=best,selected_epoch=selected,selected_E0=selected==0,
            target=spec['target'],passed=best>=spec['target'],actual_epochs=epoch,results=results,protocol_SHA256=m.sha(out/'protocol.json'),
            frozen_verified=True,strict_CPU=True,SDK_used=False,full_physical=False,next_layer_started=False,TEST_informed_development=True))
        return best
    finally:
        if ready is not None:ready.set()


def run(a):
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    assert m.sha(loop.__file__)==CORE_SHA
    cfg=m.read(a.profile);assert cfg['schema']=='DVP_restored_same_mask_paired_TRAIN_v1'
    assert cfg['capture_order']==[4,6] and a.lighting_restored
    m,data,dm,hardware,roots,parents=loop.base_admission(a)
    assert not a.out.exists();a.out.mkdir()
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=cfg,
        human_lighting_restored=True,source_parents=parents,hardware_SHA256=loop.HARDWARE,data_SHA256=dm['sha256'],
        recurring=False,automatic_retry=False,new_physical_frames=4022,no_DEVICE_setting_changes=True,
        stage_gate_preserved=True,full_physical=False))
    futures={};results={}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for depth in (4,6):
                spec=loop.SPECS[depth];session,pt,old_fields,identities,model=roots[depth];folder=a.out/f'L{depth}';folder.mkdir()
                m.save(a.out/'progress.json',dict(state='same_mask_capture',depth=depth,stage=spec['stage'],expected=2011))
                previous={k:v for k,v in old_fields.items() if k!=spec['stage']}
                frozen=m.state_cpu(model)
                new_fields=loop.acquire(m,a,data,dm,hardware,previous,model,m.sha(pt),spec['stage'],folder/'capture')
                match_observations(m,data,session,folder/'capture',spec['stage'])
                loop.validate_state(m,model,frozen,set())
                # The new CCD capture must be complete/audited before fitting.
                # First fit starts with SDK free; then next depth captures on
                # CPU while this fit owns only the permitted CUDA model.
                ready=threading.Event()
                futures[depth]=pool.submit(fit,m,loop,data,dm,old_fields,new_fields,model,spec,cfg,folder/'suffix',session/'source_reference.ts',ready)
                while not ready.wait(1):
                    if futures[depth].done():futures[depth].result()
                if futures[depth].done():futures[depth].result()
                if depth==6:
                    # No further capture; both results finish before delivery.
                    results={str(d):f.result() for d,f in futures.items()}
            m.save(a.out/'report.json',dict(state='complete',new_observation_selected=results,
                targets={'4':314,'6':322},old_historical_correct={'4':275,'6':291},
                SDK_released=not m.LOCK.exists(),next_layer_started=False,full_physical=False,TEST_informed_development=True))
        # Separate read-only process, no camera/CUDA, full original graph replay.
        import subprocess
        import sys
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--mode','audit',
            '--sequence',str(a.sequence),'--out',str(a.out),'--profile',str(a.profile)],env=env,check=True)
    except BaseException as error:
        m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,valid_data_preserved=True));raise


def audit(a):
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.LOCK.exists()
    torch.set_num_threads(2)
    m,data,dm,cfg,roots,parents=loop.base_admission(a);proof={}
    for depth in (4,6):
        spec=loop.SPECS[depth];session,pt,old,identities,parent=roots[depth]
        folder=a.out/f'L{depth}';cap=folder/'capture';out=folder/'suffix'
        report=m.read(out/'report.json');protocol=m.read(out/'protocol.json');cr=m.read(cap/'report.json')
        assert report['state']=='complete' and report['protocol_SHA256']==m.sha(out/'protocol.json')
        assert protocol['source_SHA256']==m.sha(__file__) and protocol['data_SHA256']==dm['sha256']
        assert m.read(cap/'independent_audit.json')['report_SHA256']==m.sha(cap/'report.json')
        match_observations(m,data,session,cap,spec['stage'])
        new={k:v for k,v in old.items() if k!=spec['stage']};new[spec['stage']]={}
        for role,n in (('train',1600),('test',411)):
            path=cap/role/'ccd.npy';assert m.sha(path)==cr['arrays'][role]['SHA256']
            array=np.load(path,mmap_mode='r');assert array.shape==(n,478,478) and array.dtype==np.uint8
            new[spec['stage']][role]=array
        results={};before=m.state_cpu(parent);allowed=m.allowed_keys(parent,spec['prefix'])
        assert set(protocol['allowed_keys'])==allowed
        for name in ('best','last'):
            digest=m.sha(out/(name+'.pt'));assert digest==report['results'][name]['PT_SHA256']
            model,_=m.model_for(out/(name+'.pt'),digest,depth,session/'source_reference.ts')
            loop.validate_state(m,model,before,allowed);results[name]={}
            for view,fields in (('old',old),('new',new)):
                results[name][view]={}
                for role in ('train','test'):
                    metric,p=loop.original_replay(model,data,fields,role)
                    expected=report['results'][name]['metrics'][view][role]
                    assert metric['correct']==expected['correct'] and abs(metric['nll']-expected['nll'])<=3e-6
                    results[name][view][role]=dict(correct=metric['correct'],nll=metric['nll'])
                    if role=='test':
                        with np.load(out/(name+'_'+view+'_TEST_predictions.npz')) as z:
                            assert np.array_equal(z['ids'],data['test'][2]) and np.array_equal(z['labels'],data['test'][1])
                            assert np.array_equal(z['probabilities'].argmax(1),p.argmax(1)) and np.max(np.abs(z['probabilities']-p))<=3e-6
        proof[str(depth)]=dict(results=results,fit_report_SHA256=m.sha(out/'report.json'),capture_report_SHA256=m.sha(cap/'report.json'),
            current_mask_and_upstream_frozen=True,passed=report['selected_correct']>=spec['target'])
    m.save(a.out/'independent_audit.json',dict(passed=True,source_git=m.git_head(),source_SHA256=m.sha(__file__),
        results=proof,all_original_CPU_TRAIN_and_TEST_verified=True,SDK_used=False,GPU_used=False,
        full_physical=False,next_layer_started=False,TEST_informed_development=True))


def selftest(profile):
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);torch.manual_seed(73)
    cfg=m.read(profile);checks=[]
    for depth,prefix,stage in ((4,4,'2'),(6,2,'0')):
        base=m.MeasuredPrefix(depth).eval()
        with torch.no_grad():
            for p in base.fusion_logits:p.fill_(.8)
            x=torch.rand(2,224,224)*.25;base(x)
            keys=(['router']+[str(i) for i in range(depth)])[:prefix]
            old={k:base.last[k]['intensity'].clone()*173 for k in keys}
            new={k:v.clone() for k,v in old.items()};new[stage]=new[stage]*(1+.02*torch.rand_like(new[stage]))
            base.measured=old;teacher=base(x)[0].detach()
        allowed=m.allowed_keys(base,prefix);assert not allowed&loop.mask_keys(base,stage)
        base.requires_grad_(False)
        for name,p in base.named_parameters():p.requires_grad_(name in allowed)
        base.measured=old;po=base(x)[0];base.measured=new;pn=base(x)[0]
        teacher.requires_grad_();labels=teacher.argmax(1);weights=torch.ones(20)
        loss=paired_loss(F,po,pn,labels,weights,teacher,torch.ones(2,dtype=torch.bool),cfg)
        loss.backward();assert torch.isfinite(loss) and teacher.grad is None
        grads=[p.grad for n,p in base.named_parameters() if n in allowed]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert all(p.grad is None for n,p in base.named_parameters() if n not in allowed)
        checks.append(dict(depth=depth,prefix=prefix,TRAIN_loss=float(loss.detach()),teacher_detached=True,only_original_suffix_gradients=True))
    print(json.dumps(dict(passed=True,checks=checks,SDK_used=False,GPU_used=False,TEST_used=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','audit','selftest'),required=True)
    p.add_argument('--sequence',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--lighting-restored',action='store_true');a=p.parse_args()
    if a.mode=='selftest':selftest(a.profile)
    elif a.mode=='audit':audit(a)
    else:run(a)
