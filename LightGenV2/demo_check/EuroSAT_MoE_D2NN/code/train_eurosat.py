"""Authorized EuroSAT optical/SAR training; no test-set construction in this entry point."""
import argparse, copy, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.vision_transfer.routing_objective import routing_loss
from eurosat_data import *
from eurosat_runtime import *

def train_stage(model,stage):
    source_sha=verify_source()
    checks=json.loads((ROOT/'runs/preflight.json').read_text())
    assert checks['passed'] and checks['source_sha256']==source_sha
    architecture='moe' if model=='moe' else 'd2nn'
    config=STAGES[stage] if stage in STAGES else dict(**PLAN['d2nn_'+model],domains=['A' if stage=='A_only' else 'B'])
    out=(MOE if model=='moe' else D2/model)/stage;out.mkdir(parents=True,exist_ok=True)
    if (out/'final.json').exists():
        final=json.loads((out/'final.json').read_text());assert file_sha(out/'selected.pt')==final['checkpoint_sha256'];return
    loaded,s,r,h=build(architecture,out);teacher=None
    try:
        initial_path=ROOT/'runs'/('initial_moe.pt' if architecture=='moe' else 'initial_d2nn.pt')
        M.restore(r,h,checkpoint(initial_path))
        source=None
        if model=='moe':
            if stage in ('expert_A','expert_B'):source=checkpoint(MOE/'shared/selected.pt')
            elif stage=='router':source=checkpoint(MOE/'merged.pt')
        elif model=='AB' and stage!='shared':
            sequence=list(STAGES);previous=sequence[sequence.index(stage)-1]
            source=checkpoint(D2/'AB'/previous/'latest.pt')
        if source is not None:M.restore(r,h,source)
        initial=M.clone(r,h);initial_phases=phase_snapshot(r,h);shared_digest=C.state_digest(initial,True)
        E.save(out/'initial_phases.pt',initial_phases);atomic(out/'resources.json',M.resource_report(r,h))
        if model=='moe' and stage=='router':
            teacher=(copy.deepcopy(r),copy.deepcopy(h))
            for _,module in M.modules(*teacher):module.requires_grad_(False)
        opt=None;start_epoch=1;best=(-1.,float('-inf'));best_kept=(-1.,float('-inf'))
        if (out/'latest.pt').exists():
            resume=checkpoint(out/'latest.pt');M.restore(r,h,resume)
            opt=optimizer(r,h,architecture,stage,resume['epoch'],config['epochs']);opt.load_state_dict(resume['optimizer'])
            E.restore_rng(resume['rng']);start_epoch=resume['epoch']+1
            best=tuple(resume['best_key']);best_kept=tuple(resume['best_kept_key'])
        for epoch in range(start_epoch,config['epochs']+1):
            opt=optimizer(r,h,architecture,stage,epoch,config['epochs'],opt);M.set_mode(loaded,r,h,False)
            frozen=M.digest(r,h,True);started=time.perf_counter();count=correct=0;ce_sum=0.;exposures={'A':0,'B':0};order=hashlib.sha256();phase_grads={}
            for step,(inputs,meta_cpu) in enumerate(E.base._prepared_batches(train_batches(stage,epoch,config['domains'],config['steps_per_epoch']),loaded,s),1):
                order.update(meta_cpu.numpy().tobytes());meta=meta_cpu.to(loaded.device);y=meta[:,0];d=meta[:,1]
                exposures['A']+=int((d==0).sum());exposures['B']+=int((d==1).sum())
                opt.zero_grad(set_to_none=True)
                mode='automatic' if architecture=='d2nn' or stage=='router' else ('uniform' if stage=='shared' and (epoch+step)%2==0 else 'isolated')
                with E.autocast(loaded,s):
                    logits=forward(loaded,r,h,inputs,mode,d if mode=='isolated' else None)
                    ce=F.cross_entropy(logits,y,label_smoothing=.05);kd=logits.new_zeros(());aux=logits.new_zeros(())
                    if teacher is not None:
                        with torch.no_grad():target=forward(loaded,*teacher,inputs,'isolated',d)
                        kd=4*F.kl_div(F.log_softmax(logits.float()/2,1),F.softmax(target.float()/2,1),reduction='batchmean')
                        aux,_=routing_loss(M.routes(r),d,dict(target_group_probability=.7,target_kl_weight=.02,capture_weight=.005))
                    loss=ce+.2*kd+aux
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                if step==1 and any(p.requires_grad and n.endswith(('raw_phase','raw_router_phase')) for n,p in M.named(r,h).items()):
                    phase_grads=E.phase_gradients(r,h,ce)
                    if any(not v['finite'] or v['l2']<=0 for v in phase_grads.values()):raise RuntimeError('Missing/zero trainable optical CE gradient')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                count+=len(y);correct+=int((logits.argmax(1)==y).sum());ce_sum+=float(ce.detach())*len(y)
                if step==1 or step%10==0:
                    status=dict(state='training',model=model,stage=stage,epoch=epoch,epochs=config['epochs'],step=step,steps=config['steps_per_epoch'],train_accuracy=correct/count,train_ce=ce_sum/count,elapsed_sec=time.perf_counter()-started,updated_at=time.time())
                    atomic(ROOT/'runs/progress.json',status);atomic(out/'status.json',status)
                    print(json.dumps(status),flush=True)
            assert step==config['steps_per_epoch'] and count==step*60
            if frozen!=M.digest(r,h,True):raise RuntimeError('Frozen parameters or buffers changed')
            if model=='moe' and stage in ('expert_A','expert_B') and C.state_digest(M.clone(r,h),True)!=shared_digest:raise RuntimeError('Shared state changed in isolated expert learning')
            training_sec=time.perf_counter()-started
            domains=config['domains'] if model in ('A_only','B_only') or (model=='moe' and stage.startswith('expert_')) else ['A','B']
            val=validate(loaded,r,h,s,domains,mode='automatic' if architecture=='d2nn' or stage=='router' else 'isolated')
            uniform=validate(loaded,r,h,s,mode='uniform') if model=='moe' and stage=='shared' else None
            score=(val['mean']+uniform['mean'])/2 if uniform else val['mean'];tie_ce=(val['mean_ce']+uniform['mean_ce'])/2 if uniform else val['mean_ce']
            key=(score,-tie_ce)
            record=dict(model=model,stage=stage,epoch=epoch,steps=step,train_samples=count,domain_presentations=exposures,
                sample_order_and_augmentation_sha256=order.hexdigest(),train_accuracy=correct/count,train_ce=ce_sum/count,
                train_seconds=training_sec,epoch_seconds=time.perf_counter()-started,validation=val,uniform_validation=uniform,
                selection_score=score,selection_ce=tie_ce,phase_gradients=phase_grads,frozen_unchanged=True,
                learning_rates={g['group_name']:g['lr'] for g in opt.param_groups},phase_statistics=E.phase_stats(r,h,initial_phases))
            if model=='AB':
                reference=json.loads((MOE/stage/f'epoch_{epoch:03d}.json').read_text())
                for field in ('steps','train_samples','domain_presentations','sample_order_and_augmentation_sha256'):
                    if record[field]!=reference[field]:raise RuntimeError('MoE/D2NN data budget mismatch: '+field)
                record['matched_moe_data']=True
            payload=stamped(r,h,**record)
            if key>best:best=key;E.save(out/'best.pt',payload)
            if teacher is not None:
                retained=all(source['isolated_validation'][d]['accuracy']-val[d]['accuracy']<=.03 for d in ('A','B'))
                if retained and key>best_kept:best_kept=key;E.save(out/'best_kept.pt',payload)
            atomic(out/f'epoch_{epoch:03d}.json',record)
            E.save(out/'latest.pt',dict(**payload,optimizer=opt.state_dict(),rng=E.rng_state(),best_key=best,best_kept_key=best_kept))
            print('[epoch]',json.dumps(record),flush=True)
        choice=out/('best_kept.pt' if teacher is not None and best_kept[0]>=0 else 'best.pt')
        selected=checkpoint(choice);E.save(out/'selected.pt',selected);M.restore(r,h,selected)
        E.export_visuals(r,h,initial_phases,out/'diagnostics/selected')
        history=[json.loads(p.read_text()) for p in sorted(out.glob('epoch_*.json'))]
        assert len(history)==config['epochs']
        final=dict(state='complete',model=model,stage=stage,epochs_completed=config['epochs'],selected_epoch=selected['epoch'],validation=selected['validation'],
            checkpoint=str(out/'selected.pt'),checkpoint_sha256=file_sha(out/'selected.pt'),test_evaluated=False,
            total_steps=sum(x['steps'] for x in history),total_samples=sum(x['train_samples'] for x in history),
            domain_presentations={d:sum(x['domain_presentations'][d] for x in history) for d in ('A','B')},
            total_epoch_seconds=sum(x['epoch_seconds'] for x in history),merge_retention_satisfied=(best_kept[0]>=0) if teacher is not None else None,
            source_sha256=source_sha,plan_sha256=signature(PLAN))
        atomic(out/'final.json',final);atomic(out/'status.json',final)
        if model in ('A_only','B_only'):E.save(D2/model/'selected.pt',selected)
    except BaseException as exc:
        atomic(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        r.close()
        if teacher is not None:teacher[0].close()

def merge():
    verify_source();shared=checkpoint(MOE/'shared/selected.pt');a=checkpoint(MOE/'expert_A/selected.pt');b=checkpoint(MOE/'expert_B/selected.pt')
    merged=C.merge_states(shared,a,b);loaded,s,r,h=build('moe',MOE)
    try:
        comparisons={}
        for d,state in (('A',a),('B',b)):
            images,meta=next(eval_batches(d,'validation',batch_size=10,limit=10));inputs=E.base._prepare(loaded,images,s);domain=meta[:,1].to(loaded.device)
            with torch.no_grad(),E.autocast(loaded,s):
                M.restore(r,h,state);original=forward(loaded,r,h,inputs,'isolated',domain).float().clone()
                M.restore(r,h,merged);combined=forward(loaded,r,h,inputs,'isolated',domain).float().clone()
            error=float((original-combined).abs().max());assert torch.equal(original,combined),f'Merge output changed in {d}: {error}'
            comparisons[d]=error
        M.restore(r,h,merged);val=validate(loaded,r,h,s,mode='isolated')
        merged.update(isolated_validation=val,merge_equivalence_max_error=comparisons)
        E.save(MOE/'merged.pt',merged)
        atomic(MOE/'merge_checks.json',dict(passed=True,shared_sha256=merged['shared_sha256'],isolated_validation=val,merge_equivalence_max_error=comparisons,
            isolated_validation_60percent={d:val[d]['accuracy']>=.6 for d in ('A','B')}))
    finally:r.close()

def select_ab():
    states=[checkpoint(D2/'AB'/stage/'selected.pt') for stage in STAGES]
    selected=max(states,key=lambda x:(x['selection_score'],-x['selection_ce']))
    E.save(D2/'AB/selected.pt',selected)
    atomic(D2/'AB/selection.json',dict(selected_stage=selected['stage'],selected_epoch=selected['epoch'],validation=selected['validation'],checkpoint_sha256=file_sha(D2/'AB/selected.pt'),
        policy='Best A/B mean validation over all 85 curriculum epochs; lower CE tiebreak. Curriculum transitions use last weights, optimizer reset at each phase.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['moe','A_only','B_only','AB','merge','select_ab']);p.add_argument('stage',nargs='?');args=p.parse_args()
    if args.model=='merge':merge()
    elif args.model=='select_ab':select_ab()
    else:train_stage(args.model,args.stage or args.model)
