"""Vision-only training; guided forwards and measured automatic forwards stay distinct."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import model as M
from . import engine as E
from .automatic_probe import observe, merge_observations, run_probe
from .controller import GuidanceController, routing_health, validation_routes
from .data import ArrayDataset, atomic_json, pair_loader, validation_b
from .protocol import load_policy, policy_sha256, require_authorization, require_gpu_preflight, RoutingProtocolFailure
from .selection import validate_source, selected_path


def train(loaded,r,h,bundle,s,data_root,task,variant,source=None,smoke=False,resume=False):
    require_authorization()
    if not smoke:require_gpu_preflight()
    policy=load_policy();signature=policy_sha256(policy);architecture=r.transfer_architecture
    out=s.output_dir;out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'routing_policy.json',policy)
    phases=lambda:{n:p.detach().cpu().clone() for n,p in M.named(r,h).items() if n.endswith(('raw_phase','raw_router_phase'))}
    initial=phases();origin_phases=initial;teacher=None;source_a=None
    if task=='B':
        if source is None:raise RuntimeError('B requires an explicit A checkpoint')
        src=torch.load(source,map_location='cpu',weights_only=False)
        validate_source(src,architecture,E.VERSION,signature,smoke)
        if src.get('split_sha256')!=bundle.metadata['split_sha256']:
            raise RuntimeError('Source A and B dataset splits do not match')
        M.restore(r,h,src);origin_phases=src['origin_phases'];source_a=src['validation_a']['accuracy'];initial=phases()
        cache=Path(source).parent/'teacher_clean_vo.pt'
        if cache.exists():
            teacher=torch.load(cache,map_location='cpu',weights_only=False)
            if teacher.get('source_sha256')!=M.digest(r,h) or teacher.get('policy_sha256')!=signature:
                raise RuntimeError('Teacher cache source/protocol mismatch')
        else:
            ds=ArrayDataset(np.load(data_root/'replay_clean.npy',mmap_mode='r'),np.load(data_root/'replay_labels.npy'))
            _,logits,probs=E.evaluate(loaded,r,h,ds,s,return_outputs=True)
            teacher=dict(logits=logits,routes=probs,source_sha256=M.digest(r,h),policy_sha256=signature)
            E.save(cache,teacher)

    controller=GuidanceController(fraction=policy['guidance']['initial_fraction'] if architecture=='moe' else 0.)
    best=dict(qualified_a=-1.,qualified_b=-1.,qualified_mean=-1.,observed_a=-1.,observed_b=-1.,observed_mean=-1.)
    history=[];opt=None;start=1;global_step=0;latest=out/'latest.pt';failure_reason=None
    if resume and latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if (state['version'],state['policy_sha256'],state['task'],state['variant'],state['architecture'],state['smoke'])!=(E.VERSION,signature,task,variant,architecture,smoke):
            raise RuntimeError('Resume protocol/architecture mismatch')
        if (out/'final.json').exists():raise RuntimeError('Stage already has a terminal outcome; do not silently resume it')
        M.restore(r,h,state);start=state['epoch']+1;global_step=state['global_step'];history=state['history']
        initial=state['initial_phases'];origin_phases=state['origin_phases'];best=state['best_scores']
        controller=GuidanceController(**state['controller_state'])
        opt=M.optimizer_for(r,h,task,state['epoch'],variant);opt.load_state_dict(state['optimizer']);E.restore_rng(state['rng'])
        # Drop only an uncommitted metric tail left by a crash before latest.pt.
        (out/'metrics.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in history),encoding='utf-8')
    elif latest.exists():raise RuntimeError('Output exists; explicit resume is required')
    else:
        E.save(out/'initial_phases.pt',initial);atomic_json(out/'resources.json',M.resource_report(r,h))
    epochs=([1,6,41,60] if task=='A' else [1,6,10,11,31,40]) if smoke else range(1,policy['epochs'][task]+1)

    for epoch in epochs:
        if epoch<start:continue
        opt=M.optimizer_for(r,h,task,epoch,variant,opt);M.set_mode(loaded,r,h,True,task,epoch)
        frozen=M.digest(r,h,True);epoch_initial=phases();started=time.perf_counter()
        n=correct=0;ce_sum=0.;train_routes={};auto_routes={};auto_grads={};guided_grads={};probe_batches=0;probe_seconds=0.;loss_sums={}
        fraction=0.  # Dense four-expert routing in both training and inference
        loader=E.build_train_loader(bundle,s,epoch=epoch) if task=='A' else pair_loader(data_root,epoch,s,smoke)
        print('[stage]',json.dumps(dict(task=task,variant=variant,epoch=epoch,guidance_fraction=fraction,lrs={g['group_name']:g['lr'] for g in opt.param_groups})),flush=True)
        for step,(inputs,meta) in enumerate(E.base._prepared_batches(loader,loaded,s),1):
            meta=meta.to(loaded.device)
            if task=='A':y=meta;domain=torch.zeros_like(y);idx=None
            else:y=meta[:,0];domain=meta[:,1];idx=meta[:,2]
            forced=None
            if fraction>0:
                generator=torch.Generator(device='cpu').manual_seed(3242+epoch*10000+step)
                keep=torch.rand(len(y),generator=generator).to(loaded.device)<fraction
                forced=torch.where(keep,domain,torch.full_like(domain,-1))
            M.force_route(r,forced)
            opt.zero_grad(set_to_none=True)
            with E.autocast(loaded,s):
                logits=M.classification_logits(loaded.model,r,h,inputs)[0]
                ce=F.cross_entropy(logits,y,label_smoothing=float(s.label_smoothing))
                kd=logits.new_zeros(())
                if teacher is not None:
                    clean=domain==0;ids=idx[clean].cpu();temperature=policy['loss']['clean_logit_kd_temperature']
                    target=teacher['logits'][ids].to(loaded.device)
                    kd=temperature**2*F.kl_div(F.log_softmax(logits[clean].float()/temperature,1),F.softmax(target/temperature,1),reduction='batchmean')
                auxiliary,terms=E.routing_loss(r,domain)
                loss=ce+policy['loss']['clean_logit_kd_weight']*kd+(0 if auxiliary is None else auxiliary)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
            if step==1 or step%s.log_every_steps==0:
                values=E.phase_gradients(r,h,ce)
                for name,info in values.items():guided_grads[name]=max(guided_grads.get(name,0.),info['l2'])
                E.append(out/'ce_phase_gradients.jsonl',dict(epoch=epoch,step=step,guidance_fraction=fraction,scope='training_forward',gradients=values))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step();global_step+=1
            n+=len(y);correct+=int((logits.argmax(1)==y).sum());ce_sum+=float(ce.detach())*len(y)
            merge_observations(train_routes,observe(M.routes(r),domain))
            for key,value in dict(classification_ce=ce,clean_logit_kd=kd,**terms).items():
                loss_sums[key]=loss_sums.get(key,0.)+float(value.detach())*len(y)
            if architecture=='moe' and probe_batches<policy['probe']['maximum_batches_per_epoch'] and (step==1 or step%policy['probe']['every_steps']==0):
                probe=run_probe(loaded,r,h,inputs,y,domain,s,task,epoch)
                merge_observations(auto_routes,probe['routes']);probe_batches+=1;probe_seconds+=probe['seconds']
                for name,value in probe['ce_gradient_max'].items():auto_grads[name]=max(auto_grads.get(name,0.),value)
                E.append(out/'automatic_probes.jsonl',dict(epoch=epoch,step=step,**probe))
            if step==1 or step%s.log_every_steps==0:
                status=dict(status='running',task=task,variant=variant,epoch=epoch,step=step,steps_per_epoch=len(loader),global_step=global_step,
                            guidance_fraction=fraction,train_accuracy=correct/n,train_ce=ce_sum/n,images_per_sec=n/(time.perf_counter()-started),
                            automatic_probe_batches=probe_batches,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                atomic_json(out/'status.json',status);print('[train]',json.dumps(status),flush=True)
        if frozen!=M.digest(r,h,True):raise RuntimeError('Frozen parameters or buffers changed')
        train_seconds=time.perf_counter()-started
        va=E.evaluate(loaded,r,h,bundle.validation,s)
        vb=E.evaluate_corrupted(loaded,r,h,s,validation_b(data_root)) if task=='B' else None
        if architecture=='moe':
            probe_health=routing_health(auto_routes,auto_grads,task,policy)
            validation_health=routing_health(validation_routes(va,vb),{},task,policy,require_gradients=False)
            ready=probe_health['ready'] and validation_health['ready']
            transition=controller.update(epoch,task,ready,policy) if not smoke else dict(fraction_used=fraction,fraction_next=fraction,selection_eligible=True,event='smoke_only_no_qualification',failure=None,state=controller.state_dict())
        else:
            probe_health=validation_health=dict(ready=True,reasons=[],applicable=False)
            transition=dict(fraction_used=0.,fraction_next=0.,selection_eligible=True,event='d2nn_no_router',failure=None,state=controller.state_dict())
        eligible=transition['selection_eligible'];qualified=eligible and not smoke
        phase_statistics=E.phase_stats(r,h,initial)
        record=dict(epoch=epoch,task=task,variant=variant,train_accuracy=correct/n,train_ce=ce_sum/n,train_samples=n,train_seconds=train_seconds,
                    images_per_sec=n/train_seconds,validation_a=va,validation_b=vb,routes=train_routes,automatic_routes=auto_routes,
                    loss_components={key:value/n for key,value in loss_sums.items()},frozen_sha256=frozen,frozen_unchanged=True,
                    phase_statistics=phase_statistics,phase_step_statistics=E.phase_stats(r,h,epoch_initial),global_step=global_step,
                    ce_gradient_sampled_max=guided_grads,automatic_ce_gradient_max=auto_grads,automatic_probe_seconds=probe_seconds,
                    probe_health=probe_health,validation_route_health=validation_health,guidance=transition,routing_qualified=qualified)
        history.append(record)
        payload=dict(**M.clone(r,h),version=E.VERSION,policy_sha256=signature,task=task,variant=variant,architecture=architecture,smoke=smoke,
                     epoch=epoch,global_step=global_step,validation_a=va,validation_b=vb,initial_phases=initial,origin_phases=origin_phases,
                     source=str(source) if source else None,split_sha256=bundle.metadata['split_sha256'],routing_qualified=qualified,
                     guidance_fraction_used=fraction,automatic_health=dict(probe=probe_health,validation=validation_health))
        if task=='A':
            if va['accuracy']>best['observed_a']:
                best['observed_a']=va['accuracy'];E.save(out/'best_observed.pt',payload)
            if eligible and va['accuracy']>best['qualified_a']:
                best['qualified_a']=va['accuracy'];E.save(out/'best.pt',payload)
        else:
            satisfied=va['accuracy']>=source_a-policy['selection']['maximum_clean_drop']
            payload['retention_satisfied']=satisfied;score=(va['accuracy']+vb['accuracy'])/2
            if score>best['observed_mean']:
                best['observed_mean']=score;E.save(out/'best_observed_mean.pt',payload)
            if satisfied and vb['accuracy']>best['observed_b']:
                best['observed_b']=vb['accuracy'];E.save(out/'best_observed_retained.pt',payload)
            if eligible and score>best['qualified_mean']:
                best['qualified_mean']=score;E.save(out/'best_mean.pt',payload)
            if eligible and satisfied and vb['accuracy']>best['qualified_b']:
                best['qualified_b']=vb['accuracy'];E.save(out/'best_retained.pt',payload)
        if smoke or epoch%10==0 or (task=='A' and epoch in (1,5,6,41)):
            inputs=E.base._prepare(loaded,[bundle.validation[0][0]],s)
            with torch.no_grad(),E.autocast(loaded,s):M.predict(loaded,r,h,inputs)
            E.export_visuals(r,h,initial,out/'diagnostics'/f'epoch_{epoch:03d}')
        checkpoint=dict(**payload,optimizer=opt.state_dict(),rng=E.rng_state(),history=history,best_scores=best,controller_state=controller.state_dict())
        E.save(latest,checkpoint);E.append(out/'metrics.jsonl',record)
        atomic_json(out/'status.json',dict(status='running',epoch=epoch,task=task,variant=variant,validation_a=va['accuracy'],
                                        validation_b=None if vb is None else vb['accuracy'],guidance=transition,routing_qualified=qualified))
        print('[epoch]',json.dumps(dict(epoch=epoch,accuracy_a=va['accuracy'],accuracy_b=None if vb is None else vb['accuracy'],guidance=transition)),flush=True)
        if transition['failure']:
            failure_reason=transition['failure'];break

    has_qualified=(out/'best.pt').exists() if task=='A' else (out/'best_mean.pt').exists()
    if not has_qualified and not smoke:failure_reason=failure_reason or 'no_qualified_automatic_checkpoint'
    chosen=selected_path(out,task,has_qualified)
    final=torch.load(chosen,map_location='cpu',weights_only=False)
    if task=='B' and has_qualified:
        E.save(out/'best.pt',final);chosen=out/'best.pt'
    result=dict(status='smoke_complete' if smoke else ('routing_failed' if failure_reason else 'complete'),
                failure_reason=failure_reason,task=task,variant=variant,architecture=architecture,epochs_completed=history[-1]['epoch'],
                planned_epochs=policy['epochs'][task],selected_epoch=final['epoch'],validation_a=final['validation_a'],validation_b=final['validation_b'],
                routing_qualified=final.get('routing_qualified',False),retention_satisfied=final.get('retention_satisfied'),
                protocol_succeeded=not smoke and not failure_reason and final.get('routing_qualified',False) and final.get('retention_satisfied') is not False,
                checkpoint=str(chosen),smoke=smoke,policy_sha256=signature)
    atomic_json(out/'final.json',result);atomic_json(out/'status.json',result)
    if failure_reason:
        atomic_json(out/'routing_failure.json',result)
        raise RoutingProtocolFailure(failure_reason)
    return result
