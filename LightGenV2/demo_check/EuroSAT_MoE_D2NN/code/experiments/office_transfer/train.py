"""Separate Office-Home curriculum; no CIFAR power-based stopping rule."""
import math,time,json
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.vision_transfer import model as M,engine as E
from experiments.vision_transfer.routing_objective import routing_loss
from experiments.vision_transfer.automatic_probe import observe,merge_observations
from .data import plan,signature,Images,train_batches,atomic_json

def rates(stage,epoch,mode,architecture,variant):
    cfg=plan()[mode]
    if stage=='source':
        if epoch<=cfg['warmup_epochs']:return dict(electronic=2e-4,head=1e-3)
        if epoch<=cfg['phase_end_epoch']:
            return dict(expert_a=3e-3,expert_b=3e-3,shared_phase=1e-3,router=1e-3)
        return dict(electronic=5e-5,head=1e-4,expert_a=5e-4,expert_b=5e-4,shared_phase=5e-4,router=5e-4)
    scale=.25+.75*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['transfer_epochs']-1)))
    if architecture=='d2nn':return dict(shared_phase=3e-3*scale)
    out=dict(expert_b=3e-3*scale,router=1e-3*scale)
    if variant=='all':out['expert_a']=3e-3*scale
    return out

def optimizer(r,h,stage,domain,epoch,mode,variant,old=None):
    lr=rates(stage,epoch,mode,r.transfer_architecture,variant)
    if stage=='source' and r.transfer_architecture=='moe':lr.pop('expert_b' if domain==0 else 'expert_a',None)
    groups={}
    for name,p in M.named(r,h).items():
        group=M.group_of(name);p.grad=None;p.requires_grad_(name in r.transfer_eligible and lr.get(group,0)>0)
        if p.requires_grad:groups.setdefault(group,[]).append(p)
    if old is not None and set(groups)=={g['group_name'] for g in old.param_groups}:
        for g in old.param_groups:g['lr']=lr[g['group_name']]
        return old
    return torch.optim.AdamW([dict(params=ps,lr=lr[g],group_name=g,weight_decay=0.) for g,ps in groups.items()])

def mode_for(loaded,r,h):
    # Dropout and running buffers fixed for every variant; gradients remain enabled.
    M.set_mode(loaded,r,h,False);M.force_route(r,None)

def train(loaded,r,h,s,split,out,seed,mode,stage,domain=0,variant='reserved',source=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);cfg=plan();lc=cfg['loss']
    transfer=stage=='transfer';initial=None;source_a=None;teacher=None
    if transfer:
        src=torch.load(source,map_location='cpu',weights_only=False)
        if src['plan_sha256']!=signature(cfg) or src['split_sha256']!=split['split_sha256'] or src['architecture']!=r.transfer_architecture or src['stage']!='source' or src['domain']!=0:raise RuntimeError('Source checkpoint mismatch')
        M.restore(r,h,src);source_a=src['validation_a']['accuracy']
    initial={n:p.detach().cpu().clone() for n,p in M.named(r,h).items() if n.endswith(('raw_phase','raw_router_phase'))}
    E.save(out/'initial_phases.pt',initial);atomic_json(out/'resources.json',M.resource_report(r,h))
    a=Images(split,0 if transfer else domain,'train',True);b=Images(split,1,'train',True)
    if transfer:
        # Targets use exactly the same augmented A image as the student, every batch.
        import copy
        teacher_r=copy.deepcopy(r);teacher_h=copy.deepcopy(h)
        for _,module in M.modules(teacher_r,teacher_h):module.requires_grad_(False)
        teacher=(teacher_r,teacher_h)
    va=Images(split,0,'validation');vb=Images(split,1,'validation')
    n_epochs=cfg[mode]['transfer_epochs' if transfer else 'source_epochs'];opt=None;best=-1.;best_retained=-1.;history=[]
    for epoch in range(1,n_epochs+1):
        opt=optimizer(r,h,stage,domain,epoch,mode,variant,opt);mode_for(loaded,r,h)
        frozen=M.digest(r,h,True);start=time.perf_counter();n=correct=0;total_ce=0.;routing={};ce_grads={}
        batches=train_batches(a,b,epoch,seed+1442,transfer)
        for step,(inputs,meta) in enumerate(E.base._prepared_batches(batches,loaded,s),1):
            meta=meta.to(loaded.device);y=meta[:,0];dom=meta[:,1];opt.zero_grad(set_to_none=True)
            with E.autocast(loaded,s):
                logits=M.predict(loaded,r,h,inputs);ce=F.cross_entropy(logits,y,label_smoothing=lc['label_smoothing'])
                kd=logits.new_zeros(())
                if teacher is not None:
                    with torch.no_grad():target=M.predict(loaded,*teacher,inputs)
                    t=lc['kd_temperature'];clean=dom==0
                    kd=t*t*F.kl_div(F.log_softmax(logits[clean].float()/t,1),F.softmax(target[clean].float()/t,1),reduction='batchmean')
                w=lc['route_kl_final']+(lc['route_kl_initial']-lc['route_kl_final'])*(1-(epoch-1)/max(1,n_epochs-1))
                aux,terms=routing_loss(M.routes(r),dom,dict(target_group_probability=lc['group_probability'],target_kl_weight=w,capture_weight=lc['capture_weight']))
                loss=ce+lc['kd_weight']*kd+(aux if aux is not None else 0.)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
            if step==1:
                domain_grads={}
                if any(p.requires_grad and n.endswith(('raw_phase','raw_router_phase')) for n,p in M.named(r,h).items()):
                    ce_grads=E.phase_gradients(r,h,ce)
                    for d,label in ((0,'Product'),(1,'Real_World')):
                        if bool((dom==d).any()):domain_grads[label]=E.phase_gradients(r,h,F.cross_entropy(logits[dom==d],y[dom==d]))
                E.append(out/'gradients.jsonl',dict(epoch=epoch,gradients=ce_grads,by_domain=domain_grads,scope='classification_only'))
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
            n+=len(y);correct+=int((logits.argmax(1)==y).sum());total_ce+=float(ce.detach())*len(y)
            observed=observe(M.routes(r),dom)
            merge_observations(routing,{k.replace('_clean','_Product').replace('_corrupted','_Real_World'):v for k,v in observed.items()})
            if step==1 or step%10==0:
                status=dict(status='training',epoch=epoch,epochs=n_epochs,step=step,train_accuracy=correct/n,train_ce=total_ce/n,stage=stage,variant=variant)
                atomic_json(out/'status.json',status);print('[train]',out.name,json.dumps(status),flush=True)
        if frozen!=M.digest(r,h,True):raise RuntimeError('Frozen parameters or buffers changed')
        train_seconds=time.perf_counter()-start
        validation_a=E.evaluate(loaded,r,h,va,s);validation_b=E.evaluate(loaded,r,h,vb,s)
        record=dict(epoch=epoch,validation_a=validation_a,validation_b=validation_b,train_accuracy=correct/n,train_ce=total_ce/n,train_samples=n,train_seconds=train_seconds,
            epoch_seconds=time.perf_counter()-start,phase_statistics=E.phase_stats(r,h,initial),ce_gradients=ce_grads,routes=routing,frozen_unchanged=True,
            learning_rates={g['group_name']:g['lr'] for g in opt.param_groups},gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        history.append(record)
        payload=dict(**M.clone(r,h),**record,stage=stage,domain=domain,variant=variant,architecture=r.transfer_architecture,seed=seed,mode=mode,plan_sha256=signature(cfg),split_sha256=split['split_sha256'])
        score=(validation_a['accuracy']+validation_b['accuracy'])/2 if transfer else (validation_a if domain==0 else validation_b)['accuracy']
        if score>best:best=score;E.save(out/'best_mean.pt',payload)
        if transfer and source_a-validation_a['accuracy']<=cfg['targets']['maximum_forgetting'] and validation_b['accuracy']>best_retained:
            best_retained=validation_b['accuracy'];E.save(out/'best_retained.pt',payload)
        E.save(out/'latest.pt',dict(**payload,optimizer=opt.state_dict(),rng=E.rng_state()));E.append(out/'metrics.jsonl',record)
        atomic_json(out/'status.json',dict(status='training',epoch=epoch,epochs=n_epochs,validation_a=validation_a['accuracy'],validation_b=validation_b['accuracy']))
        if epoch in (cfg[mode]['warmup_epochs']+1,n_epochs):
            with torch.no_grad(),E.autocast(loaded,s):M.predict(loaded,r,h,E.base._prepare(loaded,[va[0][0]],s))
            E.export_visuals(r,h,initial,out/'diagnostics'/f'epoch_{epoch:03d}')
        print('[epoch]',out.name,epoch,validation_a['accuracy'],validation_b['accuracy'],flush=True)
    chosen=out/('best_retained.pt' if transfer and best_retained>=0 else 'best_mean.pt')
    state=torch.load(chosen,map_location='cpu',weights_only=False);E.save(out/'selected.pt',state)
    result=dict(status='complete',checkpoint=str(out/'selected.pt'),selected_epoch=state['epoch'],epochs_completed=n_epochs,validation_a=state['validation_a'],validation_b=state['validation_b'],
        retention_satisfied=None if not transfer else source_a-state['validation_a']['accuracy']<=cfg['targets']['maximum_forgetting'],
        total_train_samples=sum(x['train_samples'] for x in history),total_epoch_seconds=sum(x['epoch_seconds'] for x in history),
        test_evaluated=False,mechanism_status='diagnostics_only_no_power_gate',plan_sha256=signature(cfg),split_sha256=split['split_sha256'])
    atomic_json(out/'final.json',result);atomic_json(out/'status.json',result)
    if teacher is not None:teacher[0].close();del teacher
    return result
