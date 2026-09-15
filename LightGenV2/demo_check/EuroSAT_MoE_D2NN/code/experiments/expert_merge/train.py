import copy,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import *
from experiments.vision_transfer.routing_objective import routing_loss
from experiments.vision_transfer.automatic_probe import observe,merge_observations

def run(stage):
    authorize();cfg=plan();split=prepare();out=ROOT/'runs/merge_seed42'/stage
    pre=json.loads((ROOT/'runs/preflight.json').read_text())
    if pre['status']!='passed' or pre['plan_sha256']!=signature(cfg) or pre['split_sha256']!=split['split_sha256']:raise RuntimeError('Preflight required')
    out.mkdir(parents=True,exist_ok=False);loaded,s=setup(cfg['seed'],out);r,h=build(loaded,s)
    source=None
    if stage in ('expert_A','expert_B'):source=load_checkpoint(out.parent/'shared/selected.pt',split)
    if stage=='router':source=load_checkpoint(out.parent/'merged.pt',split)
    if source is not None:M.restore(r,h,source)
    initial_state=M.clone(r,h);initial_phases={n:p.detach().cpu().clone() for n,p in M.named(r,h).items() if n.endswith(('raw_phase','raw_router_phase'))}
    E.save(out/'initial_phases.pt',initial_phases);atomic_json(out/'resources.json',M.resource_report(r,h))
    shared_hash=state_digest(initial_state,True);teacher=None
    if stage=='router':
        teacher=(copy.deepcopy(r),copy.deepcopy(h))
        for _,module in M.modules(*teacher):module.requires_grad_(False)
    domain=1 if stage=='expert_B' else 0;paired=stage in ('shared','router')
    a=Images(split,0 if paired else domain,'train',True);b=Images(split,1,'train',True)
    opt=None;best=-1.;best_kept=-1.;history=[]
    try:
        for epoch in range(1,cfg['epochs'][stage]+1):
            opt=optimizer(r,h,stage,epoch,opt);M.set_mode(loaded,r,h,False);frozen=M.digest(r,h,True)
            start=time.perf_counter();n=correct=0;ce_sum=0.;route_stats={};sampled_grad={}
            for step,(inputs,meta) in enumerate(E.base._prepared_batches(train_batches(a,b,epoch,cfg['seed']+1442,paired),loaded,s),1):
                meta=meta.to(loaded.device);y=meta[:,0];d=meta[:,1];opt.zero_grad(set_to_none=True)
                mode='automatic' if stage=='router' else ('uniform' if stage=='shared' and (epoch+step)%2==0 else 'isolated')
                with E.autocast(loaded,s):
                    logits=forward(loaded,r,h,inputs,mode,d if mode=='isolated' else None)
                    ce=F.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing']);kd=logits.new_zeros(());aux=logits.new_zeros(())
                    if stage=='router':
                        with torch.no_grad():target=forward(loaded,*teacher,inputs,'isolated',d)
                        t=cfg['teacher_temperature'];kd=t*t*F.kl_div(F.log_softmax(logits.float()/t,1),F.softmax(target.float()/t,1),reduction='batchmean')
                        aux,_=routing_loss(M.routes(r),d,dict(target_group_probability=cfg['route_group_probability'],target_kl_weight=cfg['route_kl_weight'],capture_weight=cfg['capture_weight']))
                    loss=ce+cfg['teacher_kd_weight']*kd+aux
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                if step==1 and any(p.requires_grad and n.endswith(('raw_phase','raw_router_phase')) for n,p in M.named(r,h).items()):
                    sampled_grad=E.phase_gradients(r,h,ce)
                    E.append(out/'phase_gradients.jsonl',dict(epoch=epoch,mode=mode,gradients=sampled_grad,classification_only=True))
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in M.named(r,h).values() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                n+=len(y);correct+=int((logits.argmax(1)==y).sum());ce_sum+=float(ce.detach())*len(y)
                merge_observations(route_stats,{k.replace('_clean','_Product').replace('_corrupted','_Real_World'):v for k,v in observe(M.routes(r),d).items()})
                if step==1 or step%10==0:
                    status=dict(status='training',stage=stage,epoch=epoch,epochs=cfg['epochs'][stage],step=step,train_accuracy=correct/n,train_ce=ce_sum/n,routing_mode=mode)
                    atomic_json(out/'status.json',status);print('[train]',json.dumps(status),flush=True)
            if frozen!=M.digest(r,h,True):raise RuntimeError('Frozen parameters/buffers changed')
            if stage in ('expert_A','expert_B') and state_digest(M.clone(r,h),True)!=shared_hash:raise RuntimeError('Shared interface changed during independent expert learning')
            train_seconds=time.perf_counter()-start
            val=evaluate(loaded,r,h,s,split,'automatic' if stage=='router' else 'isolated')
            uniform=evaluate(loaded,r,h,s,split,'uniform') if stage=='shared' else None
            score=(val['mean']+uniform['mean'])/2 if stage=='shared' else val[('A' if stage=='expert_A' else 'B')]['accuracy'] if stage.startswith('expert_') else val['mean']
            record=dict(stage=stage,epoch=epoch,validation=val,uniform_validation=uniform,score=score,train_accuracy=correct/n,train_ce=ce_sum/n,train_samples=n,train_seconds=train_seconds,epoch_seconds=time.perf_counter()-start,
                frozen_unchanged=True,phase_statistics=E.phase_stats(r,h,initial_phases),ce_gradients=sampled_grad,routes=route_stats,learning_rates={g['group_name']:g['lr'] for g in opt.param_groups})
            history.append(record)
            payload=dict(**M.clone(r,h),**record,merge_plan_sha256=signature(cfg),split_sha256=split['split_sha256'],shared_sha256=shared_hash)
            if score>best:best=score;E.save(out/'best.pt',payload)
            if stage=='router':
                baseline=source['isolated_validation'];kept=all(baseline[k]['accuracy']-val[k]['accuracy']<=cfg['targets']['maximum_merge_drop'] for k in ('A','B'))
                if kept and score>best_kept:best_kept=score;E.save(out/'best_kept.pt',payload)
            E.save(out/'latest.pt',dict(**payload,optimizer=opt.state_dict(),rng=E.rng_state()));E.append(out/'metrics.jsonl',record)
            atomic_json(out/'status.json',dict(status='training',stage=stage,epoch=epoch,epochs=cfg['epochs'][stage],validation_a=val['A']['accuracy'],validation_b=val['B']['accuracy']))
            print('[epoch]',stage,epoch,val['A']['accuracy'],val['B']['accuracy'],flush=True)
            if epoch in (1,cfg['epochs'][stage]):
                E.export_visuals(r,h,initial_phases,out/'diagnostics'/f'epoch_{epoch:03d}')
        chosen=out/('best_kept.pt' if stage=='router' and best_kept>=0 else 'best.pt');state=torch.load(chosen,map_location='cpu',weights_only=False);E.save(out/'selected.pt',state)
        final=dict(status='complete',stage=stage,epochs_completed=cfg['epochs'][stage],selected_epoch=state['epoch'],validation=state['validation'],uniform_validation=state['uniform_validation'],
            checkpoint=str(out/'selected.pt'),checkpoint_sha256=hashlib.sha256((out/'selected.pt').read_bytes()).hexdigest(),merge_plan_sha256=signature(cfg),split_sha256=split['split_sha256'],
            total_train_samples=sum(x['train_samples'] for x in history),total_epoch_seconds=sum(x['epoch_seconds'] for x in history),test_evaluated=False,
            merge_retention_satisfied=None if stage!='router' else best_kept>=0)
        atomic_json(out/'final.json',final);atomic_json(out/'status.json',final)
    except BaseException as exc:
        import traceback
        atomic_json(out/'failure.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        r.close()
        if teacher is not None:teacher[0].close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=plan()['stages']);run(p.parse_args().stage)
