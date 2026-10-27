"""Bounded seed17 Mixup / power-continuation development; final inference is full49."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha


def validate(c):
    if c['seed'] != 17 or c['recipes'] != ['mixup', 'power_curriculum_mixup']:
        raise ValueError('Only the declared seed17 pilot')
    if (c['rounds'], c['samples_per_class'], c['transition_rounds'], c['minimum_fullk_rounds'], c['patience']) != (32,128,16,12,8):
        raise ValueError('Finite adaptation budget changed')
    if c['gpus'] != {'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'} or c['recipe_gpu'] != {'mixup':1,'power_curriculum_mixup':3}:
        raise ValueError('Only GPUs1/3, one worker on each')
    if c['mixup_alpha'] != .2 or c['batch'] != 16 or c['microbatch'] != 2 or c['target_test_correct'] != 616:
        raise ValueError('Mixup, batch or development target changed')


def fraction(recipe, step, c):
    if recipe == 'mixup' or step >= c['transition_rounds']:
        return 1.
    return .5*(1-math.cos(math.pi*(step-1)/(c['transition_rounds']-1)))


def learning_rate(step,c):
    if step <= c['warmup_rounds']:
        return c['lr']*(1/3+2/3*(step-1)/(c['warmup_rounds']-1))
    z=(step-c['warmup_rounds'])/(c['rounds']-c['warmup_rounds'])
    return c['lr_end']+(c['lr']-c['lr_end'])*(1+math.cos(math.pi*z))/2


def sample_indices(labels,step,seed):
    import numpy as np
    labels=np.asarray(labels);chosen=[]
    for k in range(8):
        ix=np.flatnonzero(labels==k)
        if len(ix)<128:raise ValueError('Insufficient class coverage')
        np.random.default_rng(seed+k*100003).shuffle(ix)
        chosen.extend(np.take(ix,np.arange((step-1)*128,step*128)%len(ix)).tolist())
    chosen=np.asarray(chosen,dtype=np.int64)
    np.random.default_rng(seed+step*1000033).shuffle(chosen)
    assert len(np.unique(chosen))==1024
    return chosen


def model_factory(protocol,layers,padding,power_fraction=1.):
    import torch
    from .model import ScalingOptics

    class PowerContinuation(ScalingOptics):
        def route(self,source,dense=False,ste=True):
            f=float(self.power_fraction)
            if f>=1 or dense:
                return super().route(source,dense=dense,ste=ste)
            self.k=24
            try:amp,prob,mask=super().route(source,dense=False,ste=ste)
            finally:self.k=49
            if f<=0:return amp,prob,mask
            sparse=prob*mask
            sparse=sparse/sparse.sum(1,keepdim=True).clamp_min(1e-12)
            power=(1-f)*sparse+f*prob
            hard=power.clamp_min(0).sqrt()
            soft=prob.clamp_min(1e-12).sqrt()
            amp=soft+(hard-soft).detach() if self.training and ste else hard
            return amp,prob,torch.ones_like(mask)

    model=PowerContinuation(protocol,49,49,'moe_oeo',8,layers,padding)
    model.power_fraction=power_fraction
    return model


def sources(profile):
    commit=os.environ['T10_MIXUP_SOURCE_COMMIT']
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    paths=[Path(__file__),Path(profile),*[Path(__file__).with_name(n) for n in ['model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py']],
           Path(__file__).parent/'configs/study.json',root/'LightGenV2/common/optical_contract.py']
    hashes={}
    for p in paths:
        name=p.resolve().relative_to(root).as_posix()
        if p.read_bytes()!=subprocess.check_output(['git','show',commit+':'+name]):raise ValueError('Source changed: '+name)
        hashes[name]=sha(p)
    return commit,hashes


def worker(c,out,profile,recipe):
    import numpy as np
    import torch
    from .train import data,images,evaluate,objective
    from .diagnose_topk import classification_margin_loss,save_route_audit
    commit,hashes=sources(profile);torch.set_num_threads(4)
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    reference=Path(c['reference_run']);source=Path(c['curriculum_source'] if recipe=='power_curriculum_mixup' else c['reference_run'])
    expected=c['curriculum_sha256'] if recipe=='power_curriculum_mixup' else c['reference_sha256']
    if sha(source/'best_checkpoint.pt')!=expected or sha(reference/'best_checkpoint.pt')!=c['reference_sha256']:raise ValueError('Source weight changed')
    m=json.loads((reference/'metadata.json').read_text());sm=json.loads((source/'metadata.json').read_text());a=m['arguments']
    assert sm['arguments']['seed']==17 and sm['arguments']['experts']==49 and sm['arguments']['layers']==a['layers'] and sm['data_sha256']==m['data_sha256']
    cache=Path(a['data']);assert sha(cache)==m['data_sha256'];splits,manifest=data(cache)
    protocol=copy.deepcopy(m['protocol']);protocol['model']['feature_layers_provisional']=a['layers'];tc=protocol['training_proposal'];tc['phase_noise_std']=0.
    model=model_factory(protocol,a['layers'],a['padding'],fraction(recipe,1,c)).cuda()
    initial=torch.load(source/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(initial['model']);source_epoch=initial['epoch'];del initial
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':list(model.expert_phases),'lr_scale':1.,'group_name':'expert'},
        {'params':list(model.global_phases),'lr_scale':.5,'group_name':'global'},
        {'params':[model.router_phase],'lr_scale':.1,'group_name':'router'}],lr=c['lr'],weight_decay=0.)
    out.mkdir(parents=True,exist_ok=False)
    metadata=copy.deepcopy(m)
    for legacy in ['diagnosis','refinement','extended_adaptation','focused_adaptation']:metadata.pop(legacy,None)
    metadata.update(pid=os.getpid(),command=sys.argv,git_commit=commit,source_files=hashes,protocol=protocol,
        host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_run=str(source),source_checkpoint_sha256=expected,
        source_selected_epoch=source_epoch,profile_sha256=sha(profile),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        mixup_curriculum=dict(recipe=recipe,configuration=c,initialization='EMA weights with fresh AdamW and seeded RNG; not exact resume',
        partial_power_checkpoints_not_eligible=True,checkpoint_counter_is_stage_round=True,final_inference='Original full49 ScalingOptics'))
    metadata['arguments'].update(out=str(out),epochs=c['rounds'],lr=c['lr'],top_k=49)
    save(out/'metadata.json',metadata);shutil.copy2(reference/'best_checkpoint.pt',out/'best_checkpoint.pt')
    anchor=model_factory(protocol,a['layers'],a['padding'],1.).cuda().eval().requires_grad_(False)
    anchor.load_state_dict(torch.load(reference/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'])
    baseline,anchor_scores= evaluate(anchor,splits['val'],c['eval_batch']);initial_val,_=evaluate(ema,splits['val'],c['eval_batch'])
    expected_val=json.loads((source/'result.json').read_text())['val']['accuracy']
    if abs(initial_val['accuracy']-expected_val)>1.001/752:raise ValueError('Initialization failed validation reproduction')
    save(out/'initial_validation.json',dict(source=initial_val,full49_anchor=baseline,power_fraction=ema.power_fraction))
    np.savez_compressed(out/'best_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=anchor_scores.numpy(),round=0,power_fraction=1.)
    del anchor;torch.cuda.empty_cache()
    best_key=(-baseline['accuracy'],baseline['macro_nll']);best_round=0;best_acc=baseline['accuracy'];best_nll=baseline['macro_nll'];stale=0;full_rounds=0
    x,y,ids=splits['train'];weights=len(y)/(8*torch.bincount(y,minlength=8).float().cuda())
    rng=np.random.default_rng(c['seed']);probe_ix=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==k))[:64] for k in range(8)])
    probe=(x[probe_ix],y[probe_ix],ids[probe_ix]);history=[];started=time.time();last_progress=0.

    def payload(candidate,step,val):
        return dict(model=candidate.state_dict(),live_model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=step,round=step,
            val=val,power_fraction=candidate.power_fraction,metadata=metadata,rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()))

    for step in range(1,c['rounds']+1):
        f=fraction(recipe,step,c);model.power_fraction=ema.power_fraction=f;model.train();full_rounds+=int(f==1.)
        order=torch.from_numpy(sample_indices(y.numpy(),step,c['seed']));chosen_ids=ids[order.numpy()]
        with(out/'sampling_history.jsonl').open('a') as log:log.write(json.dumps(dict(round=step,indices=order.tolist(),ids=chosen_ids.tolist()))+'\n')
        for g in optimizer.param_groups:g['lr']=learning_rate(step,c)*g['lr_scale']
        totals=0.;batches=0;torch.cuda.synchronize();t0=time.time()
        for start in range(0,len(order),16):
            ix=order[start:start+16];generator=torch.Generator().manual_seed(c['seed']+step*100003+start)
            partner=ix[torch.randperm(len(ix),generator=generator)]
            lam=float(np.random.default_rng(c['seed']+step*1000003+start).beta(c['mixup_alpha'],c['mixup_alpha']));lam=max(lam,1-lam)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(ix),2):
                u,v=ix[offset:offset+2],partner[offset:offset+2]
                image=lam*images(x,u,10000+step,c['seed'],True)+(1-lam)*images(x,v,10000+step,c['seed'],True)
                output=model(image);ya,yb=y[u].cuda(),y[v].cuda()
                loss=lam*(objective(model,output,ya,weights,tc)+c['margin_weight']*classification_margin_loss(output['probabilities'],ya))
                loss+=(1-lam)*(objective(model,output,yb,weights,tc)+c['margin_weight']*classification_margin_loss(output['probabilities'],yb))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite Mixup objective')
                (loss*len(u)/len(ix)).backward();totals+=float(loss.detach())*len(u)/len(ix)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for target,value in zip(ema.parameters(),model.parameters()):target.lerp_(value,1-c['ema_decay'])
            batches+=1
            if time.time()-last_progress>=60:
                save(out/'batch_progress.json',dict(round=step,optimizer_batches=batches,batches_per_round=64,loss=totals/batches,gradient_norm=float(norm),power_fraction=f,full49=f==1.,timestamp=time.time(),pid=os.getpid()));last_progress=time.time()
        torch.cuda.synchronize();training_seconds=time.time()-t0
        t0=time.time();train_probe,_=evaluate(ema,probe,c['eval_batch']);val,scores=evaluate(ema,splits['val'],c['eval_batch']);eval_seconds=time.time()-t0
        np.savez_compressed(out/'latest_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=scores.numpy(),round=step,power_fraction=f)
        eligible=f==1.
        improved=eligible and (val['accuracy']>best_acc or val['macro_nll']<best_nll)
        if eligible:
            key=(-val['accuracy'],val['macro_nll'])
            if key<best_key:
                best_key=key;best_round=step;torch.save(payload(ema,step,val),out/'best_checkpoint.pt')
                np.savez_compressed(out/'best_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=scores.numpy(),round=step,power_fraction=1.)
            best_acc=max(best_acc,val['accuracy']);best_nll=min(best_nll,val['macro_nll'])
            stale=0 if full_rounds<=c['minimum_fullk_rounds'] or improved else stale+1
        gradients={n:dict(norm=float(v.grad.norm()),finite=bool(torch.isfinite(v.grad).all())) for n,v in model.named_parameters() if v.grad is not None}
        full_train=None
        if step%8==0:
            full_train,train_scores=evaluate(ema,splits['train'],c['eval_batch'])
            np.savez_compressed(out/f'train_scores_round{step}.npz',ids=ids,labels=y.numpy(),scores=train_scores.numpy(),power_fraction=f)
        entry=dict(round=step,power_fraction=f,full49=eligible,full49_rounds=full_rounds,train_probe=train_probe,val=val,train_full=full_train,
            training_seconds=training_seconds,probe_and_val_seconds=eval_seconds,elapsed_seconds=time.time()-started,gradients=gradients,gradient_norm=float(norm),mixup_loss=totals/batches,
            learning_rates={g['group_name']:g['lr'] for g in optimizer.param_groups},patience=stale,optimized_images=step*1024,best_round=best_round)
        history.append(entry);save(out/'history.json',history);torch.save(payload(ema,step,val),out/'last_checkpoint.pt')
        save(out/'status.json',dict(state='training',pid=os.getpid(),round=step,power_fraction=f,full49_rounds=full_rounds,best_round=best_round,patience=stale));print(json.dumps(entry),flush=True)
        if eligible and stale>=c['patience']:break
    selected=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if best_round and selected['power_fraction']!=1.:raise RuntimeError('Partial-power selection is forbidden')
    ema.power_fraction=1.;ema.load_state_dict(selected['model']);del selected
    train,_=evaluate(ema,splits['train'],c['eval_batch']);val,_=evaluate(ema,splits['val'],c['eval_batch'])
    for split in ['train','val']:save_route_audit(ema,splits[split],out/f'selected_{split}_routes.npz')
    result=dict(recipe=recipe,seed=17,run=str(out),rounds=len(history),full49_rounds=full_rounds,selected_round=best_round,selected_power_fraction=1.,
        train=train,val=val,checkpoint_sha256=sha(out/'best_checkpoint.pt'),optimized_images=len(history)*1024,source_commit=commit,test_role=c['test_role'])
    save(out/'result.json',result);save(out/'status.json',dict(state='validation_selected',**result))


def test_candidate(c,out,profile):
    import numpy as np
    import torch
    from .train import evaluate
    from .model import ScalingOptics
    commit,hashes=sources(profile);result=json.loads((out/'result.json').read_text());m=json.loads((out/'metadata.json').read_text())
    if result['selected_power_fraction']!=1. or sha(out/'best_checkpoint.pt')!=result['checkpoint_sha256'] or (out/'test_result.json').exists():raise ValueError('TEST lock mismatch or repeated evaluation')
    save(out/'test_selection_lock.json',dict(checkpoint_sha256=result['checkpoint_sha256'],selected_round=result['selected_round'],power_fraction=1.,inference='Original full49 ScalingOptics',selection='VAL checkpoint; authorized TEST-development recipe comparison'))
    reference=Path(c['reference_run']);assert sha(reference/'best_checkpoint.pt')==c['reference_sha256']
    if result['selected_round']==0:
        assert sha(out/'best_checkpoint.pt')==c['reference_sha256']
        old=json.loads((reference/'test_result.json').read_text());assert old['checkpoint_sha256']==c['reference_sha256'] and old['data_sha256']==m['data_sha256']
        for name,value in old['evaluation_dependencies'].items():assert sha(Path(__file__).with_name(name))==value
        assert sha(reference/'test_predictions.npz')==old['predictions_sha256'];shutil.copy2(reference/'test_predictions.npz',out/'test_predictions.npz')
        old.update(role=c['test_role'],reused_from=str(reference));save(out/'test_result.json',old);save(out/'test_reuse_receipt.json',dict(identical_checkpoint=True,gpu_forward_executed=False));return old['test']
    checkpoint=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);assert checkpoint['power_fraction']==1.
    cache=Path(m['arguments']['data']);manifest_path=cache.parent/'data_manifest.json';manifest=json.loads(manifest_path.read_text());assert sha(cache)==m['data_sha256']==manifest['cache_sha256']
    with np.load(cache,allow_pickle=False) as z:x=z['test_images'];y=z['test_labels'];ids=z['test_ids'].astype(str)
    assert ids.tolist()==manifest['split_ids']['test'] and set(ids).isdisjoint(manifest['split_ids']['train']) and set(ids).isdisjoint(manifest['split_ids']['val'])
    split=(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y.astype(np.int64)),ids)
    model=ScalingOptics(m['protocol'],49,49,'moe_oeo',8,m['arguments']['layers'],m['arguments']['padding']).cuda();model.load_state_dict(checkpoint['model']);metrics,scores=evaluate(model,split,c['eval_batch'])
    np.savez_compressed(out/'test_predictions.npz',ids=ids,labels=y,scores=scores.numpy(),predictions=scores.argmax(1).numpy())
    save(out/'test_result.json',dict(test=metrics,checkpoint_sha256=result['checkpoint_sha256'],predictions_sha256=sha(out/'test_predictions.npz'),data_sha256=sha(cache),manifest_sha256=sha(manifest_path),source_commit=commit,source_files=hashes,role=c['test_role'],power_fraction=1.))
    del model;torch.cuda.empty_cache();return metrics


def supervisor(c,out,profile):
    validate(c);commit,hashes=sources(profile);out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(configuration=c,source_commit=commit,source_files=hashes,pid=os.getpid(),host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID');running={};done=[];children=[]
    def stop(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for recipe in c['recipes']:
            index=str(c['recipe_gpu'][recipe]);gpu=c['gpus'][index]
            assert subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==gpu
            if int(subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<7000:raise RuntimeError('Insufficient authorized GPU memory')
            directory=out/(recipe+'_s17');log=directory.with_suffix('.log').open('x');proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.mixup_curriculum','--profile',str(profile),'--out',str(directory),'--worker','--recipe',recipe],env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[proc.pid]=(proc,log,directory);children.append(proc.pid)
        while running:
            for pid,(proc,log,directory) in list(running.items()):
                if proc.poll() is None:continue
                log.close();del running[pid]
                if proc.returncode:raise RuntimeError('Worker failed '+str(directory))
                done.append(json.loads((directory/'result.json').read_text()))
            save(out/'status.json',dict(state='training',scheduler_pid=os.getpid(),running={str(k):str(v[2]) for k,v in running.items()},completed=done))
            if running:time.sleep(10)
        save(out/'all_candidates_lock.json',dict(candidates=done,checkpoint_selection='VAL only, all power fractions1',recipe_selection='User-authorized TEST development'))
        save(out/'status.json',dict(state='testing',scheduler_pid=os.getpid(),running={}))
        os.environ['CUDA_VISIBLE_DEVICES']=c['gpus']['1']
        for row in done:row['test']=test_candidate(c,Path(row['run']),profile)
        selected=max(done,key=lambda row:(row['test']['accuracy'],row['val']['accuracy'],-row['val']['macro_nll']))
        correct=round(selected['test']['accuracy']*752);summary=dict(role=c['test_role'],candidates=done,selected=selected,test_correct=correct,target_correct=c['target_test_correct'],pilot_target_met=correct>=c['target_test_correct'],three_seed_mean_target_established=False)
        save(out/'summary.json',summary);save(out/'status.json',dict(state='complete',scheduler_pid=os.getpid(),running={},pilot_target_met=summary['pilot_target_met']))
    finally:
        for proc,log,directory in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        save(out/'release_check.json',dict(child_pids=children,children_reaped=True,authorized_gpus=c['gpus'],timestamp=time.time()))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--recipe',choices=['mixup','power_curriculum_mixup']);a=ap.parse_args();c=json.loads(a.profile.read_text());validate(c)
    try:
        if a.worker:worker(c,a.out,a.profile,a.recipe)
        else:supervisor(c,a.out,a.profile)
    except BaseException as error:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),scheduler_pid=os.getpid(),error=repr(error)))
        raise


if __name__=='__main__':main()
