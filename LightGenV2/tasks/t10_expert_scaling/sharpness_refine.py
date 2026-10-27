"""Paired hard-label AdamW / SAM-AdamW refinement of the original full49 model."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha
from .mixup_curriculum import learning_rate, sample_indices


def validate(c):
    if (c['seed'],c['recipes'],c['rounds'],c['samples_per_class']) != (17,['hard_label','sam'],24,128):
        raise ValueError('Only the paired finite seed17 experiment')
    if (c['batch'],c['microbatch'],c['rho'],c['lr'],c['lr_end'],c['warmup_rounds']) != (16,2,.05,.0003,.00003,3):
        raise ValueError('Declared optimization budget changed')
    if c['recipe_gpu'] != {'hard_label':1,'sam':3} or c['gpus'] != {'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}:
        raise ValueError('Only GPUs1/3, one worker each')
    if c['target_test_correct'] != 616 or c['ema_decay'] != .95:
        raise ValueError('Target or EMA changed')


def sources(profile):
    from .mixup_curriculum import sources as base_sources
    commit=os.environ['T10_SAM_SOURCE_COMMIT']
    previous=os.environ.get('T10_MIXUP_SOURCE_COMMIT')
    os.environ['T10_MIXUP_SOURCE_COMMIT']=commit
    try:
        _,hashes=base_sources(profile)
    finally:
        if previous is None:os.environ.pop('T10_MIXUP_SOURCE_COMMIT',None)
        else:os.environ['T10_MIXUP_SOURCE_COMMIT']=previous
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    name=Path(__file__).resolve().relative_to(root).as_posix()
    if Path(__file__).read_bytes()!=subprocess.check_output(['git','show',commit+':'+name]):
        raise ValueError('Active sharpness module changed')
    hashes[name]=sha(Path(__file__))
    return commit,hashes


def optimization_step(model,optimizer,closure,rho=0.,clip_norm=1.):
    """Accumulate the whole effective batch before the SAM ascent and descent."""
    import torch
    optimizer.zero_grad(set_to_none=True)
    clean_loss=closure()
    params=[p for p in model.parameters() if p.grad is not None]
    norm=torch.stack([p.grad.detach().norm() for p in params]).norm()
    if not torch.isfinite(norm) or float(norm)<=0:raise RuntimeError('Invalid clean gradient')
    first_norm=float(norm);perturbed_loss=None;actual_radius=0.;restored=True;gradient_cosine=None;gradient_relative_change=None
    if rho:
        original_gradients=[p.grad.detach().clone() for p in params]
        original=[p.detach().clone() for p in params]
        with torch.no_grad():
            for p in params:p.add_(p.grad,alpha=rho/first_norm)
            actual_radius=float(torch.stack([(p-v).norm() for p,v in zip(params,original)]).norm())
        optimizer.zero_grad(set_to_none=True)
        try:
            perturbed_loss=closure()
        finally:
            with torch.no_grad():
                for p,v in zip(params,original):p.copy_(v)
                restored=all(torch.equal(p,v) for p,v in zip(params,original))
        if not restored:raise RuntimeError('SAM restoration failed')
        second_norm=torch.stack([p.grad.norm() for p in params]).norm()
        gradient_cosine=float(torch.stack([(g*p.grad).sum() for g,p in zip(original_gradients,params)]).sum()/(norm*second_norm))
        gradient_relative_change=float(torch.stack([(p.grad-g).norm() for g,p in zip(original_gradients,params)]).norm()/norm)
    gradient_groups={n:dict(norm=float(p.grad.norm()),finite=bool(torch.isfinite(p.grad).all()))
                     for n,p in model.named_parameters() if p.grad is not None}
    if not all(g['finite'] for g in gradient_groups.values()):raise RuntimeError('Nonfinite descent gradient')
    if clip_norm is None:
        final_norm=torch.stack([p.grad.norm() for p in params]).norm()
    else:
        final_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
    optimizer.step()
    return dict(clean_loss=clean_loss,perturbed_loss=perturbed_loss,first_gradient_norm=first_norm,
        gradient_norm=float(final_norm),gradients=gradient_groups,rho_actual=actual_radius,
        restored_before_optimizer=restored,sharpness_delta=None if perturbed_loss is None else perturbed_loss-clean_loss,
        sam_gradient_cosine=gradient_cosine,sam_gradient_relative_change=gradient_relative_change)


def gradient_probe(c,out,profile):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import data,images,objective
    from .diagnose_topk import classification_margin_loss
    commit,hashes=sources(profile);assert os.environ['CUDA_VISIBLE_DEVICES'] in c['gpus'].values();torch.set_num_threads(4);torch.manual_seed(17)
    out.mkdir(parents=True,exist_ok=False);save(out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,configuration=c,role='Fixed16TRAIN numeric SAM probe, no VAL/TEST'))
    reference=Path(c['reference_run']);assert sha(reference/'best_checkpoint.pt')==c['reference_sha256'];m=json.loads((reference/'metadata.json').read_text())
    assert sha(Path(m['arguments']['data']))==m['data_sha256'];splits,_=data(Path(m['arguments']['data']));x,y,ids=splits['train'];cfg=copy.deepcopy(m['protocol']);cfg['training_proposal']['phase_noise_std']=0.
    model=ScalingOptics(cfg,49,49,'moe_oeo',8,m['arguments']['layers'],m['arguments']['padding']).cuda();checkpoint=torch.load(reference/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);del checkpoint
    original={n:p.detach().cpu().clone() for n,p in model.state_dict().items()};indices=torch.from_numpy(np.concatenate([np.flatnonzero(y.numpy()==k)[:2] for k in range(8)]));weights=len(y)/(8*torch.bincount(y,minlength=8).float().cuda())
    model.eval()
    with torch.no_grad():baseline=model(images(x,indices[:2],10001,17,True))['probabilities'].clone()
    model.train();initial_lr=learning_rate(1,c);optimizer=torch.optim.AdamW([{'params':list(model.expert_phases),'lr':initial_lr},
        {'params':list(model.global_phases),'lr':initial_lr*.5},{'params':[model.router_phase],'lr':initial_lr*.1}],weight_decay=0.)
    def closure():
        value=0.
        for start in range(0,16,2):
            ix=indices[start:start+2];output=model(images(x,ix,10001,17,True));labels=y[ix].cuda()
            assert torch.all(output['route_mask']==1) and torch.all(output['route_probabilities']>0)
            loss=objective(model,output,labels,weights,cfg['training_proposal'])+.2*classification_margin_loss(output['probabilities'],labels)
            (loss/8).backward();value+=float(loss.detach())/8
        return value
    info=optimization_step(model,optimizer,closure,c['rho']);assert len(info['gradients'])==5 and all(q['finite'] and q['norm']>0 for q in info['gradients'].values())
    assert abs(info['rho_actual']-c['rho'])<1e-5 and info['restored_before_optimizer']
    changes={n:float((p.detach().cpu()-original[n]).abs().max()) for n,p in model.named_parameters()};assert all(v>0 for v in changes.values())
    model.load_state_dict(original);model.eval()
    with torch.no_grad():assert torch.equal(model(images(x,indices[:2],10001,17,True))['probabilities'],baseline)
    save(out/'receipt.json',dict(source_commit=commit,source_files=hashes,profile_sha256=sha(profile),pid=os.getpid(),command=sys.argv,optimizer_group_lrs=[g['lr'] for g in optimizer.param_groups],
        source_checkpoint_sha256=c['reference_sha256'],data_sha256=m['data_sha256'],train_ids=ids[indices.numpy()].tolist(),samples=16,classes=8,full_parameters=sum(p.numel() for p in model.parameters()),
        actual_optimizer_step=info,parameter_max_changes=changes,source_restored_exactly=True,no_val_test_forward=True,
        numeric_gate_passed=bool(info['sam_gradient_relative_change']>=.01 and info['sam_gradient_cosine']<.9999995),numeric_gate='Gradient change>=1% and cosine<.9999995 only establishes a measurable second-pass change on16TRAIN views, not generalization.'))


def worker(c,out,profile,recipe):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import data,images,evaluate,objective
    from .diagnose_topk import classification_margin_loss,save_route_audit
    commit,hashes=sources(profile);torch.set_num_threads(4)
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    reference=Path(c['reference_run']);assert sha(reference/'best_checkpoint.pt')==c['reference_sha256']
    metadata=copy.deepcopy(json.loads((reference/'metadata.json').read_text()));a=metadata['arguments']
    assert a['seed']==17 and a['experts']==a['top_k']==49
    cache=Path(a['data']);assert sha(cache)==metadata['data_sha256'];splits,_=data(cache)
    protocol=copy.deepcopy(metadata['protocol']);protocol['training_proposal']['phase_noise_std']=0.
    tc=protocol['training_proposal']
    model=ScalingOptics(protocol,49,49,'moe_oeo',8,a['layers'],a['padding']).cuda()
    checkpoint=torch.load(reference/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);source_epoch=checkpoint['epoch'];del checkpoint
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':list(model.expert_phases),'lr_scale':1.,'group_name':'expert'},
        {'params':list(model.global_phases),'lr_scale':.5,'group_name':'global'},
        {'params':[model.router_phase],'lr_scale':.1,'group_name':'router'}],lr=c['lr'],weight_decay=0.)
    out.mkdir(parents=True,exist_ok=False)
    for old in ['diagnosis','refinement','extended_adaptation','focused_adaptation']:metadata.pop(old,None)
    metadata.update(pid=os.getpid(),command=sys.argv,git_commit=commit,source_files=hashes,
        host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol=protocol,
        source_run=str(reference),source_checkpoint_sha256=c['reference_sha256'],source_selected_epoch=source_epoch,
        profile_sha256=sha(profile),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        sharpness_refinement=dict(recipe=recipe,configuration=c,initialization='Selected EMA, fresh AdamW, seeded RNG; not exact resume',
            same_effective_batch_for_ascent_and_descent=True,clean_hard_labels=True,no_mixup=True,
            rho_definition='Global L2 norm in raw phase radians; not per-pixel noise',checkpoint_counter_is_stage_round=True))
    metadata['arguments'].update(out=str(out),epochs=c['rounds'],lr=c['lr'],top_k=49)
    save(out/'metadata.json',metadata);shutil.copy2(reference/'best_checkpoint.pt',out/'best_checkpoint.pt')
    initial,scores=evaluate(ema,splits['val'],c['eval_batch'])
    assert abs(initial['accuracy']-json.loads((reference/'result.json').read_text())['val']['accuracy'])<=1.001/752
    save(out/'initial_validation.json',initial)
    np.savez_compressed(out/'best_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=scores.numpy(),round=0,power_fraction=1.)
    x,y,ids=splits['train'];weights=len(y)/(8*torch.bincount(y,minlength=8).float().cuda())
    rng=np.random.default_rng(c['seed']);ix=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==k))[:64] for k in range(8)])
    probe=(x[ix],y[ix],ids[ix]);best_key=(-initial['accuracy'],initial['macro_nll']);best_round=0;history=[];started=time.time();last_progress=0.

    def payload(step,val):
        return dict(model=ema.state_dict(),live_model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=step,round=step,
            val=val,power_fraction=1.,metadata=metadata,rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()))

    for step in range(1,c['rounds']+1):
        model.train();order=torch.from_numpy(sample_indices(y.numpy(),step,c['seed']))
        with(out/'sampling_history.jsonl').open('a') as log:log.write(json.dumps(dict(round=step,indices=order.tolist(),ids=ids[order.numpy()].tolist()))+'\n')
        for group in optimizer.param_groups:group['lr']=learning_rate(step,c)*group['lr_scale']
        total=0.;sharpness=[];first_norms=[];cosines=[];relative_changes=[];torch.cuda.synchronize();t0=time.time()
        for start in range(0,len(order),16):
            batch=order[start:start+16]
            def closure():
                value=0.
                for offset in range(0,len(batch),2):
                    u=batch[offset:offset+2];output=model(images(x,u,10000+step,c['seed'],True));labels=y[u].cuda()
                    loss=objective(model,output,labels,weights,tc)+c['margin_weight']*classification_margin_loss(output['probabilities'],labels)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite hard-label objective')
                    (loss*len(u)/len(batch)).backward();value+=float(loss.detach())*len(u)/len(batch)
                return value
            info=optimization_step(model,optimizer,closure,c['rho'] if recipe=='sam' else 0.)
            if len(info['gradients'])!=5 or not all(g['norm']>0 for g in info['gradients'].values()):raise RuntimeError('Missing phase gradient')
            total+=info['clean_loss'];first_norms.append(info['first_gradient_norm'])
            if info['sharpness_delta'] is not None:sharpness.append(info['sharpness_delta'])
            if info['sam_gradient_cosine'] is not None:cosines.append(info['sam_gradient_cosine']);relative_changes.append(info['sam_gradient_relative_change'])
            with torch.no_grad():
                for target,value in zip(ema.parameters(),model.parameters()):target.lerp_(value,1-c['ema_decay'])
            if time.time()-last_progress>=60:
                save(out/'batch_progress.json',dict(round=step,optimizer_batches=start//16+1,batches_per_round=64,loss=total/(start//16+1),
                    gradient_norm=info['gradient_norm'],rho_actual=info['rho_actual'],sharpness_delta=info['sharpness_delta'],sam_gradient_cosine=info['sam_gradient_cosine'],sam_gradient_relative_change=info['sam_gradient_relative_change'],timestamp=time.time(),pid=os.getpid()));last_progress=time.time()
        torch.cuda.synchronize();training_seconds=time.time()-t0;t0=time.time()
        train_probe,_=evaluate(ema,probe,c['eval_batch']);val,scores=evaluate(ema,splits['val'],c['eval_batch']);eval_seconds=time.time()-t0
        np.savez_compressed(out/'latest_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=scores.numpy(),round=step,power_fraction=1.)
        key=(-val['accuracy'],val['macro_nll'])
        if key<best_key:
            best_key=key;best_round=step;torch.save(payload(step,val),out/'best_checkpoint.pt')
            np.savez_compressed(out/'best_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=scores.numpy(),round=step,power_fraction=1.)
        full_train=None
        if step%8==0:
            full_train,train_scores=evaluate(ema,splits['train'],c['eval_batch'])
            np.savez_compressed(out/f'train_scores_round{step}.npz',ids=ids,labels=y.numpy(),scores=train_scores.numpy(),power_fraction=1.)
        entry=dict(round=step,recipe=recipe,full49=True,power_fraction=1.,train_probe=train_probe,val=val,train_full=full_train,
            training_seconds=training_seconds,probe_and_val_seconds=eval_seconds,elapsed_seconds=time.time()-started,
            clean_loss=total/64,sharpness_delta_mean=None if not sharpness else sum(sharpness)/len(sharpness),
            sam_gradient_cosine_mean=None if not cosines else sum(cosines)/len(cosines),sam_gradient_relative_change_mean=None if not relative_changes else sum(relative_changes)/len(relative_changes),
            first_gradient_norm_mean=sum(first_norms)/len(first_norms),gradient_norm=info['gradient_norm'],gradients=info['gradients'],
            rho_actual_last=info['rho_actual'],restored_before_optimizer=info['restored_before_optimizer'],
            learning_rates={g['group_name']:g['lr'] for g in optimizer.param_groups},best_round=best_round,optimized_images=step*1024,
            forward_backward_passes_per_batch=2 if recipe=='sam' else 1)
        history.append(entry);save(out/'history.json',history);torch.save(payload(step,val),out/'last_checkpoint.pt')
        save(out/'status.json',dict(state='training',pid=os.getpid(),round=step,best_round=best_round));print(json.dumps(entry),flush=True)
    selected=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(selected['model']);del selected
    train,_=evaluate(ema,splits['train'],c['eval_batch']);val,_=evaluate(ema,splits['val'],c['eval_batch'])
    for split in ['train','val']:save_route_audit(ema,splits[split],out/f'selected_{split}_routes.npz')
    result=dict(recipe=recipe,seed=17,run=str(out),rounds=len(history),full49_rounds=len(history),selected_round=best_round,selected_power_fraction=1.,
        train=train,val=val,checkpoint_sha256=sha(out/'best_checkpoint.pt'),optimized_images=len(history)*1024,source_commit=commit,test_role=c['test_role'])
    save(out/'result.json',result);save(out/'status.json',dict(state='validation_selected',**result))


def evaluate_selected(c,directory,profile):
    from .mixup_curriculum import test_candidate
    commit,hashes=sources(profile)
    os.environ['T10_MIXUP_SOURCE_COMMIT']=commit
    try:metrics=test_candidate(c,directory,profile)
    finally:os.environ.pop('T10_MIXUP_SOURCE_COMMIT',None)
    result=json.loads((directory/'test_result.json').read_text())
    result.update(refinement_source_commit=commit,refinement_source_files=hashes)
    save(directory/'test_result.json',result)
    return metrics


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
            directory=out/(recipe+'_s17');log=directory.with_suffix('.log').open('x')
            proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.sharpness_refine','--profile',str(profile),'--out',str(directory),'--worker','--recipe',recipe],env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[proc.pid]=(proc,log,directory);children.append(proc.pid)
        while running:
            for pid,(proc,log,directory) in list(running.items()):
                if proc.poll() is None:continue
                log.close();del running[pid]
                if proc.returncode:raise RuntimeError('Worker failed '+str(directory))
                row=json.loads((directory/'result.json').read_text());done.append(row)
            # Use GPU1 only after its training worker exits; SAM may continue on GPU3.
            gpu1_training=any(v[2].name=='hard_label_s17' for v in running.values())
            if not gpu1_training:
                os.environ['CUDA_VISIBLE_DEVICES']=c['gpus']['1']
                for row in done:
                    if 'test' not in row:row['test']=evaluate_selected(c,Path(row['run']),profile)
            save(out/'status.json',dict(state='training',scheduler_pid=os.getpid(),running={str(k):str(v[2]) for k,v in running.items()},completed=done))
            if running:time.sleep(10)
        save(out/'all_candidates_lock.json',dict(candidates=done,checkpoint_selection='VAL only, original full49',test_timing='Each selected checkpoint tested as soon as GPU1 is free; per-candidate lock precedes inference'))
        os.environ['CUDA_VISIBLE_DEVICES']=c['gpus']['1']
        for row in done:
            if 'test' not in row:row['test']=evaluate_selected(c,Path(row['run']),profile)
        selected=max(done,key=lambda row:(row['test']['accuracy'],row['val']['accuracy'],-row['val']['macro_nll']))
        correct=round(selected['test']['accuracy']*752)
        save(out/'summary.json',dict(role=c['test_role'],candidates=done,selected=selected,test_correct=correct,target_correct=616,pilot_target_met=correct>=616,three_seed_mean_target_established=False))
        save(out/'status.json',dict(state='complete',scheduler_pid=os.getpid(),running={},pilot_target_met=correct>=616))
    finally:
        for proc,log,directory in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        save(out/'release_check.json',dict(child_pids=children,children_reaped=True,authorized_gpus=c['gpus'],timestamp=time.time()))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--recipe',choices=['hard_label','sam']);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();c=json.loads(a.profile.read_text());validate(c)
    try:
        if a.smoke:gradient_probe(c,a.out,a.profile)
        elif a.worker:worker(c,a.out,a.profile,a.recipe)
        else:supervisor(c,a.out,a.profile)
    except BaseException as error:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),scheduler_pid=os.getpid(),error=repr(error)))
        raise


if __name__=='__main__':main()
