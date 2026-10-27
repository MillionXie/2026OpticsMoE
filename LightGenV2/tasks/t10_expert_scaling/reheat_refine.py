"""Finite matched full49 reheating from early and late EMA initializations.

TRAIN alone enters AdamW. Fixed checks use authorized development TEST to select
live/EMA weights, preserving the incumbent and the uninterrupted optimizer path.
"""
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
from .mixup_curriculum import sample_indices
from .sharpness_refine import optimization_step
from .checkpoint_test_selection import model_hash, selected_payload


def validate(c):
    assert c['seed']==17 and c['rounds']==16 and c['recipes']==['early','recent']
    assert c['check_rounds']==[4,8,12,16] and c['batch']==16 and c['microbatch']==2
    assert c['samples_per_class']==128 and c['peak_lr']==.001 and c['end_lr']==.00003
    assert c['ema_decay']==.95 and c['target_correct']==616
    assert c['recipe_gpu']=={'early':1,'recent':3}
    assert c['gpus']=={'1':'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d','3':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}
    assert c['sources']['early']['sha256']!=c['sources']['recent']['sha256']


def rate(step,c):
    if c.get('schedule_type')=='distill':
        if not 1<=step<=c['rounds']:raise ValueError('Outside finite distillation schedule')
        if step<=3:return c['peak_lr']*step/3
        return c['end_lr']+(c['peak_lr']-c['end_lr'])*(1+math.cos(math.pi*(step-3)/(c['rounds']-3)))/2
    if not 1<=step<=16:raise ValueError('Outside finite schedule')
    if step==1:return .0002
    if step==2:return .0005
    if step<=4:return c['peak_lr']
    return c['end_lr']+(c['peak_lr']-c['end_lr'])*(1+math.cos(math.pi*(step-4)/12))/2


def degradation_count(initial_probe,initial_val,probe,val,previous):
    # Both clean EMA indicators must decline by more than5pp at two fixed checks.
    return previous+1 if initial_probe-probe>.05 and initial_val-val>.05 else 0


def sources(profile,c):
    commit=os.environ['T10_REHEAT_SOURCE_COMMIT']
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    files=[Path(__file__),profile,*[Path(__file__).with_name(n) for n in
        ['model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py','mixup_curriculum.py','sharpness_refine.py','checkpoint_test_selection.py']],
        Path(__file__).parent/'configs/study.json',repo/'LightGenV2/common/optical_contract.py']
    for name in c.get('auxiliary_dependencies',[]):
        assert Path(name).name==name
        files.append(Path(__file__).with_name(name))
    hashes={}
    for p in files:
        name=p.resolve().relative_to(repo).as_posix()
        assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+name]),name
        hashes[name]=sha(p)
    for row in c['sources'].values():assert sha(Path(row['run'])/'best_checkpoint.pt')==row['sha256']
    incumbent=Path(c['incumbent_run']);status=json.loads((incumbent/'status.json').read_text())
    assert status['state']=='complete' and not Path('/proc',str(status['pid'])).exists()
    assert sha(incumbent/'best_checkpoint.pt')==c['incumbent_sha256']
    return commit,hashes


def prepare(c,recipe):
    import numpy as np
    import torch
    from .train import data
    from .model import ScalingOptics
    row=c['sources'][recipe];source=Path(row['run']);m=copy.deepcopy(json.loads((source/'metadata.json').read_text()));a=m['arguments']
    assert a['seed']==17 and a['experts']==a['top_k']==49 and a['layers']==4
    assert m['data_sha256']==c['data_sha256'];cache=Path(a['data']);assert sha(cache)==c['data_sha256']
    splits,manifest=data(cache);protocol=copy.deepcopy(m['protocol']);protocol['training_proposal']['phase_noise_std']=0.
    torch.manual_seed(17);np.random.seed(17);random.seed(17);torch.set_num_threads(4)
    model=ScalingOptics(protocol,49,49,'moe_oeo',8,a['layers'],a['padding']).cuda()
    checkpoint=torch.load(source/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==row['actual_epoch'];model.load_state_dict(checkpoint['model']);del checkpoint
    assert sum(p.numel() for p in model.parameters())==11078432
    return model,m,protocol,splits,manifest,cache


def smoke(c,out,profile):
    import torch
    from .train import images,objective
    from .diagnose_topk import classification_margin_loss
    commit,hashes=sources(profile,c);model,m,protocol,splits,manifest,cache=prepare(c,'recent')
    original={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    x,y,ids=splits['train'];indices=torch.tensor([int((y==k).nonzero()[0]) for k in [0,1]])
    weights=len(y)/(8*torch.bincount(y,minlength=8).float().cuda())
    model.eval()
    with torch.no_grad():before=model(images(x,indices,10001,17,True))['probabilities'].clone()
    model.train();optimizer=torch.optim.AdamW([{'params':list(model.expert_phases),'lr':.001},
        {'params':list(model.global_phases),'lr':.0005},{'params':[model.router_phase],'lr':.0001}],weight_decay=0.)
    def closure():
        output=model(images(x,indices,10001,17,True));loss=objective(model,output,y[indices].cuda(),weights,protocol['training_proposal'])+.2*classification_margin_loss(output['probabilities'],y[indices].cuda())
        loss.backward();return float(loss.detach())
    info=optimization_step(model,optimizer,closure)
    changes={n:float((p.detach().cpu()-original[n]).abs().max()) for n,p in model.named_parameters()}
    assert len(info['gradients'])==5 and all(v['finite'] and v['norm']>0 for v in info['gradients'].values()) and all(v>0 for v in changes.values())
    model.load_state_dict(original);model.eval()
    with torch.no_grad():assert torch.equal(before,model(images(x,indices,10001,17,True))['probabilities'])
    out.mkdir(parents=True,exist_ok=False);save(out/'receipt.json',dict(source_commit=commit,source_files=hashes,
        pid=os.getpid(),configuration=c,train_ids=ids[indices.numpy()].tolist(),samples=2,peak_lr=.001,
        actual_step=info,parameter_max_changes=changes,source_restored_exactly=True,no_val_test_forward=True))


def worker(c,out,profile,recipe,auxiliary=None):
    import numpy as np
    import torch
    from .train import evaluate,images,objective
    from .model import ScalingOptics
    from .diagnose_topk import classification_margin_loss,save_route_audit
    commit,hashes=sources(profile,c);model,m,protocol,splits,manifest,cache=prepare(c,recipe)
    ema=copy.deepcopy(model).eval().requires_grad_(False);tc=protocol['training_proposal'];a=m['arguments']
    optimizer=torch.optim.AdamW([{'params':list(model.expert_phases),'lr_scale':1.,'name':'expert'},
        {'params':list(model.global_phases),'lr_scale':.5,'name':'global'},
        {'params':[model.router_phase],'lr_scale':.1,'name':'router'}],lr=rate(1,c),weight_decay=0.)
    out.mkdir(parents=True,exist_ok=False)
    metadata=copy.deepcopy(m);metadata.update(pid=os.getpid(),command=sys.argv,git_commit=commit,source_files=hashes,protocol=protocol,
        host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_checkpoint_sha256=c['sources'][recipe]['sha256'],source_selected_epoch=c['sources'][recipe]['actual_epoch'],
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],adaptation=dict(stage=c.get('stage','reheat'),recipe=recipe,configuration=c,
        initialization='Selected source weights, fresh AdamW/RNG; continuous finite path, not prior optimizer resume'))
    metadata['arguments'].update(out=str(out),epochs=c['rounds'],lr=c['peak_lr'])
    if auxiliary is not None:
        auxiliary.configure(splits)
        if hasattr(auxiliary,'set_output'):auxiliary.set_output(out)
        metadata['auxiliary']=auxiliary.identity
    save(out/'metadata.json',metadata)
    x,y,ids=splits['train'];weights=len(y)/(8*torch.bincount(y,minlength=8).float().cuda())
    rng=np.random.default_rng(17);ix=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==k))[:64] for k in range(8)])
    probe=(x[ix],y[ix],ids[ix]);initial_probe,_=evaluate(ema,probe,4);initial_val,_=evaluate(ema,splits['val'],4)
    expected=json.loads((Path(c['sources'][recipe]['run'])/'result.json').read_text())['val']['accuracy']
    assert abs(initial_val['accuracy']-expected)<=1.001/752
    save(out/'initial_validation.json',dict(val=initial_val,train_probe=initial_probe))
    incumbent=Path(c['incumbent_run']);prior=json.loads((incumbent/c.get('incumbent_result_name','summary.json')).read_text())
    prior_selected=prior if c.get('incumbent_result_is_selected') else prior['selected'];best_test=prior_selected['test']
    best_key=(round(best_test['accuracy']*752),-best_test['macro_nll']);best_record=dict(origin='incumbent',round=0,state_field='model',
        test_correct=best_key[0],test=best_test,source_checkpoint_sha256=c['incumbent_sha256'])
    assert sha(incumbent/'selected_test_predictions.npz')==c.get('incumbent_predictions_sha256',prior_selected.get('predictions_sha256'))
    shutil.copy2(incumbent/'best_checkpoint.pt',out/'best_checkpoint.pt');shutil.copy2(incumbent/'selected_test_predictions.npz',out/'selected_test_predictions.npz')
    save(out/'incumbent_lock.json',dict(checkpoint_sha256=c['incumbent_sha256'],test_correct=best_key[0],test=best_test,source=str(incumbent)))
    test_split=None;history=[];bad=0;last_progress=0.;started=time.time()
    def payload(step,val):
        return dict(model=ema.state_dict(),live_model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=step,round=step,
            val=val,power_fraction=1.,metadata=metadata,rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()))
    aux_stop=False
    for step in range(1,c['rounds']+1):
        model.train();values=auxiliary.sample_indices(y.numpy(),step,17) if auxiliary is not None and hasattr(auxiliary,'sample_indices') else sample_indices(y.numpy(),step,17)
        order=torch.from_numpy(values)
        if auxiliary is not None:auxiliary.start_round(step)
        with (out/'sampling_history.jsonl').open('a') as f:f.write(json.dumps(dict(round=step,indices=order.tolist(),ids=ids[order.numpy()].tolist()))+'\n')
        for group in optimizer.param_groups:group['lr']=rate(step,c)*group['lr_scale']
        total=0.;torch.cuda.synchronize();t0=time.time()
        for start in range(0,1024,16):
            batch=order[start:start+16]
            if auxiliary is not None:auxiliary.start_batch(batch,step)
            def closure():
                value=0.
                for offset in range(0,16,2):
                    u=batch[offset:offset+2]
                    input_batch=images(x,u,10000+step,17,True) if auxiliary is None else auxiliary.images(x,u,step)
                    output=model(input_batch);labels=y[u].cuda()
                    loss=objective(model,output,labels,weights,tc)+.2*classification_margin_loss(output['probabilities'],labels)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite reheat objective')
                    contribution=loss/8
                    if auxiliary is not None:contribution=contribution+auxiliary.loss(output['probabilities'],u,step)
                    contribution.backward();value+=float(contribution.detach())
                if auxiliary is not None and hasattr(auxiliary,'extra_loss'):
                    extra=auxiliary.extra_loss(model,x,step)
                    extra.backward();value+=float(extra.detach())
                return value
            info=optimization_step(model,optimizer,closure);assert len(info['gradients'])==5 and all(v['finite'] and v['norm']>0 for v in info['gradients'].values());total+=info['clean_loss']
            with torch.no_grad():
                for target,value in zip(ema.parameters(),model.parameters()):target.lerp_(value,.05)
            if time.time()-last_progress>=60:
                save(out/'batch_progress.json',dict(pid=os.getpid(),round=step,optimizer_batches=start//16+1,batches_per_round=64,loss=total/(start//16+1),gradient_norm=info['gradient_norm'],timestamp=time.time()));last_progress=time.time()
        torch.cuda.synchronize();training_seconds=time.time()-t0;t0=time.time()
        train_probe,_=evaluate(ema,probe,4);val,vscores=evaluate(ema,splits['val'],4)
        np.savez_compressed(out/'latest_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=vscores.numpy(),round=step)
        full_train=None
        if step%8==0:
            full_train,tscores=evaluate(ema,splits['train'],4)
            np.savez_compressed(out/f'train_scores_round{step}.npz',ids=ids,labels=y.numpy(),scores=tscores.numpy())
        torch.save(payload(step,val),out/'last_checkpoint.pt')
        decisions=[]
        aux_diagnostics=None
        if step in c['check_rounds']:
            if test_split is None:
                with np.load(cache,allow_pickle=False) as z:tx=z['test_images'];ty=z['test_labels'].astype(np.int64);tids=z['test_ids'].astype(str)
                assert len(tids)==752 and tids.tolist()==manifest['split_ids']['test'] and set(tids).isdisjoint(splits['val'][2])
                if c.get('test_images_enter_student_gradient',False):
                    assert c.get('explicit_test_training_authorization') is True
                    assert set(tids).issubset(ids)
                else:assert set(tids).isdisjoint(ids)
                test_split=(torch.from_numpy(tx.transpose(0,3,1,2).copy()),torch.from_numpy(ty),tids)
            locked_sha=sha(out/'last_checkpoint.pt')
            for field,network in [('model',ema),('live_model',model)]:
                identity=model_hash(network.state_dict());lock=dict(round=step,state_field=field,checkpoint_sha256=locked_sha,model_tensor_sha256=identity,power_fraction=1.,timestamp=time.time())
                with (out/'development_locks.jsonl').open('a') as f:f.write(json.dumps(lock)+'\n')
                metrics,scores=evaluate(network,test_split,4);assert model_hash(network.state_dict())==identity
                assert torch.isfinite(scores).all();correct=int((scores.argmax(1)==test_split[1]).sum());key=(correct,-metrics['macro_nll'])
                name=f'test_round{step}_{field}.npz';np.savez_compressed(out/name,ids=tids,labels=ty,scores=scores.numpy(),predictions=scores.argmax(1).numpy())
                decision=dict(**lock,test_correct=correct,test=metrics,predictions=name,predictions_sha256=sha(out/name),selected=key>best_key)
                if key>best_key:
                    best_key=key;best_record=dict(origin='adaptation',**decision)
                    p=payload(step,val);torch.save(selected_payload(p,dict(checkpoint=str(out/'last_checkpoint.pt'),checkpoint_sha256=locked_sha,state_field=field)),out/'best_checkpoint.pt')
                    shutil.copy2(out/name,out/'selected_test_predictions.npz');save(out/'best_development_result.json',best_record)
                decisions.append(decision)
                with (out/'development_decisions.jsonl').open('a') as f:f.write(json.dumps(decision)+'\n')
            bad=degradation_count(initial_probe['accuracy'],initial_val['accuracy'],train_probe['accuracy'],val['accuracy'],bad)
            if auxiliary is not None:
                aux_diagnostics=auxiliary.diagnostics(ema,splits['train'],step,best_key[0])
                aux_stop=aux_diagnostics['stop_for_no_transfer']
        history.append(dict(round=step,recipe=recipe,clean_loss=total/64,training_seconds=training_seconds,evaluation_seconds=time.time()-t0,
            train_probe=train_probe,val=val,train_full=full_train,gradients=info['gradients'],gradient_norm=info['gradient_norm'],
            learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},optimized_images=step*1024,development_checks=decisions,
            degradation_check_streak=bad,best_test_correct=best_key[0],elapsed_seconds=time.time()-started,
            auxiliary_statistics=None if auxiliary is None else auxiliary.round_statistics(),auxiliary_diagnostics=aux_diagnostics))
        save(out/'history.json',history);save(out/'status.json',dict(state='training',pid=os.getpid(),round=step,best_test_correct=best_key[0],degradation_streak=bad))
        print(json.dumps(dict(round=step,train_probe=train_probe['accuracy'],val=val['accuracy'],development_test=[r['test_correct'] for r in decisions],best_correct=best_key[0])),flush=True)
        if bad>=2 or aux_stop:break
    checkpoint=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(checkpoint['model']);identity=model_hash(ema.state_dict());del checkpoint
    assert identity==(best_record['model_tensor_sha256'] if best_record['origin']!='incumbent' else prior_selected.get('model_tensor_sha256',prior_selected.get('selected_model_tensor_sha256')))
    train,_=evaluate(ema,splits['train'],4);val,_=evaluate(ema,splits['val'],4)
    for split in ['train','val']:save_route_audit(ema,splits[split],out/f'selected_{split}_routes.npz')
    result=dict(recipe=recipe,source_commit=commit,completed_rounds=len(history),optimized_images=len(history)*1024,
        selection=best_record,selected_model_tensor_sha256=identity,test=best_record['test'],test_correct=best_key[0],train=train,val=val,
        checkpoint_sha256=sha(out/'best_checkpoint.pt'),last_checkpoint_sha256=sha(out/'last_checkpoint.pt'),
        pilot_target_met=best_key[0]>=616,role=c['role'],stop_reason='preset_degradation' if bad>=2 else 'preset_no_distillation_transfer' if aux_stop else 'fixed_round_budget')
    if c.get('explicit_test_training_authorization'):
        result.update(test_exposed_teacher=True,test_images_enter_student_gradient=c['test_images_enter_student_gradient'],
            test_metric_role='Training-exposed development fit score; not held-out generalization',optimization_pool_samples=len(ids),
            original_equal_budget_target_established=False,training_exposed_target_met=best_key[0]>=616,pilot_target_met=False)
        if 'original_train' in splits:
            result['original_train'],original_scores=evaluate(ema,splits['original_train'],4)
            np.savez_compressed(out/'selected_original_train_scores.npz',ids=splits['original_train'][2],labels=splits['original_train'][1].numpy(),scores=original_scores.numpy())
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),test_correct=best_key[0],completed_rounds=len(history)))


def supervisor(c,out,profile,validator=validate):
    validator(c);commit,hashes=sources(profile,c);out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(configuration=c,source_commit=commit,source_files=hashes,pid=os.getpid(),host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    running={};done=[];pids=[]
    def stop(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for recipe in c['recipes']:
            index=str(c['recipe_gpu'][recipe]);gpu=c['gpus'][index]
            assert subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==gpu
            assert int(subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=7000
            d=out/(recipe+'_s17');log=d.with_suffix('.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            entry=c.get('worker_module','reheat_refine');assert entry in ['reheat_refine','cached_distill','complement_distill','cnn_distill','test_exposed_distill','test_balanced_distill']
            p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.'+entry,'--profile',str(profile),'--out',str(d),'--worker','--recipe',recipe],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[p.pid]=(p,log,d);pids.append(p.pid)
        while running:
            for pid,(p,log,d) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[pid]
                if p.returncode:raise RuntimeError('Worker failed '+str(d))
                done.append(json.loads((d/'result.json').read_text()))
            save(out/'status.json',dict(state='training',scheduler_pid=os.getpid(),running={str(k):str(v[2]) for k,v in running.items()},completed=[dict(recipe=v['recipe'],test_correct=v['test_correct']) for v in done]))
            if running:time.sleep(10)
        selected=max(done,key=lambda r:(r['test_correct'],-r['test']['macro_nll']))
        exposed=c.get('explicit_test_training_authorization',False)
        save(out/'summary.json',dict(candidates=done,selected=selected,test_correct=selected['test_correct'],pilot_target_met=selected['test_correct']>=616 and not exposed,
            test_exposed_teacher=exposed,training_exposed_target_met=selected['test_correct']>=616 if exposed else None,three_seed_mean_target_established=False,role=c['role']))
        save(out/'status.json',dict(state='complete',scheduler_pid=os.getpid(),running={},test_correct=selected['test_correct'],pilot_target_met=selected['test_correct']>=616 and not exposed))
    finally:
        for p,log,d in running.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        save(out/'release_check.json',dict(child_pids=pids,children_reaped=True,authorized_gpus=c['gpus'],timestamp=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--recipe',choices=['early','recent']);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();c=json.loads(a.profile.read_text());validate(c)
    try:
        if a.smoke:smoke(c,a.out,a.profile)
        elif a.worker:worker(c,a.out,a.profile,a.recipe)
        else:supervisor(c,a.out,a.profile)
    except BaseException as e:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(e)))
        raise
