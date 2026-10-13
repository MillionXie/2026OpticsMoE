"""Authorized common200 continuation; preserve sealed common100 artifacts."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import medicinal_pilot as pilot
import caltech_selected_finalize as parent_tools

o, r, torch, np = pilot.o, pilot.r, pilot.torch, pilot.np
HERE, ROOT = parent_tools.HERE, parent_tools.ROOT
DEPTHS, KEYS, GPUS = parent_tools.DEPTHS, parent_tools.KEYS, parent_tools.GPUS
read, sha = parent_tools.read, parent_tools.sha
SCOPE = ('Validation-screened Caltech20 seed17, common200: exact raw/EMA/AdamW/plateau '
         'state continuation from100; unchanged regularization; previous test observed, exploratory development')


def state_digest(value):
    """Device-independent identity of nested optimizer/model/scheduler values."""
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            t = item.detach().cpu().contiguous()
            digest.update(str((str(t.dtype), tuple(t.shape))).encode())
            digest.update(t.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for v in item: visit(v)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def extension_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.update(maximum_epochs=200, minimum_epochs=200, requested_epochs=200,
               early_stopping=False, maximum_seconds=36000)
    cfg['selection'] = 'First minimum validation EMA balanced NLL across1-200, parent best remains eligible'
    cfg['extension_budget'] = dict(parent_epochs=100, additional_epochs=100, total_epochs=200,
                                   reset_optimizer=False, reset_scheduler=False, reset_ema=False)
    return cfg


def source_identity(parent):
    sources = dict(read(parent/'metadata.json')['sources'])
    for name in ('caltech_selected_extend.py', 'caltech_selected_extend_queue.py',
                 'caltech_selected_extend_plan.json', 'caltech_selected_finalize.py',
                 'caltech20_finalize.py', 'audit_medicinal_pilot.py'):
        sources[(HERE/name).relative_to(ROOT).as_posix()] = sha(HERE/name)
    return sources


def verify_sources(sources, commit):
    for path, digest in sources.items():
        assert sha(ROOT/path) == digest, path
        assert hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)).hexdigest() == digest, path


def verify_parent(parent, data):
    fit = parent_tools.verify_arm(parent)
    manifest = parent_tools.verify_data(data)
    assert fit['data_sha256'] == manifest['cache_sha256'] and fit['epochs_completed'] == 100
    last = torch.load(parent/'last_checkpoint.pt', map_location='cpu', weights_only=False)
    assert last['epoch'] == 100 and last['config'] == fit['config'] and last['sources'] == fit['sources']
    assert last['data_sha256'] == manifest['cache_sha256']
    assert last['scheduler']['last_epoch'] == 100
    assert len(last['optimizer']['state']) == 1+5*fit['depth']
    steps = (100*((sum(manifest['supports']['train'])+15)//16))
    assert all(float(v['step']) == steps for v in last['optimizer']['state'].values())
    assert last['orders'] == fit['orders'] and last['transforms'] == fit['transforms']
    assert last['initial_parameter_sha256'] == fit['initial_parameter_sha256']
    for key in ('model', 'ema'):
        assert all(bool(torch.isfinite(v).all()) for v in last[key].values())
    verify_sources(fit['sources'], read(parent/'metadata.json')['git_commit'])
    return fit, last, manifest


def restore_training(model, ema, optimizer, scheduler, checkpoint):
    model.load_state_dict(checkpoint['model'], strict=True)
    ema.load_state_dict(checkpoint['ema'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    actual = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    identity = {k:state_digest(v) for k,v in actual.items()}
    assert identity == {k:state_digest(checkpoint[k]) for k in actual}
    return identity


def warm_gate(raw, data, weights, profile, cfg, depth, micro):
    """Same restored E100 raw state; real E101 MixUp objective vs serial micro1."""
    model = pilot.execution.build(depth, profile, cfg, batched=False)
    model.load_state_dict(raw, strict=True); identity = state_digest(model.state_dict())
    idx = torch.tensor([int((data[1] == k).nonzero()[0]) for k in range(8)])
    small = (data[0][idx], data[1][idx], data[2][idx.numpy()])
    theta = o.b.affine_parameters(8, cfg['seed'], 101, cfg['augmentation'])
    mixed, target, _ = pilot.mixup_batch(small, torch.arange(8), theta, cfg, 101, 0)
    def compute(batch):
        model.zero_grad(set_to_none=True); probabilities=[]
        for start in range(0,8,batch):
            end=min(start+batch,8); p,c=o.forward(model,mixed[start:end])
            (pilot.soft_objective(model,p,c,target[start:end],weights,cfg)*(end-start)/8).backward()
            probabilities.append(p.detach())
        return torch.cat(probabilities), {n:p.grad.detach().clone() for n,p in model.named_parameters()}
    reference, gradients=compute(1);pilot.execution.enable_expert_batching(model)
    proposed, other=compute(micro)
    error=float((reference-proposed).abs().max())
    relative=max(float((other[n]-g).norm()/g.norm().clamp_min(1e-12)) for n,g in gradients.items())
    cosine=min(float(torch.nn.functional.cosine_similarity(other[n].flatten(),g.flatten(),dim=0)) for n,g in gradients.items())
    finite=all(bool(torch.isfinite(g).all()) and float(g.norm())>0 for g in other.values())
    assert error<=1e-6 and relative<=.02 and cosine>=.999 and finite, (error,relative,cosine)
    assert state_digest(model.state_dict())==identity and not any(hasattr(m,'residual_rho') for n,m in model.named_modules() if 'router' in n)
    result=dict(passed=True,depth=depth,rho=cfg['residual_rho'],microbatch=micro,expert_batched=True,
        probability_max_error=error,gradient_max_relative_l2=relative,gradient_min_cosine=cosine,
        gradients_finite_nonzero=finite,restored_raw_parameter_sha256=identity,
        scope='Restored E100 raw state, stratified8 train samples, E101 MixUp; no test or full parent re-evaluation')
    del model,reference,proposed,gradients,other,mixed,target
    gc.collect();torch.cuda.empty_cache()
    return result


def train(args):
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu and args.gpu in GPUS
    fit,last,manifest=verify_parent(args.parent,args.data)
    cfg=extension_config(fit['config']);depth=fit['depth'];micro=cfg['microbatch']
    assert depth in DEPTHS and micro==(4 if depth in (2,4) else 1)
    args.out.mkdir(parents=True,exist_ok=False)
    sources=source_identity(args.parent)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();verify_sources(sources,commit)
    o.f.gpu_budget.ALLOWED.update(GPUS)
    policy=o.f.gpu_budget.configure(torch,args.gpu,shared=False,memory_gib=3);torch.set_num_threads(4)
    profile=read(parent_tools.PROFILE)
    data,val=[pilot.load_fit_data(args.data,s,manifest) for s in ('train','val')]
    weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
    gate=warm_gate(last['model'],data,weights,profile,cfg,depth,micro);r.save(args.out/'smoke.json',gate)
    r.setseed(cfg['seed']);model=pilot.execution.build(depth,profile,cfg,batched=True)
    ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',**cfg['scheduler'])
    restored=restore_training(model,ema,optimizer,scheduler,last)
    receipt=dict(parent=str(args.parent.resolve()),parent_epoch=100,first_new_epoch=101,total_epochs=200,
        parent_last_checkpoint_sha256=sha(args.parent/'last_checkpoint.pt'),parent_best_checkpoint_sha256=sha(args.parent/'best_checkpoint.pt'),
        restored_state_sha256=restored,optimizer_reset=False,scheduler_reset=False,ema_reset=False,
        restored_lr=optimizer.param_groups[0]['lr'],restored_scheduler_last_epoch=scheduler.last_epoch)
    r.save(args.out/'resume_receipt.json',receipt);r.save(args.out/'config.json',cfg)
    r.save(args.out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,git_commit=commit,sources=sources,
        gpu_policy=policy,parent=str(args.parent.resolve()),parent_training_git_commit=read(args.parent/'metadata.json')['git_commit'],
        data_sha256=sha(args.data),manifest_sha256=sha(args.data.parent/'data_manifest.json'),time=r.now(),scope=SCOPE))
    history=copy.deepcopy(read(args.parent/'history.json'));gradients=copy.deepcopy(read(args.parent/'gradients.json'))
    orders,transforms=list(last['orders']),list(last['transforms']);initial_sha=fit['initial_parameter_sha256']
    best_checkpoint=torch.load(args.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    best_checkpoint.update(config=cfg,sources=sources);r.save_torch(args.out/'best_checkpoint.pt',best_checkpoint)
    for name in ('train_predictions.csv','val_predictions.csv'):shutil.copyfile(args.parent/name,args.out/name)
    best=best_checkpoint['validation']['balanced_nll'];start=time.monotonic();reason='maximum_epochs'
    for epoch in range(101,201):
        if time.monotonic()-start>=cfg['maximum_seconds']:reason='time_budget';break
        tick=time.monotonic();model.train()
        order=r.epoch_order(cfg['seed'],epoch,len(data[1]));theta=o.b.affine_parameters(len(data[1]),cfg['seed'],epoch,cfg['augmentation'])
        orders.append(r.sha_tensor(order));transforms.append(r.sha_tensor(theta));mix_hash=hashlib.sha256();total=0.
        for batch_index,idx in enumerate(order.split(16)):
            optimizer.zero_grad(set_to_none=True)
            mixed,target,identity=pilot.mixup_batch(data,idx,theta,cfg,epoch,batch_index);mix_hash.update(identity.encode())
            for offset in range(0,len(idx),micro):
                size=min(micro,len(idx)-offset);probability,capture=o.forward(model,mixed[offset:offset+size])
                loss=pilot.soft_objective(model,probability,capture,target[offset:offset+size],weights,cfg)
                assert bool(torch.isfinite(loss));(loss*size/len(idx)).backward();total+=float(loss.detach())*size
            if batch_index==0:
                norms={n:float(p.grad.norm()) for n,p in model.named_parameters()}
                assert all(np.isfinite(v) and v>0 for v in norms.values());gradients.append(dict(epoch=epoch,norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-cfg['ema_decay'])
        vm,rows=pilot.evaluate(ema,val,micro)
        transforms[-1]=hashlib.sha256((transforms[-1]+mix_hash.hexdigest()).encode()).hexdigest()
        entry=dict(epoch=epoch,train_loss=total/len(data[1]),lr=optimizer.param_groups[0]['lr'],val=vm,
            mixup_contract=history[0]['mixup_contract'],mixup_sha256=mix_hash.hexdigest(),stage='continuation101-200')
        if epoch%5==0:entry['train']=pilot.evaluate(ema,data,micro)[0]
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];r.save_torch(args.out/'best_checkpoint.pt',dict(model=ema.state_dict(),epoch=epoch,depth=depth,
                config=cfg,sources=sources,data_sha256=manifest['cache_sha256'],validation=vm,selected_state_kind='ema',selection_split='val'))
            r.csvwrite(args.out/'val_predictions.csv',rows)
        scheduler.step(vm['balanced_nll'])
        entry.update(seconds=history[99]['seconds']+time.monotonic()-start,continuation_seconds=time.monotonic()-start,epoch_seconds=time.monotonic()-tick)
        history.append(entry);r.save(args.out/'history.json',history);r.save(args.out/'gradients.json',gradients)
        r.save_torch(args.out/'last_checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,config=cfg,sources=sources,depth=depth,orders=orders,transforms=transforms,initial_parameter_sha256=initial_sha,data_sha256=manifest['cache_sha256']))
        status=dict(state='training',epoch=epoch,depth=depth,rho=fit['rho'],val_accuracy=vm['accuracy'],val_macro_recall=vm['balanced_accuracy'],
            val_balanced_nll=vm['balanced_nll'],best_val_balanced_nll=best,epoch_seconds=entry['epoch_seconds'],seconds=entry['continuation_seconds'],test_evaluations=0)
        if 'train' in entry:status['train_val_gap_pp']=100*(entry['train']['accuracy']-vm['accuracy'])
        r.save(args.out/'status.json',status);print(json.dumps(status),flush=True)
    assert len(history)>100,'No continuation epoch completed'
    if 'train' not in history[-1]:history[-1]['train']=pilot.evaluate(ema,data,micro)[0];r.save(args.out/'history.json',history)
    checkpoint=torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    selected=min(history,key=lambda h:h['val']['balanced_nll']);assert selected['epoch']==checkpoint['epoch']
    if checkpoint['epoch']<=100:tm=fit['metrics']['train']
    else:
        ema.load_state_dict(checkpoint['model']);tm,rows=pilot.evaluate(ema,data,micro);r.csvwrite(args.out/'train_predictions.csv',rows)
    result=dict(depth=depth,rho=fit['rho'],candidate='base',metrics=dict(train=tm,val=checkpoint['validation']),selected_epoch=checkpoint['epoch'],
        epochs_completed=len(history),stop_reason=reason,seconds=fit['seconds']+time.monotonic()-start,continuation_seconds=time.monotonic()-start,
        history_sha256=sha(args.out/'history.json'),checkpoint_sha256=sha(args.out/'best_checkpoint.pt'),selected_model_state_sha256=state_digest(checkpoint['model']),
        sources=sources,config=cfg,data_sha256=manifest['cache_sha256'],initial_parameter_sha256=initial_sha,orders=orders,transforms=transforms,
        parameters=fit['parameters'],test_evaluations=0,test_used_for_selection=False,selection_split='val',scope=SCOPE)
    r.save(args.out/'fit_result.json',result);r.save(args.out/'status.json',dict(state='fit_complete',epochs=len(history),stop_reason=reason,test_evaluations=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('parent','out','data'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',choices=GPUS,required=True)
    train(p.parse_args())
