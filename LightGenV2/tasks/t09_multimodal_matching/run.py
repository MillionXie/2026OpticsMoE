"""Validation-only text-encoding pilot; no test-set loader is implemented."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import TextEncoder, OpticalOEO, encode, loss, enlarge_tiles
from .prepare import tokens, save, digest

ARCHS = ['moe', 'd2nn']


def state_sha(module):
    h = hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setseed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def load_data(root, split, vocab, device):
    assert split in ['train','val'], 'This pilot never evaluates the reserved test set'
    manifest = json.loads((root/'manifest.json').read_text())
    for filename in [f'{split}_images.npz',f'{split}_questions.json','vocab.json']:
        assert digest((root/filename).read_bytes()) == manifest['files'][filename]
    images = torch.tensor(np.load(root/f'{split}_images.npz')['images'],device=device)
    rows = json.loads((root/f'{split}_questions.json').read_text())
    ids = torch.zeros(len(rows),32,dtype=torch.long,device=device)
    for i,row in enumerate(rows):
        words = tokens(row['question']);assert len(words)<=32
        ids[i,:len(words)] = torch.tensor([vocab.get(w,1) for w in words],device=device)
    return dict(images=images, rows=rows, ids=ids,
                index=torch.tensor([r['image_local'] for r in rows],device=device),
                labels=torch.tensor([r['label'] for r in rows],device=device))


def batches(data, batch, order=None):
    order = torch.arange(len(data['labels']),device=data['labels'].device) if order is None else order
    for chunk in order.split(batch):
        yield chunk, data['images'][data['index'][chunk]],data['ids'][chunk],data['labels'][chunk]


@torch.no_grad()
def evaluate(model, frontend, data, batch, ablation=None):
    model.eval();frontend.eval();preds=[];routes=[];captures=[]
    for chunk,images,ids,labels in batches(data,batch):
        if ablation == 'swap_pair_text':
            ids = data['ids'][chunk ^ 1]
        amplitude = encode(images,frontend(ids))
        output = model(amplitude)
        preds.append(output['probabilities'].cpu());routes.append(output['route_power'].cpu());captures.append(output['capture'].cpu())
    p = torch.cat(preds);y=data['labels'].cpu()
    if ablation == 'swap_pair_text':
        y = 1-y
    q=torch.cat(routes)
    metrics=dict(accuracy=float((p.argmax(1)==y).float().mean()),
                 nll=float(F.nll_loss(p.clamp_min(1e-12).log(),y)),
                 capture=float(torch.cat(captures).mean()),
                 route_mean=q.mean(0).tolist(), route_std=q.std(0).tolist(),
                 route_top_frequency=torch.bincount(q.argmax(1),minlength=4).div(len(q)).tolist())
    return metrics,p.numpy()


def build_models(seed, device):
    return {arch:OpticalOEO(arch,seed).to(device) for arch in ARCHS}


def train_epoch(models,frontend,data,optimizer,batch,epoch,seed):
    for model in models.values():model.train()
    frontend.train(any(p.requires_grad for p in frontend.parameters()))
    generator=torch.Generator(device=data['ids'].device).manual_seed(seed+epoch)
    order=torch.randperm(len(data['labels']),generator=generator,device=data['ids'].device)
    running=0
    for _,images,ids,y in batches(data,batch,order):
        optimizer.zero_grad(set_to_none=True)
        amplitude=encode(images,frontend(ids))
        costs=[loss(model(amplitude),y) for model in models.values()]
        cost=torch.stack(costs).mean()
        assert torch.isfinite(cost), 'Nonfinite loss'
        cost.backward()
        torch.nn.utils.clip_grad_norm_(list(frontend.parameters())+[p for m in models.values() for p in m.parameters()],1.)
        optimizer.step();running+=float(cost.detach())*len(y)
    return running/len(data['labels'])


def metadata(args, out):
    repo=Path(__file__).resolve().parents[3]
    value=dict(config=vars(args).copy(), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
               command=__import__('sys').argv, python=platform.python_version(),torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),pid=os.getpid(),
               cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), test_accessed=False)
    value['config']={k:str(v) if isinstance(v,Path) else v for k,v in value['config'].items()}
    value['sources']={p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}
    value['data_manifest_sha256']=digest((args.data/'manifest.json').read_bytes())
    backend=repo/'LightGenV2/demo_check/pure_optical'
    value['backend_sources']={str(p.relative_to(repo)):digest(p.read_bytes()) for p in
                             [backend/'models.py',backend/'config.json',
                              repo/'LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']}
    save(out/'metadata.json',value)


def smoke(args,train,vocab):
    records=[]
    for mode in ['fixed','fixed_dense','learned']:
        setseed(args.seed);frontend=TextEncoder(len(vocab),mode).cuda()
        images=train['images'][train['index'][:4]];ids=train['ids'][:4];labels=train['labels'][:4]
        coded=frontend(ids);amplitude=encode(images,coded)
        assert coded.shape==(4,32,64) and amplitude.shape==(4,224,224)
        assert torch.allclose(amplitude.square().sum((1,2)),torch.ones(4,device='cuda'),atol=1e-6)
        assert torch.allclose(amplitude[:,112:,112:].square().sum((1,2)),torch.full((4,),.5,device='cuda'),atol=1e-6)
        expanded=enlarge_tiles(amplitude,478)
        assert torch.allclose(expanded[:,239:,239:].square().sum((1,2)),torch.full((4,),.5,device='cuda'),atol=1e-6)
        if mode=='learned':
            padmask=ids.eq(0)
            assert torch.count_nonzero(coded[padmask])==0
        if mode=='fixed_dense':
            assert torch.equal(frontend.codes@frontend.codes.T,32*torch.eye(32,device='cuda'))
            z=coded[:,:,:32]-coded[:,:,32:]
            recovered=(z@frontend.codes.T).argmax(-1)
            assert torch.equal(recovered[ids!=0],ids[ids!=0])
        for arch,model in build_models(args.seed,'cuda').items():
            frontend.zero_grad(set_to_none=True)
            before=state_sha(model)
            output=model(encode(images,frontend(ids)));cost=loss(output,labels);cost.backward()
            gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
            assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
            if arch=='moe':assert all(float(g.norm())>0 for g in model.first_phase.grad)
            opt=torch.optim.Adam(model.parameters(),lr=.001);opt.step()
            assert state_sha(model)!=before
            frozen=copy.deepcopy(frontend).requires_grad_(False).eval();frozen_sha=state_sha(frozen)
            loss(model(encode(images,frozen(ids))),labels).backward()
            assert frozen_sha==state_sha(frozen)
            records.append(dict(mode=mode,arch=arch,loss=float(cost),gradients=gradients,
                                optical_parameters=sum(p.numel() for p in model.parameters()),
                                electronic_parameters=sum(p.numel() for p in frontend.parameters())))
    save(args.out/'smoke.json',dict(passed=True,records=records))


def run_mode(args,train,val,vocab,mode):
    root=args.out/mode;root.mkdir()
    setseed(args.seed);frontend=TextEncoder(len(vocab),mode).cuda()
    if mode=='learned':
        warm=root/'warmup';warm.mkdir();models=build_models(args.seed+1000,'cuda')
        optimizer=torch.optim.Adam([{'params':frontend.parameters(),'lr':args.frontend_lr},
                                    {'params':[p for m in models.values() for p in m.parameters()],'lr':args.lr}])
        best=float('inf');history=[]
        for epoch in range(1,args.warmup_epochs+1):
            start=time.time();cost=train_epoch(models,frontend,train,optimizer,args.batch,epoch,args.seed)
            scores={arch:evaluate(model,frontend,val,args.batch)[0] for arch,model in models.items()}
            objective=np.mean([x['nll'] for x in scores.values()])
            row=dict(epoch=epoch,train_online_nll=cost,val=scores,selection_mean_nll=float(objective),seconds=time.time()-start)
            history.append(row);save(warm/'history.json',history)
            checkpoint=dict(frontend=frontend.state_dict(),models={a:m.state_dict() for a,m in models.items()},
                            optimizer=optimizer.state_dict(),epoch=epoch,scores=scores)
            torch.save(checkpoint,warm/'last_checkpoint.pt')
            if objective<best:
                best=objective;torch.save(checkpoint,warm/'best_checkpoint.pt')
            print(json.dumps(dict(mode=mode,stage='warmup',**row)),flush=True)
        checkpoint=torch.load(warm/'best_checkpoint.pt',weights_only=False)
        frontend.load_state_dict(checkpoint['frontend'])
        save(root/'frontend_selection.json',dict(epoch=checkpoint['epoch'],mean_validation_nll=best))
        del models,optimizer,checkpoint;torch.cuda.empty_cache()
    frontend.requires_grad_(False).eval();frozen_hash=state_sha(frontend)
    torch.save(dict(mode=mode,state=frontend.state_dict(),sha256=frozen_hash),root/'frontend.pt')
    results={}
    for arch in ARCHS:
        path=root/arch;path.mkdir();setseed(args.seed)
        model=OpticalOEO(arch,args.seed).cuda()
        initial_phase_sha=state_sha(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.1)
        best=float('inf');history=[]
        for epoch in range(1,args.epochs+1):
            start=time.time();cost=train_epoch({arch:model},frontend,train,optimizer,args.batch,epoch,args.seed)
            train_score,_=evaluate(model,frontend,train,args.batch)
            val_score,_=evaluate(model,frontend,val,args.batch)
            assert state_sha(frontend)==frozen_hash
            scheduler.step()
            row=dict(epoch=epoch,train_online_nll=cost,train=train_score,val=val_score,seconds=time.time()-start)
            history.append(row);save(path/'history.json',history)
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                            epoch=epoch,train=train_score,val=val_score,frontend_sha256=frozen_hash,
                            initial_phase_sha256=initial_phase_sha)
            torch.save(checkpoint,path/'last_checkpoint.pt')
            if val_score['nll']<best:
                best=val_score['nll'];torch.save(checkpoint,path/'best_checkpoint.pt')
            print(json.dumps(dict(mode=mode,arch=arch,stage='frozen_optical',**row)),flush=True)
        checkpoint=torch.load(path/'best_checkpoint.pt',weights_only=False);model.load_state_dict(checkpoint['model'])
        train_score,train_pred=evaluate(model,frontend,train,args.batch)
        val_score,val_pred=evaluate(model,frontend,val,args.batch)
        counter_score,_=evaluate(model,frontend,val,args.batch,ablation='swap_pair_text')
        assert abs(val_score['nll']-checkpoint['val']['nll'])<1e-6
        np.savez_compressed(path/'predictions.npz',train=train_pred,val=val_pred,
                            train_labels=train['labels'].cpu().numpy(),val_labels=val['labels'].cpu().numpy())
        results[arch]=dict(epoch=checkpoint['epoch'],train=train_score,val=val_score,counterfactual=counter_score,
                           frontend_sha256=frozen_hash,best_checkpoint_sha256=digest((path/'best_checkpoint.pt').read_bytes()),
                           optical_parameters=sum(p.numel() for p in model.parameters()))
        save(path/'result.json',results[arch]);del model,optimizer,scheduler,checkpoint;torch.cuda.empty_cache()
    assert results['moe']['frontend_sha256']==results['d2nn']['frontend_sha256']
    save(root/'summary.json',results)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','train'],default='train')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['fixed','fixed_dense','learned','both'],default='both')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--warmup-epochs',type=int,default=8);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--frontend-lr',type=float,default=.001)
    p.add_argument('--vision-checkpoint',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    setseed(args.seed);metadata(args,args.out);save(args.out/'status.json',dict(status='running',pid=os.getpid()))
    vocab=json.loads((args.data/'vocab.json').read_text())
    train=load_data(args.data,'train',vocab,'cuda');val=load_data(args.data,'val',vocab,'cuda')
    if args.vision_checkpoint:
        from .vision import frozen_features
        visual={}
        for split,data in [('train',train),('val',val)]:
            data['images']=frozen_features(args.vision_checkpoint,data['images'])
            visual[split+'_feature_sha256']=digest(data['images'].cpu().numpy().tobytes())
        visual['checkpoint_sha256']=digest(args.vision_checkpoint.read_bytes())
        save(args.out/'shared_visual_frontend.json',visual)
    assert set(x['image_id'] for x in train['rows']).isdisjoint(x['image_id'] for x in val['rows'])
    if args.phase=='smoke':smoke(args,train,vocab)
    else:
        result={mode:run_mode(args,train,val,vocab,mode) for mode in (['fixed','learned'] if args.mode=='both' else [args.mode])}
        save(args.out/'summary.json',result)
    save(args.out/'status.json',dict(status='complete',pid=os.getpid(),test_accessed=False))


if __name__=='__main__':
    main()
