"""Independent commands: evaluate, finetune, verify. One GPU per process."""
import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import _load_contract, _gallery_centroids, _category_prototypes, _evaluate
from .io import inputs, picture, verify_assets, write_json, write_csv, sha256
from .model import OpticalRetrieval


def autocast(device):
    return torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')


@torch.no_grad()
def encode(model, processor, samples, device, batch_size):
    model.eval()
    output = []
    for start in range(0,len(samples),batch_size):
        batch = inputs(processor,[picture(s.image_path) for s in samples[start:start+batch_size]],device)
        with autocast(device):
            output.append(model(batch).detach().cpu())
        if (start//batch_size+1)%60==0:
            print(f'encoded {min(start+batch_size,len(samples))}/{len(samples)}',flush=True)
    return torch.cat(output)


@torch.no_grad()
def evaluate(model, processor, train, test, device, batch_size, output=None):
    vtrain = encode(model,processor,train,device,batch_size)
    vtest = encode(model,processor,test,device,batch_size)
    gallery,metadata = _gallery_centroids(train,F.normalize(vtrain.float(),dim=-1))
    metrics,rows,categories = _evaluate(vtest,test,gallery,metadata,_category_prototypes(gallery,metadata))
    if output:
        write_csv(output/'retrieval_predictions.csv',rows)
        write_csv(output/'per_category_metrics.csv',categories)
        torch.save({'train_ids':[s.sample_id for s in train],'test_ids':[s.sample_id for s in test],
                    'train':vtrain,'test':vtest},output/'retrieval_features.pt')
    return metrics


def supcon(z, labels):
    z = F.normalize(z.float(),dim=-1)
    scores = z@z.T/.1
    diagonal = torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive = labels[:,None].eq(labels[None,:]) & ~diagonal
    logp = scores-scores.masked_fill(diagonal,-torch.inf).logsumexp(1,keepdim=True)
    return -(logp.masked_fill(~positive,0).sum(1)/positive.sum(1)).mean()


def regularization(model):
    balances, importances, hard, dc, operating = [],[],[],[],[]
    for modality in (model.vision,model.language):
        optics = modality.optics
        r = optics.router.last
        balances.append(r['balance']);importances.append(r['importance']);hard.append(r['hard_balance'])
        for raw in [*optics.experts,optics.global_phase]:
            phase = 2*torch.pi*raw.sigmoid()
            dc.append(phase.cos().mean().square()+phase.sin().mean().square())
        operating.append(F.smooth_l1_loss(optics.operating_losses[-1],torch.full_like(optics.operating_losses[-1],np.log(.25))))
    return .08*torch.stack(balances).mean()+.02*torch.stack(importances).mean()+.5*torch.stack(hard).mean()+.005*torch.stack(dc).mean()+.02*torch.stack(operating).mean()


def finetune(model, processor, train, test, device, args, output):
    """Fresh optimizer continuation, not an exact resume of old RNG/Adam state."""
    from PIL import Image, ImageEnhance
    targets_file = args.assets/'train_targets.pt'
    if not targets_file.is_file():
        raise FileNotFoundError('Export --teacher-cache to include train-only targets; no full teacher is loaded')
    cache = torch.load(targets_file,map_location='cpu',weights_only=True)
    if cache['ids'] != [s.sample_id for s in train] or cache['manifest_sha256']!=sha256(args.data/'data/abo_similarity10_manifest.csv'):
        raise ValueError('Teacher train identity mismatch')
    targets = F.normalize(cache['vectors'].float(),dim=-1).to(device)
    labels = torch.tensor([s.category_id for s in train],device=device)
    anchors = torch.stack([F.normalize(targets[labels==c].mean(0),dim=0) for c in range(10)])
    groups = {}
    for i,s in enumerate(train):groups.setdefault(s.category_id,{}).setdefault(s.product_id,[]).append(i)
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    optgroups=[]
    for n,p in parameters:
        rate = .00002
        if 'optics.experts.' in n or n.endswith('optics.global_phase'):rate=.004
        elif 'raw_router_phase' in n:rate=.0005
        elif n.startswith('readout.'):rate=.00005
        elif any(n.startswith(m+'.'+a+'.') for m in ('vision','language') for a in ('input_adapter','input_norm','output_adapter')):rate=.00001
        optgroups.append({'params':[p],'lr':rate,'initial_lr':rate})
    optimizer = torch.optim.AdamW(optgroups,weight_decay=0)
    ema = {n:p.detach().clone() for n,p in parameters}
    initial = {n:p.detach().cpu().clone() for n,p in parameters if 'optics.experts.' in n or n.endswith('optics.global_phase')}
    history=[];best=(-1.,-1.)
    for epoch in range(1,args.epochs+1):
        model.train()
        rng = random.Random(42+epoch)
        scale = epoch/5 if epoch<=5 else .1+.9*.5*(1+np.cos(np.pi*(epoch-5)/max(1,args.epochs-5)))
        for g in optimizer.param_groups:g['lr']=g['initial_lr']*scale
        total=0.
        for step in range(args.steps):
            chosen=[]
            for products in groups.values():
                for product in rng.sample(list(products),2):chosen.append(rng.choice(products[product]))
            rng.shuffle(chosen)
            images=[]
            for i in chosen:
                image=picture(train[i].image_path);side=round(224*random.uniform(.94,1.))
                left,top=[random.randint(0,224-side) for _ in range(2)]
                image=image.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.BICUBIC)
                image=ImageEnhance.Brightness(image).enhance(random.uniform(.95,1.05))
                images.append(ImageEnhance.Contrast(image).enhance(random.uniform(.95,1.05)))
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                z=model(inputs(processor,images,device))
                ce=F.cross_entropy(z.float()@anchors.T/.1,labels[chosen])
                kd=(1-F.cosine_similarity(z.float(),targets[chosen],dim=-1)).mean()
                loss=.5*supcon(z,labels[chosen])+.1*kd+ce+regularization(model)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.)
            optimizer.step()
            with torch.no_grad():
                for n,p in parameters:ema[n].mul_(.99).add_(p,alpha=.01)
            total+=float(loss.detach())
        torch.save({'metadata':model.metadata,'state_dict':model.state_dict(),'epoch':epoch},output/'last.pt')
        row={'epoch':epoch,'loss':total/args.steps,'router_counts':{
            m:getattr(model,m).optics.router.last['selected_mask'].detach().sum(0).cpu().tolist()
            for m in ('vision','language')},'router_count_scope':'last live training batch only'}
        if epoch%5==0 or epoch==args.epochs:
            live={n:p.detach().clone() for n,p in parameters}
            with torch.no_grad():
                for n,p in parameters:p.copy_(ema[n])
            metrics=evaluate(model,processor,train,test,device,args.batch_size)
            row['test']=metrics
            score=(metrics['hit_at_1'],metrics['map_at_10'])
            if score>best:
                best=score
                torch.save({'metadata':model.metadata,'state_dict':model.state_dict(),'epoch':epoch,'test_selected':True},output/'best.pt')
            with torch.no_grad():
                for n,p in parameters:p.copy_(live[n])
        history.append(row);write_json(output/'history.json',history)
        print(json.dumps(row),flush=True)
    write_json(output/'phase_update.json',{n:float((dict(model.named_parameters())[n].detach().cpu()-p).square().mean().sqrt()) for n,p in initial.items()})
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True)['state_dict'],strict=True)


def preview(model, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,6,figsize=(14,5),constrained_layout=True)
    for row,mode in enumerate(('vision','language')):
        optics=getattr(model,mode).optics
        raws=[optics.router.raw_router_phase,*optics.experts,optics.global_phase]
        for col,(raw,label) in enumerate(zip(raws,['router','expert0','expert1','expert2','expert3','global'])):
            axes[row,col].imshow((2*torch.pi*raw.detach().cpu().sigmoid()).numpy(),vmin=0,vmax=2*np.pi,cmap='twilight')
            axes[row,col].set_title(mode+' '+label);axes[row,col].set_axis_off()
    fig.savefig(output/'phase_masks.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','evaluate','finetune'])
    parser.add_argument('--assets',type=Path,default=Path('assets'))
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--steps',type=int,default=48)
    parser.add_argument('--reference',type=Path,help='Old retrieval_features.pt; read-only comparison')
    args=parser.parse_args()
    manifest=verify_assets(args.assets)
    if args.command=='verify':
        release_root=Path(__file__).resolve().parent.parent
        if (release_root/'MANIFEST.json').is_file():
            release=json.loads((release_root/'MANIFEST.json').read_text(encoding='utf-8'))
            for name,record in release['files'].items():
                file=(release_root/name).resolve()
                if not file.is_relative_to(release_root) or not file.is_file() or sha256(file)!=record['sha256']:
                    raise RuntimeError(f'Package file changed: {name}')
            print(f"Full package verified: {len(release['files'])} files, source {release['source_commit']}")
        print(json.dumps(manifest,indent=2));return
    if args.data is None or args.output is None:parser.error('--data and --output required')
    if min(args.batch_size,args.epochs,args.steps)<1:parser.error('Positive batch size/epochs/steps required')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Choose an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    # This program never starts child GPU processes, DDP or DataParallel.
    model=None
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        payload=torch.load(args.assets/'best.pt',map_location='cpu',weights_only=True)
        model=OpticalRetrieval(payload['metadata']);model.load_state_dict(payload['state_dict'],strict=True)
        del payload
        model.to(device)
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(str(args.assets/'processor'),local_files_only=True)
        samples,_=_load_contract(args.data)
        train=[s for s in samples if s.split=='train'];test=[s for s in samples if s.split=='test']
        write_json(args.output/'execution.json',dict(command=sys.argv,pid=os.getpid(),python=sys.version,torch=torch.__version__,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                   assets_manifest_sha256=sha256(args.assets/'manifest.json'),audit=model.audit()))
        if args.command=='finetune':finetune(model,processor,train,test,device,args,args.output)
        metrics=evaluate(model,processor,train,test,device,args.batch_size,args.output)
        model.set_remove_optical(True)
        removed=evaluate(model,processor,train,test,device,args.batch_size)
        model.set_remove_optical(False)
        report=dict(status='complete',metrics=metrics,remove_optical_same_weights=removed,
                    optical_removal_hit1_drop_percentage_points=100*(metrics['hit_at_1']-removed['hit_at_1']),
                    audit=model.audit(),test_selected=True)
        if device.type=='cuda':
            report['gpu_memory_mib']={'peak_allocated':torch.cuda.max_memory_allocated(device)/2**20,
                                      'peak_reserved':torch.cuda.max_memory_reserved(device)/2**20}
        if args.reference:
            old=torch.load(args.reference,map_location='cpu',weights_only=True)
            new=torch.load(args.output/'retrieval_features.pt',map_location='cpu',weights_only=True)
            if old['train_ids']!=new['train_ids'] or old['test_ids']!=new['test_ids']:raise ValueError('Reference sample identity differs')
            report['feature_comparison']={k:{'rms_difference':float((old[k].float()-new[k].float()).square().mean().sqrt()),
                 'mean_cosine':float(F.cosine_similarity(old[k].float(),new[k].float()).mean()),
                 'bitwise_equal':torch.equal(old[k],new[k])} for k in ('train','test')}
        preview(model,args.output)
        write_json(args.output/'final_report.json',report)
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        write_json(args.output/'failure.json',{'error':str(exc),'type':type(exc).__name__})
        raise
    finally:
        if model is not None:del model
        if device.type=='cuda':torch.cuda.empty_cache()
        # Process exit releases the CUDA context, even on exception/Ctrl+C.


if __name__=='__main__':main()
