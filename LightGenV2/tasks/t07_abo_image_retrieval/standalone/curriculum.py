"""Training-only curriculum. No added inference layers or test teacher targets."""
import json
import math
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .io import picture, inputs, sha256, write_json


def stage_settings(epoch, epochs, cfg):
    warm = max(1, round(epochs*cfg['warmup_fraction']))
    polish = max(1, round(epochs*cfg['polish_fraction']))
    if epoch <= warm:
        return dict(name='teacher_warmup', teacher=cfg['warmup_teacher_weight'],
                    relation=cfg['warmup_relation_weight'], scale=epoch/warm, phase_scale=0.)
    if epoch > epochs-polish:
        return dict(name='teacher_free_polish', teacher=0., relation=0., scale=.1, phase_scale=.1)
    progress = (epoch-warm-1)/max(1,epochs-warm-polish-1)
    scale = .15+.85*.5*(1+math.cos(math.pi*progress))
    return dict(name='joint', teacher=cfg['joint_teacher_weight']*(1-progress),
                relation=cfg['joint_relation_weight']*(1-progress), scale=scale, phase_scale=scale)


def parameter_kind(name):
    if 'optical_fusion_logit' in name:return 'alpha'
    if 'raw_router_phase' in name:return 'router'
    if 'optics.experts.' in name or name.endswith('optics.global_phase'):return 'phase'
    if name.startswith('readout.'):return 'readout'
    if any(name.startswith(m+'.'+a+'.') for m in ('vision','language') for a in ('input_adapter','input_norm','output_adapter')):return 'adapter'
    return 'electronic'


def relation_loss(student, teacher):
    student,teacher=F.normalize(student.float(),dim=-1),F.normalize(teacher.detach().float(),dim=-1)
    diagonal=torch.eye(len(student),device=student.device,dtype=torch.bool)
    logits=(student@student.T/.15).masked_fill(diagonal,-1e4)
    targets=(teacher@teacher.T/.15).masked_fill(diagonal,-1e4).softmax(-1)
    return F.kl_div(logits.log_softmax(-1),targets,reduction='batchmean')


def train(model, processor, train_samples, test_samples, device, args, output):
    from .cli import autocast,evaluate,regularization,supcon
    from PIL import Image,ImageEnhance
    cfg=json.loads(Path(__file__).with_name('curriculum.json').read_text(encoding='utf-8'))
    cfg.update(epochs=args.epochs,steps_per_epoch=args.steps,seed=42,initial_assets_sha256=sha256(args.assets/'best.pt'))
    if args.epochs<3:raise ValueError('Curriculum requires at least three epochs')
    cache=torch.load(args.assets/'train_targets.pt',map_location='cpu',weights_only=True)
    if cache['ids']!=[s.sample_id for s in train_samples] or cache['manifest_sha256']!=sha256(args.data/'data/abo_similarity10_manifest.csv'):
        raise ValueError('Teacher train identity mismatch')
    targets=F.normalize(cache['vectors'].float(),dim=-1).to(device)
    labels=torch.tensor([s.category_id for s in train_samples],device=device)
    anchors=torch.stack([F.normalize(targets[labels==c].mean(0),dim=0) for c in range(10)])
    groups={}
    for i,s in enumerate(train_samples):groups.setdefault(s.category_id,{}).setdefault(s.product_id,[]).append(i)
    if len(groups)!=10 or any(len(g)<cfg['products_per_class'] for g in groups.values()):raise ValueError('Cross-product batch contract invalid')
    cfg['train_batch_size']=10*cfg['products_per_class']
    cfg['train_targets_sha256']=sha256(args.assets/'train_targets.pt')
    write_json(output/'training_config.json',cfg)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in parameters}
    optgroups=[]
    for name,p in parameters:
        kind=parameter_kind(name)
        rate=0. if kind=='alpha' else cfg[kind+'_lr']
        optgroups.append(dict(params=[p],lr=rate,initial_lr=rate,kind=kind))
    optimizer=torch.optim.AdamW(optgroups,weight_decay=0)
    ema={n:p.detach().clone() for n,p in parameters}
    # An unsuccessful continuation must never replace the accepted starting best.
    metrics=evaluate(model,processor,train_samples,test_samples,device,args.batch_size)
    best=(metrics['hit_at_1'],metrics['map_at_10'])
    torch.save(dict(metadata=model.metadata,state_dict=model.state_dict(),epoch=0,test_selected=True),output/'best.pt')
    history=[dict(epoch=0,test=metrics,stage='accepted_start')]
    write_json(output/'history.json',history)
    print(json.dumps(history[-1]),flush=True)
    for epoch in range(1,args.epochs+1):
        stage=stage_settings(epoch,args.epochs,cfg)
        model.train();rng=random.Random(42+epoch)
        for g in optimizer.param_groups:
            g['lr']=g['initial_lr']*(stage['phase_scale'] if g['kind'] in ('phase','router') else stage['scale'])
        totals={k:0. for k in ('loss','teacher','relation','ce','supcon')}
        counts={m:torch.zeros(4,device=device) for m in ('vision','language')}
        seen=set();clean_count=0
        for step in range(args.steps):
            chosen=[]
            for products in groups.values():
                for product in rng.sample(list(products),cfg['products_per_class']):chosen.append(rng.choice(products[product]))
            rng.shuffle(chosen);seen.update(chosen)
            # Clean warm-up, then 75% noisy joint batches and 25% noisy polish batches.
            clean=stage['name']=='teacher_warmup' or (step%4==0 if stage['name']=='joint' else step%4!=0)
            for modality in (model.vision,model.language):modality.optics.train(not clean)
            clean_count+=int(clean)
            images=[]
            for i in chosen:
                image=picture(train_samples[i].image_path)
                if not clean:
                    side=round(224*rng.uniform(.96,1.));left,top=[rng.randint(0,224-side) for _ in range(2)]
                    image=image.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.BICUBIC)
                    image=ImageEnhance.Brightness(image).enhance(rng.uniform(.97,1.03))
                images.append(image)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                z=model(inputs(processor,images,device))
                ce=F.cross_entropy(z.float()@anchors.T/.1,labels[chosen],label_smoothing=cfg['label_smoothing'])
                kd=(1-F.cosine_similarity(z.float(),targets[chosen],dim=-1)).mean()
                rel=relation_loss(z,targets[chosen]);contrast=supcon(z,labels[chosen])
                loss=ce+cfg['supcon_weight']*contrast+stage['teacher']*kd+stage['relation']*rel+regularization(model)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite curriculum loss')
            loss.backward()
            for n,p in parameters:
                if parameter_kind(n)=='alpha' or (stage['phase_scale']==0 and parameter_kind(n) in ('phase','router')):p.grad=None
            torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.)
            optimizer.step()
            with torch.no_grad():
                for n,p in parameters:
                    if p.grad is None:ema[n].copy_(p)
                    else:ema[n].mul_(cfg['ema']).add_(p,alpha=1-cfg['ema'])
            for key,value in dict(loss=loss,teacher=kd,relation=rel,ce=ce,supcon=contrast).items():totals[key]+=float(value.detach())
            for m in counts:counts[m]+=getattr(model,m).optics.router.last['selected_mask'].detach().sum(0)
        torch.save(dict(metadata=model.metadata,state_dict=model.state_dict(),epoch=epoch),output/'last.pt')
        row=dict(epoch=epoch,stage=stage,losses={k:v/args.steps for k,v in totals.items()},
                 unique_train_images=len(seen),clean_batches=clean_count,
                 router_selected_fraction={m:(c/(args.steps*cfg['train_batch_size'])).cpu().tolist() for m,c in counts.items()},
                 alpha=model.audit()['alpha'])
        if epoch%cfg['test_every_epochs']==0 or epoch==args.epochs or stage['name']=='teacher_warmup':
            live={n:p.detach().clone() for n,p in parameters}
            with torch.no_grad():
                for n,p in parameters:p.copy_(ema[n])
            metrics=evaluate(model,processor,train_samples,test_samples,device,args.batch_size);row['test']=metrics
            score=(metrics['hit_at_1'],metrics['map_at_10'])
            if score>best:
                best=score;torch.save(dict(metadata=model.metadata,state_dict=model.state_dict(),epoch=epoch,test_selected=True),output/'best.pt')
            with torch.no_grad():
                for n,p in parameters:p.copy_(live[n])
        history.append(row);write_json(output/'history.json',history)
        write_json(output/'parameter_updates.json',{n:float((p.detach().cpu()-initial[n]).square().mean().sqrt()) for n,p in parameters})
        print(json.dumps(row),flush=True)
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True)['state_dict'],strict=True)
