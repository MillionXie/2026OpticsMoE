"""Full PATCH16 four-slot optical/electronic MoE and ordinary D2NN training."""
import argparse,copy,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .model_four import FourRouterClassification,CONTRACT,encode_rgb
from .patch16_data import Patch16Dataset
from .prepare_patch16 import digest,write
from .train_regularized import augment_d4,update_ema,router_balance_loss

def loader(root,split,manifest,batch,workers,shuffle=False,seed=17):
    return DataLoader(Patch16Dataset(root,split,manifest),batch_size=batch,shuffle=shuffle,
        num_workers=workers,pin_memory=True,persistent_workers=workers>0,
        generator=torch.Generator().manual_seed(seed),drop_last=False)

def route_summary(q):
    mean=q.mean(0).astype(float);mean/=mean.sum()
    pairs,counts=np.unique(np.sort(np.argsort(q,axis=1)[:,-2:],axis=1),axis=0,return_counts=True)
    return dict(mean_power=q.mean(0).tolist(),std_power=q.std(0).tolist(),
        selected_fraction=(q>0).mean(0).tolist(),argmax_fraction=(np.bincount(q.argmax(1),minlength=4)/len(q)).tolist(),
        effective_experts=float(np.exp(-(mean[mean>0]*np.log(mean[mean>0])).sum())),
        mean_selected_count=float((q>0).sum(1).mean()),
        top2_pair_fraction={','.join(str(int(i)+1) for i in pair):float(n/len(q)) for pair,n in zip(pairs,counts)})

@torch.no_grad()
def evaluate(model,batches,classes=16):
    model.eval();pred=[];truth=[];ids=[];routes=[];total_loss=0.
    for images,labels,index in batches:
        labels=labels.cuda(non_blocking=True);amp=encode_rgb(images.cuda(non_blocking=True))
        out=model(amp);total_loss+=float(F.cross_entropy(out['logits'],labels,reduction='sum'))
        pred.extend(out['logits'].argmax(1).cpu().tolist());truth.extend(labels.cpu().tolist());ids.extend(index.tolist())
        if out['route_power'] is not None:routes.append(out['route_power'].cpu().numpy())
    pred=np.asarray(pred);truth=np.asarray(truth);cm=np.bincount(truth*classes+pred,minlength=classes**2).reshape(classes,classes)
    if min(cm.sum(1))<1:raise ValueError('evaluation class missing')
    result=dict(accuracy=float((pred==truth).mean()),balanced_accuracy=float((cm.diagonal()/cm.sum(1)).mean()),
        per_class_recall=(cm.diagonal()/cm.sum(1)).tolist(),confusion_matrix=cm.tolist(),n=len(pred),cross_entropy=total_loss/len(pred))
    q=np.concatenate(routes) if routes else None
    if q is not None:result['router']=route_summary(q)
    return result,dict(indices=np.asarray(ids),labels=truth,predictions=pred,route_power=q if q is not None else np.empty((len(pred),0)))

def score(metrics):return .5*(metrics['accuracy']+metrics['balanced_accuracy'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--architecture',choices=['optical','electronic','d2nn']);p.add_argument('--recipe',choices=['standard','enhanced'],default='standard')
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--evaluate-test',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.evaluate_test:
        if (a.out/'result.json').exists():raise FileExistsError('test already evaluated')
        ck=torch.load(a.out/'best_checkpoint.pt',map_location='cuda',weights_only=False);cfg=ck['config']
        if str(a.data_root)!=cfg['data_root'] or digest(a.data_root/'manifest.json')!=cfg['manifest_sha256']:raise ValueError('data identity changed')
        manifest=json.loads((a.data_root/'manifest.json').read_text())
        if digest(a.data_root/'data.zip')!=manifest['archive_sha256']:raise ValueError('archive changed')
        model=FourRouterClassification(cfg['architecture'],num_classes=16).cuda();model.load_state_dict(ck['model'])
        metrics,arrays=evaluate(model,loader(a.data_root,'test',manifest,cfg['batch'],a.workers))
        metrics.update(selected_epoch=ck['epoch'],weights_kind=ck['weights_kind'],test_evaluations=1,
            checkpoint_sha256=digest(a.out/'best_checkpoint.pt'),selection='validation only',dataset='PATCH16')
        np.savez_compressed(a.out/'test_predictions.npz',**arrays)
        write(a.out/'result.json',metrics);write(a.out/'status.json',dict(state='completed_test_once',result=metrics));print(json.dumps(metrics));return
    if a.architecture is None or (a.architecture=='d2nn' and a.recipe!='standard'):raise ValueError('D2NN is ordinary standard training')
    if a.epochs<1 or a.batch<1 or a.workers<0:raise ValueError('invalid budget')
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'status.json',dict(state='loading'))
    try:
        manifest=json.loads((a.data_root/'manifest.json').read_text())
        if manifest['dataset']!='PATCH16' or manifest['license']!='CC BY 4.0' or manifest['total']!=129364:raise ValueError('wrong dataset')
        if digest(a.data_root/'data.zip')!=manifest['archive_sha256']:raise ValueError('source checksum mismatch')
        cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        enhanced=a.recipe=='enhanced';phase_lr=.003 if enhanced else .005;head_lr=.0005 if enhanced else .001
        cfg.update(dataset='PATCH16',num_classes=16,optical_contract=CONTRACT,top_k=2,input='fixed original RGB395 -> bicubic112 -> [R,G;B,0] unit-power224',
            manifest_sha256=digest(a.data_root/'manifest.json'),archive_sha256=manifest['archive_sha256'],
            class_names=manifest['class_names'],split_counts=manifest['split_counts'],phase_lr=phase_lr,electronic_lr=head_lr,
            optimizer='AdamW' if enhanced else 'Adam',augmentation_probability=.5 if enhanced else 0.,label_smoothing=.03 if enhanced else 0.,
            class_weight_power=.5 if enhanced else 0.,ema_decay=.995 if enhanced else None,minimum_lr_ratio=.1 if enhanced else 1.,
            router_balance_weight=(.1 if a.architecture=='electronic' else .01) if enhanced else 0.,
            router_balance_mode='top2_load' if a.architecture=='electronic' else 'dense_kl',selection='validation accuracy/macro mean for MoE; validation macro for ordinary D2NN',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=sys.argv,
            python=platform.python_version(),torch=torch.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),skip_test=True)
        model=FourRouterClassification(a.architecture,num_classes=16).cuda();cfg['total_parameters']=sum(p.numel() for p in model.parameters());write(a.out/'config.json',cfg)
        ema=copy.deepcopy(model).eval() if enhanced else None
        if ema is not None:ema.requires_grad_(False)
        groups=[]
        for name,parameter in model.named_parameters():
            electronic='shared_head' in name or 'electronic_router' in name
            groups.append(dict(params=[parameter],lr=head_lr if electronic else phase_lr,
                weight_decay=(.01 if 'shared_head' in name else .0001) if enhanced and electronic else 0.))
        opt=(torch.optim.AdamW if enhanced else torch.optim.Adam)(groups)
        bases=[g['lr'] for g in opt.param_groups]
        train=loader(a.data_root,'train',manifest,a.batch,a.workers,True,a.seed)
        val=loader(a.data_root,'val',manifest,a.batch,a.workers)
        counts=torch.tensor(manifest['class_counts']['train'],device='cuda',dtype=torch.float)
        weights=(counts.mean()/counts).pow(cfg['class_weight_power']);weights/=weights.mean()
        cfg['loss_class_weights']=weights.cpu().tolist();write(a.out/'config.json',cfg)
        best=-1.;history=[]
        for epoch in range(1,a.epochs+1):
            begin=time.monotonic();model.train();loss_sum=0.;seen=0
            factor=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/max(1,a.epochs-1))) if enhanced else 1.
            for g,lr in zip(opt.param_groups,bases):g['lr']=lr*factor
            for images,labels,index in train:
                labels=labels.cuda(non_blocking=True);amp=encode_rgb(images.cuda(non_blocking=True))
                if enhanced:
                    codes=torch.randint(0,8,(len(labels),),device='cuda')
                    codes=torch.where(torch.rand(len(labels),device='cuda')<.5,codes,torch.zeros_like(codes));amp=augment_d4(amp,codes)
                opt.zero_grad(set_to_none=True);out=model(amp,return_dense_route=enhanced)
                loss=F.cross_entropy(out['logits'],labels,weight=weights if enhanced else None,label_smoothing=cfg['label_smoothing'])
                if enhanced:loss=loss+cfg['router_balance_weight']*router_balance_loss(out['dense_route_power'],cfg['router_balance_mode'])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                if ema is not None:update_ema(ema,model,cfg['ema_decay'])
                loss_sum+=float(loss.detach())*len(labels);seen+=len(labels)
            if seen!=manifest['split_counts']['train']:raise ValueError('training did not traverse full split')
            online,arrays=evaluate(model,val);options=[('online',model,online)]
            if ema is not None:
                average,_=evaluate(ema,val);options.append(('ema',ema,average))
            metric_score=(lambda v:v['balanced_accuracy']) if a.architecture=='d2nn' else score
            kind,chosen,metrics=max(options,key=lambda o:(metric_score(o[2]),-o[2]['cross_entropy']))
            selected=metric_score(metrics)>best
            if selected:
                best=metric_score(metrics);torch.save(dict(model=chosen.state_dict(),epoch=epoch,weights_kind=kind,config=cfg),a.out/'best_checkpoint.pt')
                write(a.out/'selected_validation.json',dict(epoch=epoch,weights_kind=kind,metrics=metrics,selection_score=best))
            row=dict(epoch=epoch,train_loss=loss_sum/seen,training_samples_seen=seen,validation=metrics,weights_kind=kind,selected=selected,seconds=time.monotonic()-begin)
            history.append(row);write(a.out/'metrics.json',history)
            torch.save(dict(model=model.state_dict(),ema_model=ema.state_dict() if ema is not None else None,optimizer=opt.state_dict(),
                epoch=epoch,config=cfg,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),loader_rng=train.generator.get_state()),a.out/'last_checkpoint.pt')
            write(a.out/'status.json',dict(state='training_validation_only',epoch=epoch,best_validation_score=best,seconds=row['seconds']))
            print(json.dumps(row),flush=True)
        write(a.out/'status.json',dict(state='completed_validation_only',epochs=a.epochs,best_validation=json.loads((a.out/'selected_validation.json').read_text()),test_evaluations=0))
    except Exception as exc:write(a.out/'status.json',dict(state='failed',error=repr(exc)));raise

if __name__=='__main__':main()
