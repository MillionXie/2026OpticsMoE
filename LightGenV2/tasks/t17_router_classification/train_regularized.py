"""Validation-only MoE fine-tuning; identical inference graph and fixed optical contract."""
import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model_four import FourRouterClassification,CONTRACT
from .train import encode,evaluate,write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain,sha256

def augment_d4(amplitude,codes):
    """Same D4 transform for R/G/B; keep the fourth reserved tile exactly zero."""
    rgb=torch.stack((amplitude[:,:112,:112],amplitude[:,:112,112:],
                     amplitude[:,112:,:112]),1)
    transformed=torch.empty_like(rgb)
    for code in range(8):
        selected=codes==code
        if not selected.any():continue
        value=torch.rot90(rgb[selected],code%4,(-2,-1))
        if code>=4:value=value.flip(-1)
        transformed[selected]=value
    out=torch.zeros_like(amplitude)
    out[:,:112,:112]=transformed[:,0];out[:,:112,112:]=transformed[:,1]
    out[:,112:,:112]=transformed[:,2]
    return out

@torch.no_grad()
def update_ema(average,online,decay):
    for a,p in zip(average.parameters(),online.parameters()):
        a.mul_(decay).add_(p,alpha=1-decay)
    for a,b in zip(average.buffers(),online.buffers()):a.copy_(b)

def selection_score(metrics):
    return .5*(metrics['accuracy']+metrics['balanced_accuracy'])

def class_weights(labels,power):
    counts=torch.bincount(labels,minlength=9).float()
    if (counts==0).any() or not 0<=power<=1:raise ValueError('class weighting requires all training classes')
    weights=(counts.mean()/counts).pow(power)
    return weights/weights.mean()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--init-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--phase-lr',type=float,default=.001)
    p.add_argument('--electronic-lr',type=float,default=.0002)
    p.add_argument('--head-decay',type=float,default=.001)
    p.add_argument('--label-smoothing',type=float,default=.05)
    p.add_argument('--ema-decay',type=float,default=.999)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--class-weight-power',type=float,default=0.)
    args=p.parse_args()
    if args.epochs<1 or not 0<=args.label_smoothing<1 or not 0<args.ema_decay<1:
        raise ValueError('invalid regularization budget')
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'status.json',dict(state='loading'))
    try:
        previous=json.loads((args.init_run/'config.json').read_text())
        if previous['profile']!='four_top2' or previous['architecture'] not in ('optical','electronic'):
            raise ValueError('only the existing four-top2 Linear MoE graph is authorized')
        if previous['optical_contract']!=CONTRACT:raise ValueError('optical contract mismatch')
        cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        cfg.update(profile='four_top2',readout='linear',architecture=previous['architecture'],
            data=previous['data'],manifest=previous['manifest'],batch=previous['batch'],
            data_sha256=sha256(Path(previous['data'])),
            manifest_sha256=sha256(Path(previous['manifest'])),optical_contract=CONTRACT,top_k=2,
            skip_test=True,router_features='mean',router_lr=args.electronic_lr,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            command=sys.argv,python=platform.python_version(),torch=torch.__version__,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            recipe='d4_cosine_ema_adamw_labelsmoothing',
            selection='0.5 * (full validation accuracy + macro recall)',
            augmentation='train-only common R/G/B D4; no color jitter; empty tile unchanged',
            optimizer_reinitialized=True,phase_weight_decay=0.,router_weight_decay=.0001,
            minimum_lr_ratio=.1)
        for key in ('data_sha256','manifest_sha256'):
            if cfg[key]!=previous[key]:raise ValueError(f'{key} mismatch')
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        device=torch.device('cuda')
        data,manifest=load_domain(Path(cfg['data']),Path(cfg['manifest']))
        if manifest['domain']!='A_original' or [len(data[s+'_labels']) for s in ('train','val','test')]!=[5026,718,1436]:
            raise ValueError('full original-domain split required')
        # No test image is encoded or inferred during candidate training.
        x={s:encode(data[s+'_images'],'four_top2').to(device) for s in ('train','val')}
        y={s:torch.as_tensor(data[s+'_labels'],device=device,dtype=torch.long) for s in ('train','val')}
        weights=class_weights(y['train'],args.class_weight_power)
        cfg['loss_class_weights']=weights.cpu().tolist()
        cfg['training_class_counts']=torch.bincount(y['train'],minlength=9).cpu().tolist()
        model=FourRouterClassification(cfg['architecture']).to(device)
        ck=torch.load(args.init_run/'best_checkpoint.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);initial_epoch=int(ck['epoch']);del ck
        cfg['initial_checkpoint']=dict(epoch=initial_epoch,
            sha256=sha256(args.init_run/'best_checkpoint.pt'),source_git_commit=previous['git_commit'])
        cfg['effective_final_epoch']=initial_epoch+args.epochs
        cfg['total_parameters']=sum(p.numel() for p in model.parameters())
        write(args.out/'config.json',cfg)
        ema=copy.deepcopy(model).eval()
        for parameter in ema.parameters():parameter.requires_grad_(False)
        groups=[]
        for name,parameter in model.named_parameters():
            lr=args.electronic_lr if 'shared_head' in name or 'electronic_router' in name else args.phase_lr
            decay=args.head_decay if 'shared_head' in name else (.0001 if 'electronic_router' in name else 0.)
            groups.append(dict(params=[parameter],lr=lr,weight_decay=decay))
        opt=torch.optim.AdamW(groups)
        base_lrs=[g['lr'] for g in opt.param_groups]
        initial_val,_,_=evaluate(model,x['val'],y['val'],cfg['batch'])
        initial_train,_,_=evaluate(model,x['train'],y['train'],cfg['batch'])
        write(args.out/'initial_diagnostics.json',dict(epoch=initial_epoch,training=initial_train,validation=initial_val))
        best=selection_score(initial_val);history=[]
        def save_best(chosen,epoch,kind):
            torch.save(dict(model=chosen.state_dict(),epoch=initial_epoch+epoch,
                fine_tune_epoch=epoch,weights_kind=kind,config=cfg,optimizer=None,
                checkpoint_role='selected inference weights; optimizer must be reinitialized'),
                args.out/'best_checkpoint.pt')
        save_best(model,0,'parent')
        rng=np.random.default_rng(args.seed)
        for epoch in range(1,args.epochs+1):
            begin=time.time();model.train();loss_sum=0.;indices=rng.permutation(len(y['train']))
            # Cosine from the requested LR to 10% of it; preserve per-group ratios.
            factor=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/max(1,args.epochs-1)))
            for group,base in zip(opt.param_groups,base_lrs):group['lr']=base*factor
            for start in range(0,len(indices),cfg['batch']):
                ix=torch.as_tensor(indices[start:start+cfg['batch']],device=device)
                codes=torch.randint(0,8,(len(ix),),device=device)
                opt.zero_grad(set_to_none=True)
                out=model(augment_d4(x['train'][ix],codes))
                loss=F.cross_entropy(out['logits'],y['train'][ix],
                    weight=weights if args.class_weight_power>0 else None,
                    label_smoothing=args.label_smoothing)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step();update_ema(ema,model,args.ema_decay)
                loss_sum+=float(loss.detach())*len(ix)
            raw_val,_,_=evaluate(model,x['val'],y['val'],cfg['batch'])
            ema_val,_,_=evaluate(ema,x['val'],y['val'],cfg['batch'])
            kind,chosen,metrics=max((('online',model,raw_val),('ema',ema,ema_val)),
                key=lambda t:(selection_score(t[2]),-t[2]['cross_entropy']))
            score=selection_score(metrics);improved=score>best
            if improved:best=score;save_best(chosen,epoch,kind)
            row=dict(epoch=initial_epoch+epoch,fine_tune_epoch=epoch,train_loss=loss_sum/len(indices),
                validation=metrics,online_validation=raw_val,ema_validation=ema_val,
                evaluated_kind=kind,selection_score=score,selected=improved,
                learning_rates=[g['lr'] for g in opt.param_groups])
            if epoch%5==0 or epoch==args.epochs:
                row['training'],_,_=evaluate(chosen,x['train'],y['train'],cfg['batch'])
            row['seconds']=time.time()-begin
            history.append(row);write(args.out/'metrics.json',history)
            torch.save(dict(model=model.state_dict(),ema_model=ema.state_dict(),optimizer=opt.state_dict(),
                epoch=initial_epoch+epoch,fine_tune_epoch=epoch,config=cfg,
                shuffle_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state()),args.out/'last_checkpoint.pt')
            write(args.out/'status.json',dict(state='training_validation_only',epoch=initial_epoch+epoch,
                fine_tune_epoch=epoch,best_validation_score=best))
            print(json.dumps(row),flush=True)
        ck=torch.load(args.out/'best_checkpoint.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model'])
        selected_train,_,_=evaluate(model,x['train'],y['train'],cfg['batch'])
        selected_val,_,_=evaluate(model,x['val'],y['val'],cfg['batch'])
        selected=dict(epoch=ck['epoch'],fine_tune_epoch=ck['fine_tune_epoch'],weights_kind=ck['weights_kind'],
            training=selected_train,validation=selected_val,selection_score=selection_score(selected_val),
            checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),test_evaluations=0)
        write(args.out/'selected_diagnostics.json',selected)
        write(args.out/'selected_validation.json',dict(epoch=ck['epoch'],metrics=selected_val,selection_score=selection_score(selected_val)))
        write(args.out/'status.json',dict(state='completed_validation_only',selected=selected))
    except Exception as exc:
        write(args.out/'status.json',dict(state='failed',error=repr(exc)));raise

if __name__=='__main__':main()
