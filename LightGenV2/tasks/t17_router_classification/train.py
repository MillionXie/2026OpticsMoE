"""Full split training, validation-only selection, one final held-out evaluation."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain, sha256


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def encode(images,profile='sixteen_dense'):
    if profile in ('four_top2','four_top2_ccd'):
        from .model_four import encode_rgb
        return encode_rgb(torch.as_tensor(images.copy()))
    rgb = torch.as_tensor(images.copy()).float().permute(0, 3, 1, 2) / 255
    rgb = F.interpolate(rgb, (112, 112), mode='bilinear', align_corners=False)
    return torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                      torch.cat((rgb[:, 2], torch.zeros_like(rgb[:, 2])), -1)), -2)


def shuffle_rng(seed,completed_epochs,samples,state=None):
    rng=np.random.default_rng(seed)
    if state is not None:rng.bit_generator.state=state
    else:
        # Legacy checkpoints did not persist shuffle state. Replay only RNG draws.
        for _ in range(completed_epochs):rng.permutation(samples)
    return rng


def validate_resume(previous,current):
    for key in ('architecture','profile','data_sha256','manifest_sha256','batch','seed',
                'router_features','router_lr','optical_contract'):
        if previous.get(key)!=current.get(key):raise ValueError(f'resume contract mismatch: {key}')


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval()
    predictions, routes = [], []
    loss_sum=0.
    for start in range(0, len(y), batch):
        out = model(x[start:start+batch])
        loss_sum+=float(F.cross_entropy(out['logits'],y[start:start+batch],reduction='sum'))
        predictions.extend(out['logits'].argmax(1).cpu().tolist())
        if out['route_power'] is not None:
            routes.append(out['route_power'].cpu().numpy())
    pred = np.array(predictions)
    truth = y.cpu().numpy()
    cm = np.bincount(truth*9+pred, minlength=81).reshape(9,9)
    recall = cm.diagonal()/cm.sum(1)
    result = dict(accuracy=float((pred==truth).mean()), balanced_accuracy=float(recall.mean()),
                  per_class_recall=recall.tolist(), confusion_matrix=cm.tolist(), n=len(y),
                  cross_entropy=loss_sum/len(y))
    if routes:
        q = np.concatenate(routes)
        result['router'] = dict(mean_power=q.mean(0).tolist(), std_power=q.std(0).tolist(),
            argmax_fraction=(np.bincount(q.argmax(1), minlength=q.shape[1])/len(q)).tolist(),
            selected_fraction=(q>0).mean(0).tolist(),
            mean_selected_count=float((q>0).sum(1).mean()))
    return result, pred, np.concatenate(routes) if routes else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--architecture', choices=['optical','electronic','d2nn'], required=True)
    p.add_argument('--profile',choices=['four_top2','four_top2_ccd','sixteen_dense'],default='four_top2')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--min-epochs', type=int, default=8)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--router-features', choices=['mean','centered'], default='mean')
    p.add_argument('--router-lr', type=float, default=.001)
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--resume-run',type=Path)
    p.add_argument('--train-eval-every',type=int,default=0)
    args=p.parse_args()
    if args.profile.startswith('four_top2') and (args.router_features!='mean' or args.router_lr!=.001):
        raise ValueError('four_top2 fixes standard electronic router features and learning rate')
    args.out.mkdir(parents=True, exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  command=sys.argv, python=platform.python_version(), torch=torch.__version__,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  data_sha256=sha256(args.data), manifest_sha256=sha256(args.manifest))
    write(args.out/'config.json', config)
    write(args.out/'status.json', {'state':'loading'})
    try:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        data, manifest=load_domain(args.data,args.manifest)
        if manifest['domain'] != 'A_original': raise ValueError('original images only')
        if [len(data[s+'_labels']) for s in ('train','val','test')] != [5026,718,1436]:
            raise ValueError('unexpected full split counts')
        device=torch.device('cuda')
        # Test images are encoded only after checkpoint selection.
        x={s:encode(data[s+'_images'],args.profile).to(device) for s in ('train','val')}
        y={s:torch.as_tensor(data[s+'_labels'],device=device,dtype=torch.long) for s in ('train','val')}
        if args.profile.startswith('four_top2'):
            from .model_four import FourRouterClassification,CONTRACT
            model=FourRouterClassification(args.architecture,
                readout='ccd_grid' if args.profile=='four_top2_ccd' else 'linear').to(device)
            config['optical_contract']=CONTRACT
            config['top_k']=2 if args.architecture!='d2nn' else None
            config['readout']=model.readout
            if model.readout=='ccd_grid':
                config['ccd_readout']=dict(edges_active_pixels=[0,159,318,478],
                    class_order='row-major 0..8',gap_pixels=0,coverage=1.,
                    loss='cross_entropy(log(normalized_region_energy + relative epsilon))',
                    relative_epsilon=1e-12,trainable_readout_parameters=0)
        else:
            from .model import RouterClassification
            model=RouterClassification(args.architecture,args.router_features).to(device)
        groups=[]
        for name,param in model.named_parameters():
            lr=.001 if 'shared_head' in name or 'electronic_router' in name else .005
            if 'electronic_router' in name: lr=args.router_lr
            groups.append({'params':[param], 'lr':lr})
        opt=torch.optim.Adam(groups)
        config['parameters']={name:sum(v.numel() for n,v in model.named_parameters() if name in n)
            for name in ('router_phase','electronic_router','first_phase','global_phase','shared_head')}
        config['total_parameters']=sum(v.numel() for v in model.parameters())
        write(args.out/'config.json',config)
        history=[]; best=-1.; stale=0; completed=0; rng_state=None
        if args.resume_run is not None:
            if args.profile!='four_top2':raise ValueError('continuation currently restricted to Linear four_top2')
            parent=args.resume_run
            previous=json.loads((parent/'config.json').read_text())
            validate_resume(previous,config)
            history=json.loads((parent/'metrics.json').read_text())
            ck=torch.load(parent/'last_checkpoint.pt',map_location=device,weights_only=False)
            completed=int(ck['epoch'])
            if history[-1]['epoch']!=completed or args.epochs<=completed:
                raise ValueError('invalid completed epoch or total training budget')
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            rng_state=ck.get('shuffle_rng_state')
            best=max(row['validation']['balanced_accuracy'] for row in history)
            best_ck=torch.load(parent/'best_checkpoint.pt',map_location=device,weights_only=False)
            config['resume_source']=dict(run=str(parent),completed_epoch=completed,
                last_sha256=sha256(parent/'last_checkpoint.pt'),
                best_sha256=sha256(parent/'best_checkpoint.pt'),source_git_commit=previous['git_commit'],
                optimizer_restored=True,shuffle_state='saved' if rng_state else 'reconstructed')
            write(args.out/'config.json',config)
            shutil.copyfile(parent/'best_checkpoint.pt',args.out/'best_checkpoint.pt')
            diagnosis={}
            for name,state in (('last',ck),('best',best_ck)):
                model.load_state_dict(state['model'])
                train_metrics,_,_=evaluate(model,x['train'],y['train'],args.batch)
                val_metrics,_,_=evaluate(model,x['val'],y['val'],args.batch)
                diagnosis[name]=dict(epoch=state['epoch'],training=train_metrics,validation=val_metrics)
            write(args.out/'resume_diagnostics.json',diagnosis)
            model.load_state_dict(ck['model'])
            del ck,best_ck
        rng=shuffle_rng(args.seed,completed,len(y['train']),rng_state)
        for epoch in range(completed+1,args.epochs+1):
            begin=time.time(); model.train(); loss_sum=0.
            indices=rng.permutation(len(y['train']))
            for start in range(0,len(indices),args.batch):
                ix=torch.as_tensor(indices[start:start+args.batch],device=device)
                opt.zero_grad(set_to_none=True)
                out=model(x['train'][ix]); loss=F.cross_entropy(out['logits'],y['train'][ix])
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step(); loss_sum+=float(loss.detach())*len(ix)
            metrics,_,_=evaluate(model,x['val'],y['val'],args.batch)
            score=metrics['balanced_accuracy']; improved=score>best
            if improved: best=score; stale=0
            else: stale+=1
            row={'epoch':epoch,'train_loss':loss_sum/len(indices),'validation':metrics,
                 'seconds':time.time()-begin,'selected':improved}
            if args.train_eval_every>0 and (epoch%args.train_eval_every==0 or epoch==args.epochs):
                row['training'],_,_=evaluate(model,x['train'],y['train'],args.batch)
                row['seconds']=time.time()-begin
            history.append(row); write(args.out/'metrics.json',history)
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':config,
                'shuffle_rng_state':rng.bit_generator.state}
            torch.save(ck,args.out/'last_checkpoint.pt')
            if improved: torch.save(ck,args.out/'best_checkpoint.pt')
            write(args.out/'status.json',{'state':'training','epoch':epoch,'best_validation':best})
            print(json.dumps(row),flush=True)
            if epoch>=args.min_epochs and stale>=args.patience: break
        chosen=torch.load(args.out/'best_checkpoint.pt',map_location=device,weights_only=False)
        model.load_state_dict(chosen['model'])
        if args.train_eval_every>0:
            selected_train,_,_=evaluate(model,x['train'],y['train'],args.batch)
            selected_val,_,_=evaluate(model,x['val'],y['val'],args.batch)
            write(args.out/'selected_diagnostics.json',dict(epoch=chosen['epoch'],
                training=selected_train,validation=selected_val))
        if args.skip_test:
            selected,_,_=evaluate(model,x['val'],y['val'],args.batch)
            write(args.out/'selected_validation.json',dict(epoch=chosen['epoch'],metrics=selected))
            write(args.out/'status.json',{'state':'completed_validation_only','epoch':chosen['epoch'],
                                          'validation':selected})
            return
        if args.resume_run is not None:
            old_result=json.loads((args.resume_run/'result.json').read_text())
            if sha256(args.out/'best_checkpoint.pt')==old_result['checkpoint_sha256']:
                result=dict(old_result)
                result.update(test_evaluations=0,test_receipt_reused_from=str(args.resume_run),
                              interpretation='unchanged previously tested checkpoint; no new test inference')
                shutil.copyfile(args.resume_run/'test_predictions.npz',args.out/'test_predictions.npz')
                write(args.out/'result.json',result)
                write(args.out/'status.json',{'state':'completed','result':result})
                return
        xt=encode(data['test_images'],args.profile).to(device)
        yt=torch.as_tensor(data['test_labels'],device=device,dtype=torch.long)
        result,pred,q=evaluate(model,xt,yt,args.batch)
        result.update(selected_epoch=chosen['epoch'],checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),
                      test_evaluations=1)
        if args.resume_run is not None:
            result['interpretation']='extended-budget exploratory; original baseline test previously seen'
        np.savez_compressed(args.out/'test_predictions.npz',ids=data['test_ids'],labels=data['test_labels'],
                            predictions=pred,route_power=q if q is not None else np.empty((len(pred),0)))
        write(args.out/'result.json',result)
        write(args.out/'status.json',{'state':'completed','result':result})
    except Exception as exc:
        write(args.out/'status.json',{'state':'failed','error':repr(exc)})
        raise

if __name__=='__main__': main()
