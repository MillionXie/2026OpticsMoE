"""Full split training, validation-only selection, one final held-out evaluation."""
import argparse
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
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain, sha256


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def encode(images,profile='sixteen_dense'):
    if profile=='four_top2':
        from .model_four import encode_rgb
        return encode_rgb(torch.as_tensor(images.copy()))
    rgb = torch.as_tensor(images.copy()).float().permute(0, 3, 1, 2) / 255
    rgb = F.interpolate(rgb, (112, 112), mode='bilinear', align_corners=False)
    return torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                      torch.cat((rgb[:, 2], torch.zeros_like(rgb[:, 2])), -1)), -2)


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval()
    predictions, routes = [], []
    for start in range(0, len(y), batch):
        out = model(x[start:start+batch])
        predictions.extend(out['logits'].argmax(1).cpu().tolist())
        if out['route_power'] is not None:
            routes.append(out['route_power'].cpu().numpy())
    pred = np.array(predictions)
    truth = y.cpu().numpy()
    cm = np.bincount(truth*9+pred, minlength=81).reshape(9,9)
    recall = cm.diagonal()/cm.sum(1)
    result = dict(accuracy=float((pred==truth).mean()), balanced_accuracy=float(recall.mean()),
                  per_class_recall=recall.tolist(), confusion_matrix=cm.tolist(), n=len(y))
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
    p.add_argument('--profile',choices=['four_top2','sixteen_dense'],default='four_top2')
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
    args=p.parse_args()
    if args.profile=='four_top2' and (args.router_features!='mean' or args.router_lr!=.001):
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
        if args.profile=='four_top2':
            from .model_four import FourRouterClassification,CONTRACT
            model=FourRouterClassification(args.architecture).to(device)
            config['optical_contract']=CONTRACT
            config['top_k']=2 if args.architecture!='d2nn' else None
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
        history=[]; best=-1.; stale=0
        rng=np.random.default_rng(args.seed)
        for epoch in range(1,args.epochs+1):
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
            history.append(row); write(args.out/'metrics.json',history)
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':config}
            torch.save(ck,args.out/'last_checkpoint.pt')
            if improved: torch.save(ck,args.out/'best_checkpoint.pt')
            write(args.out/'status.json',{'state':'training','epoch':epoch,'best_validation':best})
            print(json.dumps(row),flush=True)
            if epoch>=args.min_epochs and stale>=args.patience: break
        chosen=torch.load(args.out/'best_checkpoint.pt',map_location=device,weights_only=False)
        model.load_state_dict(chosen['model'])
        if args.skip_test:
            selected,_,_=evaluate(model,x['val'],y['val'],args.batch)
            write(args.out/'selected_validation.json',dict(epoch=chosen['epoch'],metrics=selected))
            write(args.out/'status.json',{'state':'completed_validation_only','epoch':chosen['epoch'],
                                          'validation':selected})
            return
        xt=encode(data['test_images'],args.profile).to(device)
        yt=torch.as_tensor(data['test_labels'],device=device,dtype=torch.long)
        result,pred,q=evaluate(model,xt,yt,args.batch)
        result.update(selected_epoch=chosen['epoch'],checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),
                      test_evaluations=1)
        np.savez_compressed(args.out/'test_predictions.npz',ids=data['test_ids'],labels=data['test_labels'],
                            predictions=pred,route_power=q if q is not None else np.empty((len(pred),0)))
        write(args.out/'result.json',result)
        write(args.out/'status.json',{'state':'completed','result':result})
    except Exception as exc:
        write(args.out/'status.json',{'state':'failed','error':repr(exc)})
        raise

if __name__=='__main__': main()
