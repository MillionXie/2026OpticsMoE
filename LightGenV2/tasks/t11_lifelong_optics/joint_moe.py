"""Offline joint four-pathology-dataset training for the 16-expert optical MoE."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .cross_dataset import balanced_subset, evaluate, load_dataset
from .data import sha
from .joint_d2nn import joint_epoch_indices
from .model import OpticalMoE, loss
from .run import save


def train_epoch(model, optimizer, tasks, per_task, steps, rng):
    model.train(); total = correct = count = 0
    for ids_by_task in joint_epoch_indices([len(t['y']) for t in tasks], per_task, steps, rng):
        xb = torch.cat([task['x'][ids] for task, ids in zip(tasks, ids_by_task)])
        yb = torch.cat([task['y'][ids] for task, ids in zip(tasks, ids_by_task)])
        order = torch.from_numpy(rng.permutation(len(yb))); xb, yb = xb[order], yb[order]
        device = next(model.parameters()).device; xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True); output = model(xb); value = loss(output, yb)
        if not torch.isfinite(value): raise RuntimeError('Nonfinite loss')
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); total += value.item()*len(yb)
        correct += int((output['probabilities'].argmax(1) == yb).sum()); count += len(yb)
    return {'nll': total/count, 'accuracy_online': correct/count, 'samples': count}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    for name in 'abcd':
        parser.add_argument('--task-'+name,type=Path,required=True)
        parser.add_argument('--task-'+name+'-manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if cfg.get('num_experts')!=16:raise ValueError('Joint MoE requires the fixed 16-expert canvas')
    if cfg['batch_size']%4:raise ValueError('batch_size must be divisible by four')
    if args.pilot:cfg['epochs']=1;cfg['steps_per_epoch']=min(4,cfg['steps_per_epoch'])
    args.out.mkdir(parents=True,exist_ok=False);save(args.out/'status.json',{'state':'preparing'})
    try:
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4);rng=np.random.default_rng(cfg['seed'])
        raw=[];manifests=[]
        for name in 'abcd':
            data,manifest=load_dataset(getattr(args,'task_'+name),getattr(args,'task_'+name+'_manifest'))
            raw.append(data);manifests.append(manifest)
        tasks=[]
        for i,(name,data) in enumerate(zip('ABCD',raw)):
            tr=balanced_subset(data['train_labels'],cfg['train_per_class_'+name],cfg['seed']+i)
            va=balanced_subset(data['val_labels'],cfg['val_per_class_'+name],cfg['seed']+10+i)
            tasks.append({'x':torch.from_numpy(data['train_images'][tr]),'y':torch.from_numpy(data['train_labels'][tr]).long(),
                          'vx':torch.from_numpy(data['val_images'][va]),'vy':torch.from_numpy(data['val_labels'][va]).long(),
                          'train_ids':data['train_ids'][tr],'val_ids':data['val_ids'][va]})
        save(args.out/'config.json',cfg);save(args.out/'split.json',{
            name:{'train_ids':t['train_ids'].tolist(),'val_ids':t['val_ids'].tolist()} for name,t in zip('ABCD',tasks)
        }|{'test_images_read':False})
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save(args.out/'metadata.json',{'command':sys.argv,'commit':commit,'python':sys.version,'torch':torch.__version__,
             'platform':platform.platform(),'device':args.device,'manifests':manifests,
             'data_sha256':{name:sha(getattr(args,'task_'+name.lower())) for name in 'ABCD'},
             'scope':'joint MoE pilot' if args.pilot else 'joint MoE validation-selected experiment',
             'test_images_read':False,'sampling':{'tasks_per_batch':4,'samples_per_task':cfg['batch_size']//4,
                                                   'steps_per_epoch':cfg['steps_per_epoch']}})
        model=OpticalMoE(cfg).to(args.device);model.configure_all()
        optimizer=torch.optim.Adam([{'params':list(model.experts),'lr':cfg['lr_expert']},
                                    {'params':[model.router,model.global_phase],'lr':cfg['lr_shared']}])
        history=[];best=-1.;best_epoch=None;per_task=cfg['batch_size']//4
        for epoch in range(1,cfg['epochs']+1):
            train=train_epoch(model,optimizer,tasks,per_task,cfg['steps_per_epoch'],rng)
            validation={name:evaluate(model,t['vx'],t['vy'],cfg['eval_batch_size'])[0] for name,t in zip('ABCD',tasks)}
            score=float(np.mean([v['balanced_accuracy'] for v in validation.values()]))
            row={'epoch':epoch,'train':train,'validation':validation,'selection_score':score};history.append(row)
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'epoch':epoch,
                   'validation':validation,'selection_score':score}
            torch.save(state,args.out/'last_checkpoint.pt')
            if score>best:best=score;best_epoch=epoch;torch.save(state,args.out/'best_checkpoint.pt')
            save(args.out/'history.json',history);save(args.out/'status.json',{'state':'training','epoch':epoch,
                 'best_epoch':best_epoch,'best_selection_score':best})
            print(json.dumps({'epoch':epoch,'train':train,'val_bal_acc':{k:v['balanced_accuracy'] for k,v in validation.items()},
                              'selection_score':score,'mean_route':{k:v['mean_route'] for k,v in validation.items()}}),flush=True)
        selected=torch.load(args.out/'best_checkpoint.pt',map_location=args.device,weights_only=False);model.load_state_dict(selected['model'])
        results={'selected_epoch':selected['epoch'],'selection_score':selected['selection_score'],
                 'headline_metric':'balanced_accuracy','parameter_count':sum(p.numel() for p in model.parameters()),
                 'test_images_read':False,'selection_rule':'maximum mean validation balanced accuracy across A/B/C/D'}
        for name,t in zip('ABCD',tasks):
            train_metrics,_,_=evaluate(model,t['x'],t['y'],cfg['eval_batch_size'])
            val_metrics,p,q=evaluate(model,t['vx'],t['vy'],cfg['eval_batch_size'])
            results[name]={'train':train_metrics,'validation':val_metrics}
            np.savez_compressed(args.out/f'{name}_validation_predictions.npz',ids=t['val_ids'],labels=t['vy'].numpy(),
                                probabilities=p.numpy(),routes=q.numpy())
        save(args.out/'metrics.json',results);save(args.out/'status.json',{'state':'complete','selected_epoch':selected['epoch']})
    except BaseException as error:
        save(args.out/'status.json',{'state':'failed','error':repr(error)});raise


if __name__=='__main__':main()
