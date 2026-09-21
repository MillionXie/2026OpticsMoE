"""A->B->C->D continual training of one fixed-capacity full-aperture D2NN."""
import argparse,json,math,platform,subprocess,sys
from pathlib import Path

import numpy as np
import torch

from .cross_dataset import balanced_subset,load_dataset
from .data import sha
from .joint_d2nn import binary_metrics,evaluate,joint_epoch_indices
from .model import OpticalD2NN,loss
from .run import save


def optimizer_for(model,cfg):
    return torch.optim.Adam([
        {'params':[model.phase_1],'lr':cfg['lr_expert']},
        {'params':[model.phase_2],'lr':cfg['lr_shared']},
    ])


def train_current_replay_epoch(model,optimizer,current,order,current_batch,rng,replays=()):
    """One current-task pass with fixed per-update samples from each old memory."""
    model.train();total=correct=count=updates=0;device=next(model.parameters()).device
    for start in range(0,len(order),current_batch):
        ids=order[start:start+current_batch];xs=[current['x'][ids]];ys=[current['y'][ids]]
        for old,n in replays:
            rid=rng.choice(len(old['y']),size=n,replace=len(old['y'])<n)
            xs.append(old['x'][rid]);ys.append(old['y'][rid])
        xb=torch.cat(xs).to(device);yb=torch.cat(ys).to(device)
        permutation=torch.from_numpy(rng.permutation(len(yb))).to(device);xb,yb=xb[permutation],yb[permutation]
        optimizer.zero_grad(set_to_none=True);output=model(xb);value=loss(output,yb)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        total+=value.item()*len(yb);correct+=int((output['probabilities'].argmax(1)==yb).sum());count+=len(yb);updates+=1
    return {'nll':total/count,'accuracy_online':correct/count,'samples':count,'updates':updates}


def train_full_replay_epoch(model,optimizer,tasks,batch_size,steps,rng):
    if batch_size%len(tasks):raise ValueError('batch_size must be divisible by number of seen tasks')
    per_task=batch_size//len(tasks);model.train();total=correct=count=0;device=next(model.parameters()).device
    for ids_by_task in joint_epoch_indices([len(t['y']) for t in tasks],per_task,steps,rng):
        xb=torch.cat([task['x'][ids] for task,ids in zip(tasks,ids_by_task)])
        yb=torch.cat([task['y'][ids] for task,ids in zip(tasks,ids_by_task)])
        permutation=torch.from_numpy(rng.permutation(len(yb)));xb,yb=xb[permutation].to(device),yb[permutation].to(device)
        optimizer.zero_grad(set_to_none=True);output=model(xb);value=loss(output,yb)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        total+=value.item()*len(yb);correct+=int((output['probabilities'].argmax(1)==yb).sum());count+=len(yb)
    return {'nll':total/count,'accuracy_online':correct/count,'samples':count,'updates':steps,
            'samples_per_task_per_step':per_task}


def relative_change(before,after):
    delta=torch.linalg.vector_norm((after-before).float())
    scale=torch.linalg.vector_norm(before.float()).clamp_min(1e-12)
    return float(delta/scale)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    for name in 'abcd':
        p.add_argument('--task-'+name,type=Path,required=True)
        p.add_argument('--task-'+name+'-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text());mode=cfg['replay_mode']
    if mode not in ('small','full'):raise ValueError('replay_mode must be small or full')
    if a.pilot:
        for key in ('epochs_A','epochs_adapt_B','epochs_B','epochs_adapt_C','epochs_C','epochs_adapt_D','epochs_D'):cfg[key]=1
        for key in ('steps_A','steps_B','steps_C','steps_D'):
            if key in cfg:cfg[key]=min(4,cfg[key])
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'status.json',{'state':'preparing'})
    try:
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4);rng=np.random.default_rng(cfg['seed'])
        raw=[];manifests=[]
        for name in 'abcd':
            data,manifest=load_dataset(getattr(a,'task_'+name),getattr(a,'task_'+name+'_manifest'));raw.append(data);manifests.append(manifest)
        tasks=[]
        for i,(name,data) in enumerate(zip('ABCD',raw)):
            tr=balanced_subset(data['train_labels'],cfg['train_per_class_'+name],cfg['seed']+i)
            va=balanced_subset(data['val_labels'],cfg['val_per_class_'+name],cfg['seed']+10+i)
            tasks.append({'x':torch.from_numpy(data['train_images'][tr]),'y':torch.from_numpy(data['train_labels'][tr]).long(),
                          'vx':torch.from_numpy(data['val_images'][va]),'vy':torch.from_numpy(data['val_labels'][va]).long(),
                          'train_ids':data['train_ids'][tr],'val_ids':data['val_ids'][va]})
        memories=[]
        if mode=='small':
            for i,task in enumerate(tasks):
                ids=balanced_subset(task['y'].numpy(),cfg['replay_per_class'],cfg['seed']+20+i)
                memories.append({'x':task['x'][ids],'y':task['y'][ids],'ids':task['train_ids'][ids]})
        split={name:{'train_ids':task['train_ids'].tolist(),'val_ids':task['val_ids'].tolist(),
                     **({'replay_ids':memories[i]['ids'].tolist()} if mode=='small' else {})}
               for i,(name,task) in enumerate(zip('ABCD',tasks))}
        split.update(test_images_read=False,replay='fixed small memory' if mode=='small' else 'all selected old-task training samples')
        save(a.out/'config.json',cfg);save(a.out/'split.json',split)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save(a.out/'metadata.json',{'command':sys.argv,'commit':commit,'python':sys.version,'torch':torch.__version__,
             'platform':platform.platform(),'device':a.device,'manifests':manifests,
             'data_sha256':{name:sha(getattr(a,'task_'+name.lower())) for name in 'ABCD'},
             'scope':f'continual D2NN {mode}-replay '+('pilot' if a.pilot else 'validation-selected experiment'),
             'test_images_read':False,'adaptation':'all D2NN phases update on current task; no new parameters exist'})
        model=OpticalD2NN(cfg).to(a.device);history=[];snapshots={};audit=[];total_updates=0
        for group,name in enumerate('ABCD'):
            if group:
                before={n:p.detach().clone() for n,p in model.named_parameters()};optimizer=optimizer_for(model,cfg)
                for epoch in range(1,cfg['epochs_adapt_'+name]+1):
                    train=train_current_replay_epoch(model,optimizer,tasks[group],rng.permutation(len(tasks[group]['y'])),cfg['batch_size'],rng)
                    total_updates+=train['updates']
                    seen={task_name:evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['eval_batch_size'])[0]
                          for j,task_name in enumerate('ABCD'[:group+1])}
                    history.append({'stage':'adapt_'+name,'epoch':epoch,'train':train,'seen':seen})
                    save(a.out/'history.json',history);print(json.dumps({'stage':'adapt_'+name,'epoch':epoch,'train':train,
                          'val_bal_acc':{k:v['balanced_accuracy'] for k,v in seen.items()}}),flush=True)
                audit.append({'stage':'adapt_'+name,'relative_parameter_change':
                              {n:relative_change(before[n],p.detach()) for n,p in model.named_parameters()}})
            stage=a.out/name;stage.mkdir();optimizer=optimizer_for(model,cfg);best=-1.;best_epoch=None
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            for epoch in range(1,cfg['epochs_'+name]+1):
                if mode=='full':
                    train=train_full_replay_epoch(model,optimizer,tasks[:group+1],cfg['batch_size'],cfg['steps_'+name],rng)
                else:
                    if group==0:replay_spec=();current=cfg['batch_size']
                    elif group==1:replay_spec=((memories[0],cfg['replay_A_in_B']),);current=cfg['current_batch_B']
                    else:
                        n=cfg['replay_each_old_in_'+name];replay_spec=tuple((memories[j],n) for j in range(group));current=cfg['current_batch_'+name]
                    train=train_current_replay_epoch(model,optimizer,tasks[group],rng.permutation(len(tasks[group]['y'])),current,rng,replay_spec)
                total_updates+=train['updates']
                seen={task_name:evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['eval_batch_size'])[0]
                      for j,task_name in enumerate('ABCD'[:group+1])}
                score=float(np.mean([v['balanced_accuracy'] for v in seen.values()]))
                row={'stage':name,'epoch':epoch,'train':train,'seen':seen,'selection_score':score};history.append(row)
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'stage':name,'epoch':epoch,
                       'validation':seen,'selection_score':score,'total_updates':total_updates}
                torch.save(state,stage/'last_checkpoint.pt')
                if score>best:best=score;best_epoch=epoch;torch.save(state,stage/'best_checkpoint.pt')
                save(a.out/'history.json',history);save(a.out/'status.json',{'state':'training','stage':name,'epoch':epoch,
                     'best_epoch':best_epoch,'best_selection_score':best})
                print(json.dumps({'stage':name,'epoch':epoch,'train':train,'val_bal_acc':{k:v['balanced_accuracy'] for k,v in seen.items()},
                                  'selection_score':score}),flush=True)
            audit.append({'stage':name,'relative_parameter_change':
                          {n:relative_change(before[n],p.detach()) for n,p in model.named_parameters()}})
            selected=torch.load(stage/'best_checkpoint.pt',map_location=a.device,weights_only=False);model.load_state_dict(selected['model'])
            total_updates=selected['total_updates'];snapshots[name]={'epoch':selected['epoch'],'selection_score':selected['selection_score'],
                                                                    'validation':selected['validation'],'total_updates':total_updates}
        results={'snapshots':snapshots,'selected_stage':'D','selected_epoch':snapshots['D']['epoch'],
                 'selection_score':snapshots['D']['selection_score'],'headline_metric':'balanced_accuracy',
                 'parameter_count':sum(p.numel() for p in model.parameters()),'test_images_read':False,
                 'replay_mode':mode,'total_selected_optimizer_updates':total_updates,
                 'selection_rule':'at each stage, maximum mean validation balanced accuracy over all seen tasks'}
        for name,task in zip('ABCD',tasks):
            train_metrics,_=evaluate(model,task['x'],task['y'],cfg['eval_batch_size'])
            val_metrics,probabilities=evaluate(model,task['vx'],task['vy'],cfg['eval_batch_size'])
            results[name]={'train':train_metrics,'validation':val_metrics,
                           'BWT_after_D':val_metrics['balanced_accuracy']-snapshots[name]['validation'][name]['balanced_accuracy']}
            np.savez_compressed(a.out/f'{name}_validation_predictions.npz',ids=task['val_ids'],labels=task['vy'].numpy(),
                                probabilities=probabilities.numpy())
        save(a.out/'metrics.json',results);save(a.out/'audit.json',audit)
        save(a.out/'status.json',{'state':'complete','selected_epoch':snapshots['D']['epoch']})
    except BaseException as error:
        save(a.out/'status.json',{'state':'failed','error':repr(error)});raise


if __name__=='__main__':main()
