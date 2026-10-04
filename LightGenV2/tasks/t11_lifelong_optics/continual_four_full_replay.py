"""A->B->C->D optical MoE with full old-dataset replay at every main stage."""
import argparse,json,math,platform,subprocess,sys
from pathlib import Path
import numpy as np,torch

from .continual_three_dataset import optimizer_for,train_epoch as train_current_epoch
from .cross_dataset import balanced_subset,evaluate,load_dataset
from .data import sha
from .joint_d2nn import joint_epoch_indices
from .model import OpticalMoE,loss
from .run import save


def train_full_replay_epoch(model,optimizer,tasks,batch_size,steps,rng):
    if batch_size%len(tasks):raise ValueError('batch_size must be divisible by the number of seen tasks')
    per_task=batch_size//len(tasks);model.train();total=correct=count=0
    for ids_by_task in joint_epoch_indices([len(t['y']) for t in tasks],per_task,steps,rng):
        xb=torch.cat([task['x'][ids] for task,ids in zip(tasks,ids_by_task)])
        yb=torch.cat([task['y'][ids] for task,ids in zip(tasks,ids_by_task)])
        order=torch.from_numpy(rng.permutation(len(yb)));xb,yb=xb[order],yb[order]
        device=next(model.parameters()).device;xb,yb=xb.to(device),yb.to(device)
        optimizer.zero_grad(set_to_none=True);output=model(xb);value=loss(output,yb)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        total+=value.item()*len(yb);correct+=int((output['probabilities'].argmax(1)==yb).sum());count+=len(yb)
    return {'nll':total/count,'accuracy_online':correct/count,'samples':count,'samples_per_task_per_step':per_task}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    for name in 'abcd':p.add_argument('--task-'+name,type=Path,required=True);p.add_argument('--task-'+name+'-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if cfg.get('num_experts')!=16:raise ValueError('Full-replay protocol requires 16 experts')
    if a.pilot:
        for key in ('epochs_A','epochs_warmup_B','epochs_B','epochs_warmup_C','epochs_C','epochs_warmup_D','epochs_D'):cfg[key]=1
        for key in ('steps_A','steps_B','steps_C','steps_D'):cfg[key]=min(4,cfg[key])
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
        save(a.out/'config.json',cfg);save(a.out/'split.json',{
          name:{'train_ids':t['train_ids'].tolist(),'val_ids':t['val_ids'].tolist()} for name,t in zip('ABCD',tasks)
        }|{'test_images_read':False,'replay':'all selected old-task training samples'})
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save(a.out/'metadata.json',{'command':sys.argv,'commit':commit,'python':sys.version,'torch':torch.__version__,
             'platform':platform.platform(),'device':a.device,'manifests':manifests,
             'data_sha256':{name:sha(getattr(a,'task_'+name.lower())) for name in 'ABCD'},
             'scope':'full-replay MoE pilot' if a.pilot else 'full-replay MoE validation-selected experiment',
             'test_images_read':False,'replay':'full selected training subset of every old task; task-balanced batches'})
        model=OpticalMoE(cfg).to(a.device);history=[];audit=[];snapshots={}
        for group,name in enumerate('ABCD'):
            if group:
                model.configure_group(group,warmup=True);frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
                opt=optimizer_for(model,cfg,False)
                for epoch in range(1,cfg['epochs_warmup_'+name]+1):
                    value=train_current_epoch(model,opt,tasks[group]['x'],tasks[group]['y'],rng.permutation(len(tasks[group]['y'])),
                                              cfg['batch_size'],rng,warmup=True)
                    old_mask=[i < 4*group for i in range(cfg['num_experts'])]
                    old_validation={old:evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['eval_batch_size'],mask=old_mask)[0]
                                    for j,old in enumerate('ABCD'[:group])}
                    new_validation=evaluate(model,tasks[group]['vx'],tasks[group]['vy'],cfg['eval_batch_size'],warmup=True)[0]
                    history.append({'stage':'warmup_'+name,'epoch':epoch,'train_nll':value,
                                    'old_prefix':old_validation,'new_uniform':new_validation})
                    save(a.out/'history.json',history);print(json.dumps({'stage':'warmup_'+name,'epoch':epoch,'loss':value,
                          'old_bal_acc':{k:v['balanced_accuracy'] for k,v in old_validation.items()},
                          'new_bal_acc':new_validation['balanced_accuracy']}),flush=True)
                for n,pv in model.named_parameters():
                    if n in frozen and not torch.equal(pv,frozen[n]):raise RuntimeError('Frozen changed '+n)
                audit.append({'stage':'warmup_'+name,'frozen_unchanged':list(frozen)})
            model.configure_group(group,warmup=False);frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            opt=optimizer_for(model,cfg,True);stage_dir=a.out/name;stage_dir.mkdir();best=-1.;best_epoch=None
            for epoch in range(1,cfg['epochs_'+name]+1):
                train=train_full_replay_epoch(model,opt,tasks[:group+1],cfg['batch_size'],cfg['steps_'+name],rng)
                validation={task_name:evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['eval_batch_size'])[0]
                            for j,task_name in enumerate('ABCD'[:group+1])}
                score=float(np.mean([v['balanced_accuracy'] for v in validation.values()]))
                row={'stage':name,'epoch':epoch,'train':train,'validation':validation,'selection_score':score};history.append(row)
                state={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'stage':name,'epoch':epoch,
                       'validation':validation,'selection_score':score}
                torch.save(state,stage_dir/'last_checkpoint.pt')
                if score>best:best=score;best_epoch=epoch;torch.save(state,stage_dir/'best_checkpoint.pt')
                save(a.out/'history.json',history);save(a.out/'status.json',{'state':'training','stage':name,'epoch':epoch,
                     'best_epoch':best_epoch,'best_selection_score':best})
                print(json.dumps({'stage':name,'epoch':epoch,'train':train,'val_bal_acc':{k:v['balanced_accuracy'] for k,v in validation.items()},
                                  'selection_score':score}),flush=True)
            for n,pv in model.named_parameters():
                if n in frozen and not torch.equal(pv,frozen[n]):raise RuntimeError('Frozen changed '+n)
            audit.append({'stage':name,'frozen_unchanged':list(frozen)})
            selected=torch.load(stage_dir/'best_checkpoint.pt',map_location=a.device,weights_only=False);model.load_state_dict(selected['model'])
            snapshots[name]={'epoch':selected['epoch'],'selection_score':selected['selection_score'],'validation':selected['validation']}
        results={'snapshots':snapshots,'selected_stage':'D','selected_epoch':snapshots['D']['epoch'],
                 'selection_score':snapshots['D']['selection_score'],'headline_metric':'balanced_accuracy',
                 'parameter_count':sum(p.numel() for p in model.parameters()),'test_images_read':False,
                 'selection_rule':'at each stage, maximum mean validation balanced accuracy over all seen tasks'}
        for name,t in zip('ABCD',tasks):
            train_metrics,_,_=evaluate(model,t['x'],t['y'],cfg['eval_batch_size'])
            val_metrics,pv,q=evaluate(model,t['vx'],t['vy'],cfg['eval_batch_size'])
            results[name]={'train':train_metrics,'validation':val_metrics}
            results[name]['BWT_after_D']=val_metrics['balanced_accuracy']-snapshots[name]['validation'][name]['balanced_accuracy']
            np.savez_compressed(a.out/f'{name}_validation_predictions.npz',ids=t['val_ids'],labels=t['vy'].numpy(),
                                probabilities=pv.numpy(),routes=q.numpy())
        save(a.out/'metrics.json',results);save(a.out/'audit.json',audit);save(a.out/'status.json',{'state':'complete','selected_epoch':snapshots['D']['epoch']})
    except BaseException as error:
        save(a.out/'status.json',{'state':'failed','error':repr(error)});raise


if __name__=='__main__':main()
