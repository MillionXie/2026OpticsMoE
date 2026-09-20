"""Train four immutable task-specific D2NNs and evaluate the complete cross-domain matrix."""
import argparse,json,math,platform,subprocess,sys
from pathlib import Path

import numpy as np
import torch

from .continual_d2nn import optimizer_for,train_current_replay_epoch
from .cross_dataset import balanced_subset,load_dataset
from .data import sha
from .joint_d2nn import evaluate
from .model import OpticalD2NN
from .run import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    for name in 'abcd':
        p.add_argument('--task-'+name,type=Path,required=True);p.add_argument('--task-'+name+'-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text());epochs=1 if a.pilot else cfg['epochs']
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'status.json',{'state':'preparing'})
    try:
        torch.set_num_threads(4);raw=[];manifests=[]
        for name in 'abcd':
            data,manifest=load_dataset(getattr(a,'task_'+name),getattr(a,'task_'+name+'_manifest'));raw.append(data);manifests.append(manifest)
        tasks=[]
        for i,(name,data) in enumerate(zip('ABCD',raw)):
            tr=balanced_subset(data['train_labels'],cfg['train_per_class_'+name],cfg['seed']+i)
            va=balanced_subset(data['val_labels'],cfg['val_per_class_'+name],cfg['seed']+10+i)
            tasks.append({'x':torch.from_numpy(data['train_images'][tr]),'y':torch.from_numpy(data['train_labels'][tr]).long(),
                          'vx':torch.from_numpy(data['val_images'][va]),'vy':torch.from_numpy(data['val_labels'][va]).long(),
                          'train_ids':data['train_ids'][tr],'val_ids':data['val_ids'][va]})
        save(a.out/'config.json',cfg);save(a.out/'split.json',{name:{'train_ids':task['train_ids'].tolist(),'val_ids':task['val_ids'].tolist()}
             for name,task in zip('ABCD',tasks)}|{'test_images_read':False})
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save(a.out/'metadata.json',{'command':sys.argv,'commit':commit,'python':sys.version,'torch':torch.__version__,
             'platform':platform.platform(),'device':a.device,'manifests':manifests,
             'data_sha256':{name:sha(getattr(a,'task_'+name.lower())) for name in 'ABCD'},
             'scope':'single-task D2NN transfer pilot' if a.pilot else 'single-task D2NN frozen transfer matrix',
             'test_images_read':False})
        history=[];models={};rngs={name:np.random.default_rng(cfg['seed']+100*i) for i,name in enumerate('ABCD')}
        for i,(name,task) in enumerate(zip('ABCD',tasks)):
            torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);model=OpticalD2NN(cfg).to(a.device);optimizer=optimizer_for(model,cfg)
            stage=a.out/name;stage.mkdir();best=-1.;best_epoch=None
            for epoch in range(1,epochs+1):
                rng=rngs[name];train=train_current_replay_epoch(model,optimizer,task,rng.permutation(len(task['y'])),cfg['batch_size'],rng)
                validation=evaluate(model,task['vx'],task['vy'],cfg['eval_batch_size'])[0];score=validation['balanced_accuracy']
                row={'trained_on':name,'epoch':epoch,'train':train,'validation':validation};history.append(row)
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'trained_on':name,'epoch':epoch,
                       'validation':validation,'selection_score':score}
                torch.save(state,stage/'last_checkpoint.pt')
                if score>best:best=score;best_epoch=epoch;torch.save(state,stage/'best_checkpoint.pt')
                save(a.out/'history.json',history);save(a.out/'status.json',{'state':'training','trained_on':name,'epoch':epoch,
                     'best_epoch':best_epoch,'best_selection_score':best})
                print(json.dumps({'trained_on':name,'epoch':epoch,'train':train,'val_bal_acc':score}),flush=True)
            selected=torch.load(stage/'best_checkpoint.pt',map_location=a.device,weights_only=False);model.load_state_dict(selected['model'])
            models[name]=(model,selected['epoch'],selected['selection_score'])
        matrix={};details={}
        for source,(model,epoch,score) in models.items():
            matrix[source]={};details[source]={'selected_epoch':epoch,'own_selection_score':score}
            for target,task in zip('ABCD',tasks):
                metrics,probabilities=evaluate(model,task['vx'],task['vy'],cfg['eval_batch_size']);matrix[source][target]=metrics['balanced_accuracy']
                details[source][target]=metrics
                np.savez_compressed(a.out/f'{source}_to_{target}_validation_predictions.npz',ids=task['val_ids'],labels=task['vy'].numpy(),
                                    probabilities=probabilities.numpy())
        save(a.out/'metrics.json',{'transfer_balanced_accuracy':matrix,'details':details,'headline_metric':'balanced_accuracy',
             'parameter_count_per_model':sum(p.numel() for p in next(iter(models.values()))[0].parameters()),
             'test_images_read':False,'selection_rule':'each D2NN selected only by its own-domain validation balanced accuracy'})
        save(a.out/'status.json',{'state':'complete'})
    except BaseException as error:
        save(a.out/'status.json',{'state':'failed','error':repr(error)});raise


if __name__=='__main__':main()
