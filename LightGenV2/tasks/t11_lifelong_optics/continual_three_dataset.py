"""A->B->C optical continual learning with frozen expert memories and old-task replay."""
import argparse,json,platform,subprocess,sys
from pathlib import Path
import numpy as np,torch
from .cross_dataset import load_dataset,balanced_subset,evaluate
from .model import OpticalMoE,loss
from .run import save

GROUP_MASKS=[[True]*4+[False]*8,[False]*4+[True]*4+[False]*4,[False]*8+[True]*4]
PREFIX_MASKS=[[True]*4+[False]*8,[True]*8+[False]*4,[True]*12]

def train_epoch(model,opt,x,y,order,current_batch,rng,replays=(),warmup=False):
    model.train(); total=count=0
    for start in range(0,len(order),current_batch):
        ids=order[start:start+current_batch]; xs=[x[ids]]; ys=[y[ids]]
        for rx,ry,n in replays:
            rid=rng.choice(len(ry),size=n,replace=len(ry)<n); xs.append(rx[rid]);ys.append(ry[rid])
        xb=torch.cat(xs).to(next(model.parameters()).device);yb=torch.cat(ys).to(xb.device)
        opt.zero_grad(set_to_none=True); value=loss(model(xb,warmup=warmup),yb)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=value.item()*len(yb);count+=len(yb)
    return total/count

def optimizer_for(model,cfg,shared):
    groups=[dict(params=[p for p in model.experts if p.requires_grad],lr=cfg['lr_expert'])]
    if shared:groups.append(dict(params=[model.router,model.global_phase],lr=cfg['lr_shared']))
    return torch.optim.Adam(groups)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    for name in 'abc':p.add_argument('--task-'+name,type=Path,required=True);p.add_argument('--task-'+name+'-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.pilot:
        for key in ('epochs_A','epochs_warmup_B','epochs_B','epochs_warmup_C','epochs_C'):cfg[key]=1
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'status.json',{'state':'preparing'})
    try:
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4);rng=np.random.default_rng(cfg['seed'])
        raw=[];manifests=[]
        for name in 'abc':
            d,m=load_dataset(getattr(a,'task_'+name),getattr(a,'task_'+name+'_manifest'));raw.append(d);manifests.append(m)
        tasks=[]
        for i,(name,d) in enumerate(zip('ABC',raw)):
            tr=balanced_subset(d['train_labels'],cfg['train_per_class_'+name],cfg['seed']+i)
            va=balanced_subset(d['val_labels'],cfg['val_per_class_'+name],cfg['seed']+10+i)
            tasks.append(dict(x=torch.from_numpy(d['train_images'][tr]),y=torch.from_numpy(d['train_labels'][tr]).long(),vx=torch.from_numpy(d['val_images'][va]),vy=torch.from_numpy(d['val_labels'][va]).long(),train_ids=d['train_ids'][tr],val_ids=d['val_ids'][va]))
        replays=[]
        for i,t in enumerate(tasks):
            rid=balanced_subset(t['y'].numpy(),cfg['replay_per_class'],cfg['seed']+20+i);replays.append((t['x'][rid],t['y'][rid],t['train_ids'][rid]))
        save(a.out/'config.json',cfg);save(a.out/'split.json',{name:{'train_ids':t['train_ids'].tolist(),'val_ids':t['val_ids'].tolist(),'replay_ids':replays[i][2].tolist()} for i,(name,t) in enumerate(zip('ABC',tasks))})
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();save(a.out/'metadata.json',{'command':sys.argv,'commit':commit,'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'device':a.device,'manifests':manifests,'scope':'three-task pilot' if a.pilot else 'three-task validation-selected experiment','test_images_read':False})
        model=OpticalMoE(cfg).to(a.device);geometry={k:list(v.shape) for k,v in model.state_dict().items()};history=[];audit=[];snapshots={}
        for group,name in enumerate('ABC'):
            if group:
                model.configure_group(group,warmup=True);frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad};opt=optimizer_for(model,cfg,False)
                for epoch in range(1,cfg['epochs_warmup_'+name]+1):
                    value=train_epoch(model,opt,tasks[group]['x'],tasks[group]['y'],rng.permutation(len(tasks[group]['y'])),cfg['batch_size'],rng,warmup=True)
                    old={old:evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['batch_size'],PREFIX_MASKS[group-1])[0] for j,old in enumerate('ABC'[:group])}
                    new=evaluate(model,tasks[group]['vx'],tasks[group]['vy'],cfg['batch_size'],warmup=True)[0];row={'stage':'warmup_'+name,'epoch':epoch,'loss':value,'old_prefix':old,'new_uniform':new};history.append(row);save(a.out/'history.json',history);print(json.dumps({'stage':'warmup_'+name,'epoch':epoch,'loss':value,'old_bal_acc':{k:v['balanced_accuracy'] for k,v in old.items()},'new_bal_acc':new['balanced_accuracy']}),flush=True)
                for n,pv in model.named_parameters():
                    if n in frozen and not torch.equal(pv,frozen[n]):raise RuntimeError('Frozen changed '+n)
                audit.append({'stage':'warmup_'+name,'frozen_unchanged':list(frozen),'geometry_unchanged':geometry=={k:list(v.shape) for k,v in model.state_dict().items()}})
            model.configure_group(group,warmup=False);frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad};opt=optimizer_for(model,cfg,True);stage=a.out/name;stage.mkdir();best=-1.
            if group==0: replay_spec=();current=cfg['batch_size']
            elif group==1: replay_spec=((replays[0][0],replays[0][1],cfg['replay_A_in_B']),);current=cfg['current_batch_B']
            else: replay_spec=tuple((replays[j][0],replays[j][1],cfg['replay_each_old_in_C']) for j in range(2));current=cfg['current_batch_C']
            for epoch in range(1,cfg['epochs_'+name]+1):
                value=train_epoch(model,opt,tasks[group]['x'],tasks[group]['y'],rng.permutation(len(tasks[group]['y'])),current,rng,replay_spec)
                seen={task:evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['batch_size'])[0] for j,task in enumerate('ABC'[:group+1])};score=float(np.mean([v['balanced_accuracy'] for v in seen.values()]));row={'stage':name,'epoch':epoch,'loss':value,'seen':seen,'selection_score':score};history.append(row)
                state={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'stage':name,'epoch':epoch,'validation':row};torch.save(state,stage/'last_checkpoint.pt')
                if score>best:best=score;torch.save(state,stage/'best_checkpoint.pt')
                save(a.out/'history.json',history);save(a.out/'status.json',{'state':'training','stage':name,'epoch':epoch});print(json.dumps({'stage':name,'epoch':epoch,'loss':value,'bal_acc':{k:v['balanced_accuracy'] for k,v in seen.items()},'score':score}),flush=True)
            for n,pv in model.named_parameters():
                if n in frozen and not torch.equal(pv,frozen[n]):raise RuntimeError('Frozen changed '+n)
            audit.append({'stage':name,'frozen_unchanged':list(frozen),'geometry_unchanged':geometry=={k:list(v.shape) for k,v in model.state_dict().items()}});state=torch.load(stage/'best_checkpoint.pt',map_location=a.device,weights_only=False);model.load_state_dict(state['model']);snapshots[name]={'epoch':state['epoch'],'selection_score':state['validation']['selection_score'],'metrics':state['validation']['seen']}
        results={'snapshots':snapshots,'headline_metric':'balanced_accuracy'}
        for j,name in enumerate('ABC'):
            for label,mask in [('all',None),('own_group',GROUP_MASKS[j]),('old_prefix',PREFIX_MASKS[j])]:results[name+'_'+label]=evaluate(model,tasks[j]['vx'],tasks[j]['vy'],cfg['batch_size'],mask)[0]
        results['A_BWT_after_C']=results['A_all']['balanced_accuracy']-snapshots['A']['metrics']['A']['balanced_accuracy'];results['B_BWT_after_C']=results['B_all']['balanced_accuracy']-snapshots['B']['metrics']['B']['balanced_accuracy'];results['interpretation']='Masks alter coherent interference; own_group and prefix results are diagnostics.'
        save(a.out/'metrics.json',results);save(a.out/'audit.json',audit);save(a.out/'status.json',{'state':'complete'})
    except BaseException as e:save(a.out/'status.json',{'state':'failed','error':repr(e)});raise

if __name__=='__main__':main()
