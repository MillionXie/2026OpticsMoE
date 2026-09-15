"""Paired depth/OEO ablation; validation selection precedes any final test load."""
import argparse,csv,hashlib,json,os,random,sys,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.nn import functional as F
from experiments import ROOT,EXP,VARIANTS,CONFIGS,TOTAL_RUNS,run_dir,parameters
from models import build,objective,probabilities
from metrics import metrics,better

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_tensor(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def save_torch(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(p)
def csvwrite(p,rows):
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def snapshot():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and p.suffix in {'.py','.yaml'} and '__pycache__' not in p.parts
            and not any(x in p.relative_to(ROOT).parts for x in ['runs','reports','protocol'])}
def status(**kw):save(ROOT/'status.json',{'updated_at':now(),**kw})
def setseed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
def epoch_order(seed,epoch,n):return torch.randperm(n,generator=torch.Generator().manual_seed(seed*1_000_003+epoch))
def setup():
    torch.set_num_threads(EXP['num_threads']);assert torch.cuda.is_available()
    assert sha(EXP['data_npz'])==EXP['data_sha256']
def getdata(split):
    if split=='test':
        lock=read(ROOT/'protocol/final_lock.json');assert lock['source_hashes']==snapshot()
        assert sha(ROOT/'protocol/activation_selection.json')==lock['activation_selection_sha256']
        for rel,digest in lock['thresholds'].items():assert sha(ROOT/rel)==digest
        assert len(lock['checkpoints'])==TOTAL_RUNS
        for rel,digest in lock['checkpoints'].items():assert sha(ROOT/rel)==digest
    with np.load(EXP['data_npz'],allow_pickle=False) as z:
        x=z[split+'_images'].copy();y=z[split+'_labels'].reshape(-1).copy();ids=z[split+'_ids'].copy()
    assert x.shape==(len(y),28,28) and x.dtype==np.float32
    assert np.isfinite(x).all() and x.min()>=0 and x.max()<=1
    assert np.bincount(y,minlength=2).tolist()=={'train':[929,259],'val':[76,22],'test':[229,69]}[split]
    assert len(set(ids))==len(ids)
    xt=F.interpolate(torch.from_numpy(x[:,None]),size=(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).cuda()
    return xt,torch.from_numpy(y).long().cuda(),ids
def batches(data,order=None):
    x,y,ids=data
    if order is None:order=torch.arange(len(y))
    for b in order.split(EXP['batch_size']):yield x[b],y[b],b

@torch.no_grad()
def evaluate(model,data,predictions=False):
    model.eval();ps=[];losses=[];routes=[];soft=[];powers=[]
    for x,y,_ in batches(data):
        out=model(x);loss=objective(out,y,model.masks);assert torch.isfinite(loss)
        losses.append((loss.item(),len(y)));ps.append(probabilities(out).cpu().numpy())
        if out['routing'] is not None:
            routes.append(out['routing'].cpu().numpy());soft.append(out['route_probabilities'].cpu().numpy())
            powers.append(out['stage_input_power'].cpu().numpy())
    p=np.concatenate(ps);y=data[1].cpu().numpy();result=metrics(y,p)
    result['detector_plane_mse']=sum(v*n for v,n in losses)/len(y);routing=None
    if routes:
        r=np.concatenate(routes);sp=np.concatenate(soft);power=np.concatenate(powers)
        assert (r.sum(1)==9).all() and np.isfinite(sp).all() and (sp>0).all()
        assert np.allclose(sp.sum(1),1,atol=1e-6) and np.isfinite(power).all()
        routing={'hard_counts':r.sum(0).tolist(),'mean_probabilities':sp.mean(0).tolist(),
                 'cycle_expert_mean_input_power':power.mean(0).tolist(),
                 'cycle_expert_nonzero_input_counts':(power>0).sum(0).tolist()}
    result['routing']=routing
    rows=[{'sample_id':str(sid),'label_true':int(label),'label_pred':int(prob[1]>.5),
           'score0':float(prob[0]),'score1':float(prob[1])} for sid,label,prob in zip(data[2],y,p)] if predictions else None
    return result,rows

def train_one(v,seed,train,val,index,source_hashes):
    dest=run_dir(v,seed)
    if (dest/'completed.json').exists():
        assert sha(dest/'best.pt')==read(dest/'selection.json')['sha256'];return
    cfg=CONFIGS[v['id']];setseed(seed);model=build(v['architecture'],cfg).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=EXP['lr'],weight_decay=EXP['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=EXP['step_size'],gamma=EXP['gamma'])
    history=[];orders=[];best_auc=-1.;best_mse=float('inf');first_epoch=1;elapsed=0.
    if (dest/'last.pt').exists():
        ck=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        assert ck['source_hashes']==source_hashes and ck['variant']==v and ck['seed']==seed
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];orders=ck['orders'];best_auc=ck['best_auc'];best_mse=ck['best_mse']
        first_epoch=ck['epoch']+1;elapsed=ck['elapsed'];initial=ck['initial']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
    else:
        dest.mkdir(parents=True,exist_ok=False)
        initial={n:sha_tensor(p) for n,p in model.named_parameters()}
        reference=yaml.safe_load((ROOT/'reference_initialization.yaml').read_text())
        assert initial==reference[f"{v['architecture']}_L{v['depth']}_seed{seed}"]
        save(dest/'initialization.json',initial)
    start_run=time.perf_counter()
    for epoch in range(first_epoch,EXP['epochs']+1):
        start=time.perf_counter();model.train();total=0.;n=0
        order=epoch_order(seed,epoch,len(train[1]));orders.append(sha_tensor(order))
        for x,y,_ in batches(train,order):
            opt.zero_grad(set_to_none=True);out=model(x);loss=objective(out,y,model.masks)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();opt.step();total+=loss.item()*len(y);n+=len(y)
        vm,_=evaluate(model,val)
        row={'epoch':epoch,'train_detector_plane_mse':total/n,'val_auroc':vm['auroc'],
             'val_accuracy':vm['accuracy'],'val_balanced_accuracy':vm['balanced_accuracy'],
             'val_detector_plane_mse':vm['detector_plane_mse'],'lr':opt.param_groups[0]['lr'],
             'seconds':time.perf_counter()-start}
        history.append(row);csvwrite(dest/'history.csv',history);save(dest/f'val_epoch_{epoch:03d}.json',vm)
        if better(vm['auroc'],vm['detector_plane_mse'],best_auc,best_mse):
            best_auc,best_mse=vm['auroc'],vm['detector_plane_mse']
            save_torch(dest/'best.pt',{'model':model.state_dict(),'epoch':epoch,'seed':seed,'variant':v,
                       'validation':vm,'source_hashes':source_hashes,'data_sha256':EXP['data_sha256']})
            save(dest/'selection.json',{'epoch':epoch,'validation':vm,'sha256':sha(dest/'best.pt')})
        scheduler.step()
        save_torch(dest/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                   'epoch':epoch,'seed':seed,'variant':v,'source_hashes':source_hashes,'history':history,'orders':orders,
                   'best_auc':best_auc,'best_mse':best_mse,'initial':initial,'elapsed':elapsed+time.perf_counter()-start_run,
                   'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state_all()})
        status(state='training',stage='screening' if index<=12 else 'formal',training_slot=index,total_training_slots=72,formal_run_target=TOTAL_RUNS,variant=v['id'],seed=seed,epoch=epoch,
               total_epochs=EXP['epochs'],best_val_auroc=best_auc,last_epoch=row)
        print(json.dumps({'variant':v['id'],'seed':seed,**row}),flush=True)
    changed={n:initial[n]!=sha_tensor(p) for n,p in model.named_parameters()}
    assert all(changed.values()),('Untrained phase planes',changed)
    save(dest/'completed.json',{'time':now(),'seconds':elapsed+time.perf_counter()-start_run,'epochs':EXP['epochs'],
                               'updates':EXP['epochs']*((len(train[1])+EXP['batch_size']-1)//EXP['batch_size']),
                               'order_sha256':orders,'changed_phase_planes':changed,'parameters':parameters(v)})
    del model,opt,scheduler;torch.cuda.empty_cache()

