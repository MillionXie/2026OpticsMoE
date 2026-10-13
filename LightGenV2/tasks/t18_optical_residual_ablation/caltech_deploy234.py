"""Frozen-data six-arm deployment experiment. Fit never opens test arrays."""
import argparse, copy, csv, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from caltech_deploy234_model import DeployMoE,input_amplitude,deployment_contract
import gpu_budget

ALLOWED={0:'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',1:'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',2:'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',4:'GPU-1b963983-7909-af6e-0528-f0f0661ab549'}
PROFILE=HERE/'caltech_deploy234_profile.json'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(path,value,exclusive=False):
    path=Path(path)
    if exclusive:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    else:
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def save_pt(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp)
    with temp.open('rb') as f:os.fsync(f.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def csvwrite(path,rows):
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)
def digest(value):
    h=hashlib.sha256()
    if isinstance(value,dict):
        for name,p in sorted(value.items()):h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    else:h.update(np.asarray(value).tobytes())
    return h.hexdigest()
def seed(value):torch.manual_seed(value);np.random.seed(value)
def sources():
    names=[HERE/n for n in ('caltech_deploy234_model.py','caltech_deploy234.py','caltech_deploy234_queue.py','caltech_deploy234_profile.json','gpu_budget.py')]
    names.extend([ROOT/'LightGenV2/tasks/t07_abo_image_retrieval/standalone/optics.py',ROOT/'LightGenV2/tasks/t07_abo_image_retrieval/standalone/__init__.py'])
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in names}
def verify_sources(src,commit):
    for name,h in src.items():
        assert sha(ROOT/name)==h and hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)).hexdigest()==h,(name,'source mismatch')
def manifest(data):
    cfg=read(PROFILE);m=read(data.parent/'data_manifest.json')
    assert sha(data)==cfg['cache_sha256']==m['cache_sha256']
    assert sha(data.parent/'data_manifest.json')==cfg['manifest_sha256']
    assert len(m['classes'])==20 and m['total_records']==3719
    return m

def load_fit(data,split,m):
    assert split in ('train','val'),'No test materialization during fit/probe'
    with np.load(data,allow_pickle=False) as z:
        x,y,ids=[z[split+'_'+k].copy() for k in ('images','labels','ids')]
    assert x.dtype==np.uint8 and x.shape==(len(y),150,150,3)
    assert ids.tolist()==m['split_ids'][split] and np.bincount(y,minlength=20).tolist()==m['supports'][split]
    return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids

def metrics(y,p):
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    pred=p.argmax(1);support=np.bincount(y,minlength=20);assert (support>0).all()
    cm=np.zeros((20,20),np.int64);np.add.at(cm,(y,pred),1)
    recall=cm.diagonal()/support;nll=-np.log(p[np.arange(len(y)),y].clip(1e-12))
    return dict(n=len(y),accuracy=float((pred==y).mean()),balanced_accuracy=float(recall.mean()),recall=recall.tolist(),support=support.tolist(),confusion_matrix=cm.tolist(),nll=float(nll.mean()),balanced_nll=float(np.mean([nll[y==k].mean() for k in range(20)])))

@torch.no_grad()
def evaluate(model,data,micro):
    model.eval();p=[];c=[]
    device=model.raw_router.device
    for idx in torch.arange(len(data[1])).split(micro):
        prob,capture=model(input_amplitude(data[0][idx].to(device)));p.append(prob.cpu().numpy());c.append(capture.cpu().numpy())
    prob=np.concatenate(p).astype(np.float64);result=metrics(data[1].numpy(),prob);result['detector_capture']=float(np.concatenate(c).mean())
    rows=[dict(sample_id=str(sid),label_true=int(y),label_pred=int(v.argmax()),**{f'score{k}':float(v[k]) for k in range(20)}) for sid,y,v in zip(data[2],data[1],prob)]
    return result,rows

def epoch_plan(epoch,n,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]))
    order=rng.permutation(n).astype('<i8')
    angle=np.deg2rad(rng.uniform(-20,20,n));scale=rng.uniform(.9,1.1,n)
    theta=np.zeros((n,2,3),np.float32)
    theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0]
    theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale
    theta[:,:,2]=rng.uniform(-5,5,(n,2))*2/150
    return order,theta

def mixed_batch(data,idx,theta,epoch,batch,cfg,device):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]))
    lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam)
    permutation=rng.permutation(len(idx))
    x=input_amplitude(data[0][idx].to(device),torch.as_tensor(theta[idx],device=device))
    y=F_onehot(data[1][idx].to(device),cfg['label_smoothing'])
    perm=torch.as_tensor(permutation,device=device)
    return lam*x+(1-lam)*x[perm],lam*y+(1-lam)*y[perm],digest(np.r_[lam,permutation.astype(float)].astype('<f8'))
def F_onehot(y,smoothing):return torch.nn.functional.one_hot(y,20).float()*(1-smoothing)+smoothing/20

def objective(model,prob,capture,target,weights,cfg):
    return -(target*weights[None]*prob.clamp_min(1e-12).log()).sum(1).mean()-cfg['capture_weight']*capture.clamp_min(1e-12).log().mean()+cfg['phase_smooth_weight']*model.smoothness()

def fresh_gate(depth,rho,data,cfg,device,micro):
    seed(cfg['seed']);model=DeployMoE(depth,rho).to(device);initial=digest(model.state_dict())
    counts=torch.bincount(data[1],minlength=20).float().to(device);weights=len(data[1])/(20*counts)
    idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(8)])
    _,theta=epoch_plan(1,len(data[1]),cfg);x,y,_=mixed_batch(data,idx,theta,1,0,cfg,device)
    def compute(batch):
        model.zero_grad(set_to_none=True);p=[]
        for start in range(0,8,batch):
            q,c=model(x[start:start+batch]);(objective(model,q,c,y[start:start+batch],weights,cfg)*len(q)/8).backward();p.append(q.detach())
        return torch.cat(p),{n:v.grad.detach().clone() for n,v in model.named_parameters()}
    p,g=compute(1);q,h=compute(micro)
    rel=max(float((h[n]-v).norm()/v.norm().clamp_min(1e-12)) for n,v in g.items())
    cosine=min(float(torch.nn.functional.cosine_similarity(h[n].flatten(),v.flatten(),dim=0)) for n,v in g.items())
    error=float((p-q).abs().max());finite=all(torch.isfinite(v).all() and v.norm()>0 for v in h.values())
    assert error<=1e-6 and rel<=.02 and cosine>=.999 and finite,(error,rel,cosine,finite)
    result=dict(passed=True,depth=depth,rho=rho,micro=micro,probability_max_error=error,gradient_max_relative_l2=rel,gradient_min_cosine=cosine,all_phase_gradients_finite_nonzero=bool(finite),initial_parameter_sha256=initial,parameters=sum(v.numel() for v in model.parameters()),test_evaluations=0)
    del model;torch.cuda.empty_cache() if device=='cuda' else None
    return result

def fit(a):
    assert a.gpu in ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    gpu_budget.ALLOWED.update(ALLOWED.values());policy=gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=3)
    torch.set_num_threads(4);cfg=read(PROFILE);m=manifest(a.data)
    cfg.update(depth=a.depth,residual_rho=a.rho,classes=m['classes'],contract=deployment_contract(a.depth,a.rho))
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'config.json',cfg);save(a.out/'metadata.json',dict(pid=os.getpid(),gpu_uuid=a.gpu,gpu_policy=policy,git_commit=commit,sources=src,command=sys.argv,test_evaluations=0))
    data,val=[load_fit(a.data,s,m) for s in ('train','val')]
    gate=fresh_gate(a.depth,a.rho,data,cfg,'cuda',cfg['microbatch']);save(a.out/'smoke.json',gate)
    seed(cfg['seed']);model=DeployMoE(a.depth,a.rho).cuda();initial=digest(model.state_dict());assert initial==gate['initial_parameter_sha256']
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=6,min_lr=2e-5)
    weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
    history=[];gradients=[];orders=[];transforms=[];best=float('inf');significant=float('inf');stale=0;started=time.monotonic();reason='maximum_epochs'
    for epoch in range(1,cfg['maximum_epochs']+1):
        if time.monotonic()-started>=cfg['maximum_seconds']:reason='time_budget';break
        begin=time.monotonic();model.train();order,theta=epoch_plan(epoch,len(data[1]),cfg);orders.append(digest(order));mixhash=hashlib.sha256();total=0.
        for batch,start in enumerate(range(0,len(order),cfg['batch_size'])):
            idx=order[start:start+cfg['batch_size']];x,y,mix=mixed_batch(data,idx,theta,epoch,batch,cfg,'cuda');mixhash.update(mix.encode());opt.zero_grad(set_to_none=True)
            for k in range(0,len(idx),cfg['microbatch']):
                p,c=model(x[k:k+cfg['microbatch']]);loss=objective(model,p,c,y[k:k+cfg['microbatch']],weights,cfg)
                assert torch.isfinite(loss);(loss*len(p)/len(idx)).backward();total+=float(loss.detach())*len(p)
            if batch==0:
                norms={n:float(v.grad.norm()) for n,v in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in norms.values());gradients.append(dict(epoch=epoch,norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-cfg['ema_decay'])
        vm,rows=evaluate(ema,val,cfg['microbatch']);entry=dict(epoch=epoch,train_loss=total/len(data[1]),val=vm,mixup_sha256=mixhash.hexdigest(),lr=opt.param_groups[0]['lr'])
        transforms.append(hashlib.sha256((digest(theta)+mixhash.hexdigest()).encode()).hexdigest())
        if epoch==1 or epoch%5==0:entry['train']=evaluate(ema,data,cfg['microbatch'])[0]
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];save_pt(a.out/'best_checkpoint.pt',dict(model=ema.state_dict(),epoch=epoch,config=cfg,sources=src,validation=vm,data_sha256=m['cache_sha256'],selection_split='val',selected_state_kind='ema'));csvwrite(a.out/'val_predictions.csv',rows)
        if vm['balanced_nll']<significant-cfg['early_min_delta']:significant=vm['balanced_nll'];stale=0
        else:stale+=1
        scheduler.step(vm['balanced_nll']);entry.update(seconds=time.monotonic()-started,epoch_seconds=time.monotonic()-begin);history.append(entry)
        save(a.out/'history.json',history);save(a.out/'gradients.json',gradients)
        save_pt(a.out/'last_checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,config=cfg,sources=src,orders=orders,transforms=transforms,initial_parameter_sha256=initial,data_sha256=m['cache_sha256']))
        status=dict(state='training',depth=a.depth,rho=a.rho,epoch=epoch,val_accuracy=vm['accuracy'],val_macro_recall=vm['balanced_accuracy'],val_balanced_nll=vm['balanced_nll'],epoch_seconds=entry['epoch_seconds'],test_evaluations=0)
        if 'train' in entry:status['train_val_gap_pp']=100*(entry['train']['accuracy']-vm['accuracy'])
        save(a.out/'status.json',status);print(json.dumps(status),flush=True)
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']:reason='validation_plateau';break
    assert history
    ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(ck['model']);tm,rows=evaluate(ema,data,cfg['microbatch']);csvwrite(a.out/'train_predictions.csv',rows)
    result=dict(depth=a.depth,rho=a.rho,epochs_completed=len(history),selected_epoch=ck['epoch'],stop_reason=reason,metrics=dict(train=tm,val=ck['validation']),history_sha256=sha(a.out/'history.json'),checkpoint_sha256=sha(a.out/'best_checkpoint.pt'),config=cfg,sources=src,initial_parameter_sha256=initial,orders=orders,transforms=transforms,seconds=time.monotonic()-started,test_evaluations=0,test_used_for_selection=False,scope='Existing validation-screened classes and previously observed old test; exploratory development. Accuracy75 target not guaranteed. Early-stop budgets can differ, not a pure depth/residual causal estimate.')
    save(a.out/'fit_result.json',result);save(a.out/'status.json',dict(state='fit_complete',epochs=len(history),selected_epoch=ck['epoch'],test_evaluations=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--depth',type=int,choices=[2,3,4],required=True);parser.add_argument('--rho',type=float,choices=[0.,.3],required=True);parser.add_argument('--gpu',required=True)
    a=parser.parse_args()
    try:fit(a)
    except BaseException as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),test_evaluations=0))
        raise