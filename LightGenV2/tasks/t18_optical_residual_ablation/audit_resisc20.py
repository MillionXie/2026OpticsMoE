"""Manifest-count-aware saved CPU pair review; never invokes model forward/test."""
import argparse,copy,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
import resisc20 as s
from audit_caltech_repair234 import predictions,exited

def verify(base,key,data,m):
    arm=base/key;f=s.read(arm/'fit_result.json');cfg=s.read(arm/'config.json');meta=s.read(arm/'metadata.json');hist=s.read(arm/'history.json');grads=s.read(arm/'gradients.json')
    assert exited(meta['pid']) and f['config']==cfg and not f['test_used_for_selection'] and f['test_evaluations']==0
    s.engine.verify_sources(meta['sources'],meta['git_commit']);assert f['sources']==meta['sources']
    best=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert s.sha(arm/'best_checkpoint.pt')==f['checkpoint_sha256'] and s.sha(arm/'history.json')==f['history_sha256']
    assert best['config']==last['config']==cfg and best['sources']==last['sources']==f['sources']
    assert best['data_sha256']==last['data_sha256']==m['cache_sha256']
    assert [h['epoch'] for h in hist]==list(range(1,len(hist)+1)) and len(hist)==last['epoch']==f['epochs_completed']<=30
    chosen=min(hist,key=lambda h:h['val']['balanced_nll']);assert chosen['epoch']==best['epoch']==f['selected_epoch']
    assert chosen['val']==best['validation']==f['metrics']['val'] and best['selection_split']=='val' and best['selected_state_kind']=='ema'
    s.engine.seed(cfg['seed']);model=s.r.CorrectedMoE(4,f['rho']);assert s.digest(model.state_dict())==f['initial_parameter_sha256']==last['initial_parameter_sha256']
    shapes={n:tuple(v.shape) for n,v in model.named_parameters()};assert sum(v.numel() for v in model.parameters())==908552
    for kind in ('model','ema'):
        assert set(last[kind])==set(shapes) and all(tuple(v.shape)==shapes[n] and torch.isfinite(v).all() for n,v in last[kind].items())
    model.load_state_dict(best['model'],strict=True);assert all(torch.isfinite(v).all() for v in best['model'].values())
    assert [g['epoch'] for g in grads]==list(range(1,len(hist)+1)) and all(set(g['norms'])==set(shapes) and all(math.isfinite(v) and v>0 for v in g['norms'].values()) for g in grads)
    n=sum(m['supports']['train']);steps=math.ceil(n/16);assert last['scheduler']['last_epoch']==len(hist)
    assert len(last['optimizer']['state'])==11 and all(int(v['step'])==steps*len(hist) for v in last['optimizer']['state'].values())
    orders=[];transforms=[]
    for epoch,h in enumerate(hist,1):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(n).astype('<i8')
        angle=np.deg2rad(rng.uniform(-20,20,n));scale=rng.uniform(.9,1.1,n);theta=np.zeros((n,2,3),np.float32)
        theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0];theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale;theta[:,:,2]=rng.uniform(-5,5,(n,2))*2/150
        mix=hashlib.sha256()
        for b,start in enumerate(range(0,n,16)):
            count=min(16,n-start);draw=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,b,20261013,4]));lam=float(draw.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);permutation=draw.permutation(count)
            mix.update(hashlib.sha256(np.r_[lam,permutation.astype(float)].astype('<f8').tobytes()).hexdigest().encode())
        assert mix.hexdigest()==h['mixup_sha256'];orders.append(hashlib.sha256(order.tobytes()).hexdigest());transforms.append(hashlib.sha256((hashlib.sha256(theta.tobytes()).hexdigest()+mix.hexdigest()).encode()).hexdigest())
    assert orders==last['orders']==f['orders'] and transforms==last['transforms']==f['transforms']
    assert s.read(arm/'smoke.json')['passed'] and not (arm/'failure.json').exists()
    for role in ('train','val'):
        with np.load(data,allow_pickle=False) as z:labels=z[role+'_labels'].copy()
        predictions(arm/(role+'_predictions.csv'),f['metrics'][role],m['split_ids'][role],labels)
    assert s.eligibility(f,hist)==s.read(arm/'validation_gate.json')
    return f

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and not (a.base/'cpu_report_failure.json').exists()
    s.bind();m=s.manifest(a.data);meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json')
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==2
    assert sorted(j['rho'] for j in state['completed'])==[0.,.3];s.engine.verify_sources(meta['sources'],meta['git_commit']);fits={}
    for j in state['completed']:
        assert exited(j['pid']) and j['exit_code']==0;f=verify(a.base,j['case'],a.data,m)
        assert f['sources']==meta['sources'] and f['rho']==j['rho'];fits[j['case']]=f
    assert len({f['initial_parameter_sha256'] for f in fits.values()})==1
    normalized=[]
    for f in fits.values():
        cfg=copy.deepcopy(f['config'])
        for k in ('residual_rho','contract'):cfg.pop(k)
        normalized.append(cfg)
    assert normalized[0]==normalized[1]
    common=min(f['epochs_completed'] for f in fits.values())
    assert len({tuple(f['orders'][:common]) for f in fits.values()})==len({tuple(f['transforms'][:common]) for f in fits.values()})==1
    gates={k:s.read(a.base/k/'validation_gate.json') for k in fits};a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'audit.json',dict(passed=True,fits=fits,gates=gates,all_pair_eligible=all(g['passed'] for g in gates.values()),
           train_samples=sum(m['supports']['train']),optimizer_steps_per_epoch=math.ceil(sum(m['supports']['train'])/16),tail_batch=sum(m['supports']['train'])%16 or 16,
           model_forward_calls=0,test_arrays_read=0,test_evaluations=0,formal_authorized_by_this_receipt=False,
           scope='Full-resolution bounded feasibility pair; shared init/data/order/augmentation; selected trained native and independent2/3 still required'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','out','data'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        marker=a.base/'cpu_report_failure.json'
        if not marker.exists():s.save(marker,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
