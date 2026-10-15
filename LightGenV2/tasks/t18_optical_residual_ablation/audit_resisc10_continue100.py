"""Independent saved resume mathematics; no optical forward or test images."""
import argparse,csv,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
import resisc10_continue100 as study
from audit_caltech_repair234 import exited
from audit_resisc10 import predictions
s=study.s;base=study.base
def independent_hashes(hist):
    orders=[];transforms=[];n=5600
    for epoch,row in enumerate(hist,1):
        alpha=.2 if epoch<=100 else .4
        rng=np.random.default_rng(np.random.SeedSequence([17,epoch,20261013]));order=rng.permutation(n).astype('<i8')
        angle=np.deg2rad(rng.uniform(-20,20,n));scale=rng.uniform(.9,1.1,n);theta=np.zeros((n,2,3),np.float32)
        theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0];theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale;theta[:,:,2]=rng.uniform(-5,5,(n,2))*2/150
        mixed=hashlib.sha256()
        for batch,start in enumerate(range(0,n,16)):
            draw=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(draw.beta(alpha,alpha));lam=max(lam,1-lam);perm=draw.permutation(min(16,n-start))
            mixed.update(hashlib.sha256(np.r_[lam,perm.astype(float)].astype('<f8').tobytes()).hexdigest().encode())
        assert mixed.hexdigest()==row['mixup_sha256'];oh=hashlib.sha256(order.tobytes()).hexdigest();orders.append(oh);transforms.append(hashlib.sha256((hashlib.sha256(theta.tobytes()).hexdigest()+mixed.hexdigest()).encode()).hexdigest())
    return orders,transforms
def lr_replay(hist,last):
    rate=.002;best=float('inf');bad=0
    for epoch,row in enumerate(hist,1):
        if epoch==101:rate=max(rate*.5,2e-5)
        assert math.isclose(row['lr'],rate,rel_tol=0,abs_tol=1e-15)
        value=row['val']['balanced_nll']
        if value<best*(1-1e-4):best=value;bad=0
        else:bad+=1
        if bad>6:
            lowered=max(rate*.5,2e-5)
            if rate-lowered>1e-8:rate=lowered
            bad=0
    sch=last['scheduler'];assert sch['last_epoch']==len(hist) and sch['best']==best and sch['num_bad_epochs']==bad
    assert sch['factor']==.5 and sch['patience']==6 and sch['min_lrs']==[2e-5] and sch['threshold']==1e-4 and sch['threshold_mode']=='rel'
    assert sch['_last_lr']==[rate] and math.isclose(last['optimizer']['param_groups'][0]['lr'],rate,rel_tol=0,abs_tol=1e-15)
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();stage=a.base.name;assert stage in ('complete100','regularize20')
    cfg,arm,parent_ck=study.configuration(stage);f=s.read(a.base/'fit_result.json');meta=s.read(a.base/'metadata.json');exit_rec=s.read(a.base/'producer_exit.json');hist=s.read(a.base/'history.json');grads=s.read(a.base/'gradients.json')
    assert exited(meta['pid']) and exit_rec['returncode']==0 and exit_rec['actual_exit'] and exit_rec['pid']==meta['pid']
    s.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['sources']==f['sources']==study.sources()
    assert cfg==f['config']==s.read(a.base/'config.json') and f['rho']==.3 and f['depth']==4 and f['test_evaluations']==0
    prefix=92 if stage=='complete100' else 100;assert hist[:prefix]==s.read(arm/'history.json') and grads[:prefix]==s.read(arm/'gradients.json')
    assert [x['epoch'] for x in hist]==[x['epoch'] for x in grads]==list(range(1,cfg['maximum_epochs']+1))
    assert f['epochs_completed']==len(hist)==cfg['maximum_epochs'] and f['stop_reason']=='maximum_epochs'
    best=torch.load(a.base/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(a.base/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert best['config']==last['config']==cfg and best['sources']==last['sources']==f['sources']
    assert s.sha(a.base/'history.json')==f['history_sha256'] and s.sha(a.base/'best_checkpoint.pt')==f['checkpoint_sha256']
    chosen=min(hist,key=lambda x:x['val']['balanced_nll']);assert chosen['epoch']==f['selected_epoch']==best['epoch'] and chosen['val']==best['validation']==f['metrics']['val']
    assert best['selection_split']=='val' and best['selected_state_kind']=='ema'
    s.engine.seed(17);model=base.SubsetMoE(4,.3);shapes={k:tuple(v.shape) for k,v in model.named_parameters()}
    assert len(shapes)==11 and sum(v.numel() for v in model.parameters())==908552 and s.digest(model.state_dict())==f['initial_parameter_sha256']==last['initial_parameter_sha256']
    for state in (best['model'],last['model'],last['ema']):assert set(state)==set(shapes) and all(tuple(v.shape)==shapes[k] and torch.isfinite(v).all() for k,v in state.items())
    assert all(set(g['norms'])==set(shapes) and all(math.isfinite(v) and v>0 for v in g['norms'].values()) for g in grads)
    assert last['epoch']==len(hist) and len(last['optimizer']['state'])==11 and all(int(v['step'])==350*len(hist) for v in last['optimizer']['state'].values())
    orders,transforms=independent_hashes(hist);assert orders==last['orders']==f['orders'] and transforms==last['transforms']==f['transforms'];lr_replay(hist,last)
    offset=hist[prefix-1]['seconds'];prior=offset
    for row in hist[prefix:]:assert prior-offset<cfg['maximum_seconds'] and row['seconds']>=prior;prior=row['seconds']
    receipt=s.read(a.base/'resume_receipt.json');assert receipt['parent_epoch']==prefix and receipt['first_new_epoch']==prefix+1 and receipt['last_checkpoint_sha256']==s.sha(arm/'last_checkpoint.pt') and receipt['prefix_history_sha256']==s.sha(arm/'history.json')
    assert not receipt['optimizer_reset'] and not receipt['ema_reset'] and not receipt['scheduler_reset']
    assert receipt['restored_state_sha256']=={k:s.digest(parent_ck[k]) for k in ('model','ema')}
    smoke=s.read(a.base/'smoke.json');assert smoke['passed'] and smoke['initial_parameter_sha256']==s.digest(parent_ck['model'])
    assert len(smoke['phase_gradients'])==11 and all(math.isfinite(v) and v>0 for v in smoke['phase_gradients'].values())
    m=base.manifest(Path(cfg['data']));assert best['data_sha256']==last['data_sha256']==m['cache_sha256']
    for role in ('train','val'):
        with np.load(cfg['data'],allow_pickle=False) as z:labels=base.subset_labels(z[role+'_labels'].copy())
        predictions(a.base/(role+'_predictions.csv'),f['metrics'][role],m['split_ids'][role],labels)
    assert base.eligibility(f,hist)==s.read(a.base/'validation_gate.json') and not (a.base/'failure.json').exists()
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'audit.json',dict(passed=True,stage=stage,sources=f['sources'],fit=f,parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),prefix_epochs=prefix,independent_order_affine_mixup_lr_checks=True,all_saved_ID_label_CM_NLL_verified=True,actual_adam_steps=350*len(hist),cpu_models_for_shape_seed=1,model_forward_calls=0,test_arrays_read=0,test_evaluations=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        mark=a.base/'cpu_report_failure.json'
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
