"""Saved-result CPU audit, exclusive all-six locks, single selected tests and export."""
import argparse,csv,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech_deploy234 as t
from caltech_deploy234_model import DeployMoE,deployment_contract
KEYS=[f'L{d}_'+r for d in (2,3,4) for r in ('rho0','rho03')]

def exited(pid):
    p=Path('/proc')/str(pid)
    return not p.exists() or not p.joinpath('cmdline').read_bytes()

def predictions(path,metric,ids,labels):
    rows=list(csv.DictReader(path.open()));assert [r['sample_id'] for r in rows]==list(ids)
    y=np.array([int(r['label_true']) for r in rows]);assert np.array_equal(y,labels)
    p=np.array([[float(r[f'score{k}']) for k in range(20)] for r in rows]);assert np.array_equal(p.argmax(1),[int(r['label_pred']) for r in rows])
    calculated=t.metrics(y,p)
    for k,v in calculated.items():
        if isinstance(v,float):assert math.isclose(v,metric[k],rel_tol=1e-9,abs_tol=1e-9),(k,v,metric[k])
        else:assert v==metric[k],k
    return calculated

def verify_arm(base,key,data,m):
    arm=base/key;fit=t.read(arm/'fit_result.json');cfg=t.read(arm/'config.json');meta=t.read(arm/'metadata.json');hist=t.read(arm/'history.json');grads=t.read(arm/'gradients.json')
    assert cfg==fit['config'] and fit['test_evaluations']==0 and not fit['test_used_for_selection']
    assert exited(meta['pid']);t.verify_sources(meta['sources'],meta['git_commit']);assert meta['sources']==fit['sources']
    best=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert t.sha(arm/'best_checkpoint.pt')==fit['checkpoint_sha256'] and t.sha(arm/'history.json')==fit['history_sha256']
    assert best['config']==last['config']==cfg and best['sources']==last['sources']==fit['sources']
    assert best['data_sha256']==last['data_sha256']==m['cache_sha256']
    assert len(hist)==last['epoch']==fit['epochs_completed'] and 1<=len(hist)<=cfg['maximum_epochs']
    assert [h['epoch'] for h in hist]==list(range(1,len(hist)+1))
    chosen=min(hist,key=lambda h:h['val']['balanced_nll']);assert chosen['epoch']==fit['selected_epoch']==best['epoch'] and chosen['val']==best['validation']==fit['metrics']['val']
    assert best['selection_split']=='val' and best['selected_state_kind']=='ema'
    t.seed(cfg['seed']);model=DeployMoE(cfg['depth'],cfg['residual_rho']);assert t.digest(model.state_dict())==fit['initial_parameter_sha256']==last['initial_parameter_sha256']
    names=set(dict(model.named_parameters()));model.load_state_dict(best['model'],strict=True)
    assert all(torch.isfinite(v).all() for v in best['model'].values())
    assert [g['epoch'] for g in grads]==list(range(1,len(hist)+1))
    assert all(set(g['norms'])==names and all(math.isfinite(v) and v>0 for v in g['norms'].values()) for g in grads)
    assert last['scheduler']['last_epoch']==len(hist)
    assert all(int(s['step'])==86*len(hist) for s in last['optimizer']['state'].values())
    orderhash=[];transformhash=[]
    # Reproduce schedule independently from the training batch implementation.
    for epoch,h in enumerate(hist,1):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(1372).astype('<i8')
        angle=np.deg2rad(rng.uniform(-20,20,1372));scale=rng.uniform(.9,1.1,1372);theta=np.zeros((1372,2,3),np.float32)
        theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0];theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale;theta[:,:,2]=rng.uniform(-5,5,(1372,2))*2/150
        mix=hashlib.sha256()
        for b,start in enumerate(range(0,1372,16)):
            count=min(16,1372-start);r=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,b,20261013,4]));lam=float(r.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);permutation=r.permutation(count)
            identity=hashlib.sha256(np.r_[lam,permutation.astype(float)].astype('<f8').tobytes()).hexdigest();mix.update(identity.encode())
        assert mix.hexdigest()==h['mixup_sha256']
        orderhash.append(hashlib.sha256(order.tobytes()).hexdigest());transformhash.append(hashlib.sha256((hashlib.sha256(theta.tobytes()).hexdigest()+mix.hexdigest()).encode()).hexdigest())
    assert orderhash==fit['orders']==last['orders'] and transformhash==fit['transforms']==last['transforms']
    smoke=t.read(arm/'smoke.json');assert smoke['passed'] and smoke['micro']==1
    for split in ('train','val'):
        with np.load(data,allow_pickle=False) as z:labels=z[split+'_labels'].copy()
        predictions(arm/(split+'_predictions.csv'),fit['metrics'][split],m['split_ids'][split],labels)
    return fit

def audit_all(a):
    m=t.manifest(a.data);meta=t.read(a.base/'metadata.json');assert exited(meta['pid'])
    state=t.read(a.base/'status.json');assert state['state']=='fit_complete_awaiting_cpu_review' and len(state['completed'])==6
    fits={k:verify_arm(a.base,k,a.data,m) for k in KEYS}
    for d in (2,3,4):
        left,right=[fits[f'L{d}_{r}'] for r in ('rho0','rho03')]
        assert left['initial_parameter_sha256']==right['initial_parameter_sha256'] and left['sources']==right['sources']
        cfgs=[dict(f['config']) for f in (left,right)]
        for cfg in cfgs:cfg.pop('residual_rho');cfg['contract']=dict(cfg['contract']);cfg['contract'].pop('rho')
        assert cfgs[0]==cfgs[1]
        common=min(left['epochs_completed'],right['epochs_completed'])
        assert left['orders'][:common]==right['orders'][:common] and left['transforms'][:common]==right['transforms'][:common]
    return fits,m

def lock(a):
    assert not a.out.exists(),'Never retry a partial or completed lock directory'
    fits,m=audit_all(a);a.out.mkdir(parents=True,exist_ok=False)
    t.save(a.out/'cpu_fit_audit.json',dict(passed=True,fits=fits,all_six=True,test_arrays_read=0,test_evaluations=0))
    for d in (2,3,4):t.save(a.out/f'pair_lock_L{d}.json',dict(depth=d,config_only_rho=True,keys=[f'L{d}_rho0',f'L{d}_rho03']),exclusive=True)
    locks={}
    for key in KEYS:
        (a.out/key).mkdir();fit=fits[key];value=dict(epoch=fit['selected_epoch'],checkpoint_sha256=fit['checkpoint_sha256'],data_sha256=m['cache_sha256'],config=fit['config'],sources=fit['sources'],selection_split='val',test_evaluations_before_lock=0)
        t.save(a.out/key/'selection_lock.json',value,exclusive=True);locks[key]=t.sha(a.out/key/'selection_lock.json')
    t.save(a.out/'global_selection_lock.json',dict(locks=locks,all_six_fit_audited=True,data_sha256=m['cache_sha256'],test_evaluations_before_lock=0),exclusive=True)

def validate_lock(a):
    lock=t.read(a.out/'global_selection_lock.json');assert set(lock['locks'])==set(KEYS) and lock['all_six_fit_audited']
    assert lock['data_sha256']==t.sha(a.data)
    for key,h in lock['locks'].items():assert t.sha(a.out/key/'selection_lock.json')==h
    return lock

def test_worker(a):
    assert a.gpu in t.ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    validate_lock(a);selection=t.read(a.out/a.key/'selection_lock.json');t.gpu_budget.ALLOWED.update(t.ALLOWED.values());t.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    fit=t.read(a.base/a.key/'fit_result.json');assert selection['checkpoint_sha256']==t.sha(a.base/a.key/'best_checkpoint.pt')==fit['checkpoint_sha256']
    t.save(a.out/a.key/'test_started.json',dict(pid=os.getpid(),gpu_uuid=a.gpu,checkpoint_sha256=selection['checkpoint_sha256'],data_sha256=selection['data_sha256'],time_unix=time.time()),exclusive=True)
    # Only AFTER durable file + directory fsync claim materialize test arrays.
    m=t.manifest(a.data)
    with np.load(a.data,allow_pickle=False) as z:x=z['test_images'].copy();y=z['test_labels'].copy();ids=z['test_ids'].copy()
    assert ids.tolist()==m['split_ids']['test'] and np.bincount(y,minlength=20).tolist()==m['supports']['test']
    data=(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids)
    ck=torch.load(a.base/a.key/'best_checkpoint.pt',map_location='cpu',weights_only=False);cfg=ck['config'];model=DeployMoE(cfg['depth'],cfg['residual_rho']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    metric,rows=t.evaluate(model,data,1);t.csvwrite(a.out/a.key/'test_predictions.csv',rows)
    t.save(a.out/a.key/'result.json',dict(test=metric,selected_epoch=ck['epoch'],checkpoint_sha256=selection['checkpoint_sha256'],test_evaluations=1,test_used_for_selection=False,test_accuracy_target_passed=(metric['accuracy']>=.75 if a.key=='L4_rho03' else None)))
    # Static deployment export on CPU; no extra sample/model inference.
    from export_caltech_deploy234 import export_selected
    export_selected(model.cpu(),cfg,a.out/a.key/'deployment',a.base/a.key/'best_checkpoint.pt',selection['checkpoint_sha256'],m['classes'],fit['sources'])

def suite(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';validate_lock(a)
    t.save(a.out/'suite_started.json',dict(pid=os.getpid(),time_unix=time.time(),keys=KEYS),exclusive=True)
    for key in KEYS:
        from caltech_deploy234_queue import gpu_available
        gpu=next((u for u in (t.ALLOWED[2],t.ALLOWED[1],t.ALLOWED[4],t.ALLOWED[0]) if gpu_available(u)),None)
        assert gpu
        cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','test_worker','--base',str(a.base),'--out',str(a.out),'--data',str(a.data),'--key',key,'--gpu',gpu]
        with (a.out/(key+'.test.log')).open('x') as log:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    t.save(a.out/'results.json',{key:t.read(a.out/key/'result.json') for key in KEYS})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','lock','suite','test_worker'],required=True)
    for name in ('base','out','data'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--key',choices=KEYS);p.add_argument('--gpu');a=p.parse_args()
    if a.phase=='audit':
        fits,m=audit_all(a);a.out.mkdir(parents=True,exist_ok=False);t.save(a.out/'cpu_fit_audit.json',dict(passed=True,fits=fits,all_six=True,test_arrays_read=0,test_evaluations=0))
    elif a.phase=='lock':lock(a)
    elif a.phase=='suite':suite(a)
    else:test_worker(a)