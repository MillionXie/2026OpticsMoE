"""Independent saved CPU review of the three bounded repair candidates; no inference/test."""
import argparse,csv,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
import caltech_repair234 as r
from caltech_repair234_model import CorrectedMoE
t=r.engine
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
    t.seed(cfg['seed']);model=CorrectedMoE(cfg['depth'],cfg['residual_rho']);assert t.digest(model.state_dict())==fit['initial_parameter_sha256']==last['initial_parameter_sha256']
    names=set(dict(model.named_parameters()));model.load_state_dict(best['model'],strict=True)
    assert all(torch.isfinite(v).all() for v in best['model'].values())
    assert [g['epoch'] for g in grads]==list(range(1,len(hist)+1))
    assert all(set(g['norms'])==names and all(math.isfinite(v) and v>0 for v in g['norms'].values()) for g in grads)
    assert last['scheduler']['last_epoch']==len(hist)
    assert all(int(s['step'])==78*len(hist) for s in last['optimizer']['state'].values())
    orderhash=[];transformhash=[]
    # Reproduce schedule independently from the training batch implementation.
    for epoch,h in enumerate(hist,1):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(1246).astype('<i8')
        angle=np.deg2rad(rng.uniform(-20,20,1246));scale=rng.uniform(.9,1.1,1246);theta=np.zeros((1246,2,3),np.float32)
        theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0];theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale;theta[:,:,2]=rng.uniform(-5,5,(1246,2))*2/150
        mix=hashlib.sha256()
        for b,start in enumerate(range(0,1246,16)):
            count=min(16,1246-start);r=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,b,20261013,4]));lam=float(r.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);permutation=r.permutation(count)
            identity=hashlib.sha256(np.r_[lam,permutation.astype(float)].astype('<f8').tobytes()).hexdigest();mix.update(identity.encode())
        assert mix.hexdigest()==h['mixup_sha256']
        orderhash.append(hashlib.sha256(order.tobytes()).hexdigest());transformhash.append(hashlib.sha256((hashlib.sha256(theta.tobytes()).hexdigest()+mix.hexdigest()).encode()).hexdigest())
    assert orderhash==fit['orders']==last['orders'] and transformhash==fit['transforms']==last['transforms']
    smoke=t.read(arm/'smoke.json');assert smoke['passed'] and smoke['micro']==1
    for split in ('train','val'):
        with np.load(data,allow_pickle=False) as z:labels=z[split+'_labels'].copy()
        predictions(arm/(split+'_predictions.csv'),fit['metrics'][split],m['split_ids'][split],labels)
    return fit

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists()
    meta=r.read(a.base/'metadata.json');state=r.read(a.base/'status.json');assert exited(meta['pid']) and not state['failures'] and len(state['completed'])==3 and state['active']==state['pending']==0
    m=r.manifest(a.data);r.engine.verify_sources(meta['sources'],meta['git_commit']);fits={}
    for job in state['completed']:
        key=job['case'];assert exited(job['pid']) and job['exit_code']==0;r.bind(job['alpha']);fits[key]=verify_arm(a.base,key,a.data,m)
        f=fits[key];assert f['config']['mixup_alpha']==job['alpha'] and f['depth']==4 and f['rho']==.3
        assert f['sources']==meta['sources'];assert not (a.base/key/'failure.json').exists()
    assert sorted(f['config']['mixup_alpha'] for f in fits.values())==[.2,.4,.6]
    assert len({f['initial_parameter_sha256'] for f in fits.values()})==1
    # Shared order and affine draws; MixUp deliberately differs by candidate.
    minimum=min(f['epochs_completed'] for f in fits.values());assert len({tuple(f['orders'][:minimum]) for f in fits.values()})==1
    gates={};g=r.read(r.PLAN)['formal_gate']
    for key,f in fits.items():
        hist=r.read(a.base/key/'history.json');tm,vm=f['metrics']['train'],f['metrics']['val'];tail=hist[-min(20,len(hist)):]
        checks=dict(val_accuracy=vm['accuracy']>=g['L4_val_accuracy_at_selected_ema_min'],val_macro=vm['balanced_accuracy']>=g['L4_val_macro_recall_min'],no_zero_train_val=min(tm['recall'])>0 and min(vm['recall'])>0,val_each_recall=min(vm['recall'])>=g['selected_val_each_recall_min'],no_persistent_zero=all(sum(h['val']['recall'][k]==0 for h in tail)<=len(tail)/2 for k in range(20)),gap=100*(tm['accuracy']-vm['accuracy'])<=15,routing_all_slots=all(v==1 for split in (tm,vm) for v in split['routing']['nonzero_fraction']),routing_input_dependent=all(max(split['routing']['weight_std'])>1e-4 for split in (tm,vm)))
        gate=r.read(a.base/key/'validation_gate.json');assert gate['checks']==checks and gate['passed']==all(checks.values());gates[key]=gate
        for split in (tm,vm):
            route=split['routing'];assert np.asarray(route['per_class_weight_mean']).shape==(20,4)
            assert all(np.isfinite(np.asarray(route[name])).all() for name in ('weight_mean','weight_std','weight_min','weight_max','per_class_weight_mean'))
    eligible=[key for key in sorted(fits) if gates[key]['passed']];chosen=min(eligible,key=lambda k:fits[k]['metrics']['val']['balanced_nll']) if eligible else None
    a.out.mkdir(parents=True,exist_ok=False);r.save(a.out/'audit.json',dict(passed=True,fits=fits,gates=gates,eligible=eligible,chosen_profile_case=chosen,formal_authorized_by_this_receipt=False,next_step='Selected weight native parity and independent 2/3 validation before formal' if chosen else 'No candidate eligible; report bounded failure, do not start formal',model_forward_calls=0,test_arrays_read=0,test_evaluations=0,tail_batch=14,optimizer_steps_per_epoch=78,routing_scope='Recorded during existing complete evaluations; route statistics checked for consistency, not independently re-inferred'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','out','data'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a)
