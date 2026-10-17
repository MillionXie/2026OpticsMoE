"""Independent NumPy saved-math, schedule and state audit; no model forwards."""
import argparse,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_original_equal60 as study
from caltech20_alt246_genreg_model import RegularizedMoE as HybridMoE

def metrics(labels,p):
    assert p.shape==(len(labels),20) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
    cm=np.zeros((20,20),np.int64)
    for y,pred in zip(labels,p.argmax(1)):cm[int(y),int(pred)]+=1
    support=cm.sum(1);assert min(support)>0;recall=np.diag(cm)/support;nll=-np.log(np.maximum(p[np.arange(len(labels)),labels],1e-12))
    return dict(n=len(labels),accuracy=float(np.trace(cm)/len(labels)),balanced_accuracy=float(recall.mean()),support=support.tolist(),recall=recall.tolist(),confusion_matrix=cm.tolist(),nll=float(nll.mean()),balanced_nll=float(np.mean([nll[labels==k].mean() for k in range(20)])))

def agree(actual,expected):
    for k in ('n','support','confusion_matrix'):assert actual[k]==expected[k],k
    for k in ('accuracy','balanced_accuracy','nll','balanced_nll','recall'):assert np.allclose(actual[k],expected[k],rtol=1e-7,atol=1e-7),(k,actual[k],expected[k])

def history_metric(v,support):
    cm=np.asarray(v['confusion_matrix']);assert cm.shape==(20,20) and (cm>=0).all() and cm.sum(1).tolist()==support
    assert abs(np.trace(cm)/cm.sum()-v['accuracy'])<1e-12 and np.allclose(np.diag(cm)/cm.sum(1),v['recall'],atol=1e-12)
    assert abs(np.mean(v['recall'])-v['balanced_accuracy'])<1e-12 and np.isfinite([v['nll'],v['balanced_nll']]).all()
    assert all(.7<x<=.95 for x in v['alphas'])

def pool_layernorm(a):
    n,h,w=a.shape;pooled=np.empty((n,16,16),np.float64)
    for i in range(16):
        for j in range(16):pooled[:,i,j]=a[:,math.floor(i*h/16):math.ceil((i+1)*h/16),math.floor(j*w/16):math.ceil((j+1)*w/16)].astype(np.float64).mean((1,2))
    f=pooled.reshape(n,256);return (f-f.mean(1,keepdims=True))/np.sqrt(f.var(1,keepdims=True)+1e-5)

def verify_predictions(path,state,role,m):
    with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
    assert v['ids'].tolist()==m['split_ids'][role] and np.bincount(v['labels'],minlength=20).tolist()==m['supports'][role]
    cfg=study.config()
    with np.load(cfg['data'],allow_pickle=False) as z:assert np.array_equal(v['labels'],z[role+'_labels'])
    W=state['linear.weight'].numpy().astype(np.float64);b=state['linear.bias'].numpy().astype(np.float64);logits=v['features'].astype(np.float64)@W.T+b
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);error=float(np.abs(p-v['probabilities']).max());assert error<=3e-6,error
    f=pool_layernorm(v['probe_amplitudes']);poolerr=float(np.abs(f-v['features'][:len(f)]).max());assert poolerr<=3e-4,poolerr
    return metrics(v['labels'],v['probabilities']),dict(probability_maxabs=error,pooling_maxabs=poolerr,prediction_sha256=study.s.sha(path))

def plans(epoch,cfg,n):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(n).astype('<i8')
    angle=np.deg2rad(rng.uniform(-20,20,n));scale=rng.uniform(.9,1.1,n);theta=np.zeros((n,2,3),np.float32)
    theta[:,0,0]=np.cos(angle)/scale;theta[:,1,1]=theta[:,0,0];theta[:,0,1]=-np.sin(angle)/scale;theta[:,1,0]=np.sin(angle)/scale;theta[:,:,2]=rng.uniform(-5,5,(n,2))*2/150
    mix=[]
    for batch,start in enumerate(range(0,n,16)):
        rg=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rg.permutation(min(16,n-start))
        mix.append(hashlib.sha256(np.r_[lam,perm.astype(float)].astype('<f8').tobytes()).hexdigest())
    return tuple(hashlib.sha256(v).hexdigest() for v in (order.tobytes(),theta.tobytes(),''.join(mix).encode()))

def mask_plans(epoch,cfg,n):
    hashes={k:[] for k in ('flip','mask')}
    for batch,off in enumerate(range(0,n,16)):
        count=min(16,n-off)
        flip=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261016,1])).random(count)<cfg['horizontal_flip_probability']
        keep=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261016,2])).random((count,256))>=cfg['feature_dropout'];mask=keep.astype(np.float32)/(1-cfg['feature_dropout'])
        for k,v in [('flip',flip),('mask',mask)]:hashes[k].append(hashlib.sha256(v.tobytes()).hexdigest())
    return tuple(hashlib.sha256(''.join(hashes[k]).encode()).hexdigest() for k in ('flip','mask'))

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and base.name==study.RUN and not out.exists()
    state=s.read(base/'status.json');assert not state['active'] and not state['pending'] and not state['failures']
    cfg=study.config();m=study.dataset.manifest();meta=s.read(base/'metadata.json');assert meta['sources']==study.sources() and meta['config']==cfg
    study.engine.verify_sources(meta['sources'],meta['git_commit']);checks={};paired={};torch.set_num_threads(4)
    if phase=='fit':
        for key,(depth,rho) in study.CASES.items():
            folder=base/key;exit=s.read(folder.with_name(key+'.fit_exit.json'));assert exit['returncode']==0 and exit['actual_exit'] and not Path('/proc',str(exit['pid'])).exists()
            metadata=s.read(folder/'metadata.json');assert metadata['sources']==meta['sources'] and metadata['config']==cfg and metadata['depth']==depth and metadata['rho']==rho
            h=s.read(folder/'history.json');f=s.read(folder/'fit_result.json');assert len(h)==cfg['maximum_epochs'] and f['epochs_completed']==cfg['maximum_epochs'] and f['steps']==cfg['maximum_epochs']*cfg['steps_per_epoch'] and f['stop_reason']=='maximum_epochs'
            assert f['history_sha256']==s.sha(folder/'history.json') and f['best_sha256']==s.sha(folder/'best.pt') and f['last_sha256']==s.sha(folder/'last.pt')
            best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            assert best['sources']==last['sources']==meta['sources'] and best['config']==last['config']==cfg
            study.engine.seed(17);model=HybridMoE(depth,rho);initial=s.digest(model.state_dict());assert initial==f['initial_digest']==best['initial_digest']==last['initial_digest']
            assert best['contract']==model.contract();model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True)
            assert f['fit_seconds']<=cfg['maximum_seconds']+max(r['epoch_seconds'] for r in h)+60
            assert last['epoch']==last['scheduler']['last_epoch']==cfg['maximum_epochs'] and last['steps']==cfg['maximum_epochs']*cfg['steps_per_epoch']
            optimizer=last['optimizer'];assert all(float(v['step'])==cfg['maximum_epochs']*cfg['steps_per_epoch'] for v in optimizer['state'].values()) and len(optimizer['state'])==len(list(model.parameters()))
            assert [q['weight_decay'] for q in optimizer['param_groups']]==[0,cfg['electronic_weight_decay']]
            assert set(last['previous_final_ema'])==set(last['model'])==set(last['ema'])
            for name,previous in last['previous_final_ema'].items():
                expected=previous.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][name].numpy().astype(np.float64)*(1-cfg['ema_decay'])
                assert np.allclose(expected,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6),('EMA',name)
            for i,row in enumerate(h,1):
                assert row['epoch']==i and row['adam_steps']==cfg['steps_per_epoch']*i
                assert tuple(row[k] for k in ('order_sha256','affine_sha256','mix_sha256'))==plans(i,cfg,cfg['training_samples'])
                assert (row['flip_sha256'],row['mask_sha256'])==mask_plans(i,cfg,cfg['training_samples'])
                lr=cfg['minimum_lr']+(cfg['lr']-cfg['minimum_lr'])*(1+math.cos(math.pi*(i-1)/cfg['maximum_epochs']))/2
                assert np.allclose(row['lr'],[lr,lr],atol=1e-12,rtol=0)
                for role in ('train','val'):history_metric(row[role],m['supports'][role])
            selected=min(h,key=lambda row:row['val']['balanced_nll']);assert best['epoch']==f['selected_epoch']==selected['epoch']
            assert s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest'] and s.digest(best['model'])==selected['ema_digest']
            for role in ('train','val'):
                actual,receipt=verify_predictions(folder/('selected_'+role+'.npz'),best['model'],role,m);agree(actual,f['metrics'][role]);agree(actual,best['metrics'][role]);agree(actual,selected[role])
                checks[key+'_'+role]=receipt
            assert all(.7<x<=.95 for x in model.alphas());gate=s.read(folder/'numeric_gate.json');assert gate['passed'] and gate['initial_parameter_sha256']==initial and len(gate['checks'])==6
            norms=gate['checks'][-1]['gradient_norms'];assert set(norms)==set(dict(model.named_parameters())) and all(np.isfinite(v) and v>0 for v in norms.values())
            paired.setdefault(depth,[]).append(dict(initial=initial,orders=[(r['order_sha256'],r['affine_sha256'],r['mix_sha256'],r['flip_sha256'],r['mask_sha256']) for r in h]))
            del model,best,last
        assert all(v[0]==v[1] for v in paired.values())
    else:
        receipt=s.read(base/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(base/'science/audit.json')
        for key in study.CASES:
            folder=base/'once_tests'/key;ex=s.read(folder.with_name(key+'.test_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid'])).exists()
            started=s.read(folder/'started.json');result=s.read(folder/'result.json');best=torch.load(base/key/'best.pt',map_location='cpu',weights_only=False)
            assert started['sources']==meta['sources'] and started['selected_sha256']==s.sha(base/key/'best.pt') and started['model_digest']==s.digest(best['model']) and started['selected_epoch']==best['epoch']==result['selected_epoch']
            assert result['test_evaluations']==started['test_evaluations']==1 and result['test_used_for_selection'] is False
            actual,checks[key]=verify_predictions(folder/'predictions.npz',best['model'],'test',m);agree(actual,result['metrics'])
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=meta['sources'],training_commit=meta['git_commit'],checks=checks,model_forward_calls=0,test_images_read=0,cpu_models_for_shape_seed=len(study.CASES) if phase=='fit' else 0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
