"""Independent NumPy saved-head audit; never constructs an optical model."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch
import caltech10_sim_head as run
s=run.s
def metric(y,p):
    y=np.asarray(y);p=np.asarray(p,np.float64);pred=p.argmax(1);cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,pred),1);n=cm.sum(1);assert (n>0).all();loss=-np.log(p[np.arange(len(y)),y].clip(1e-12));return dict(n=len(y),accuracy=float((pred==y).mean()),balanced_accuracy=float(np.mean(cm.diagonal()/n)),recall=(cm.diagonal()/n).tolist(),support=n.tolist(),confusion_matrix=cm.tolist(),nll=float(loss.mean()),balanced_nll=float(np.mean([loss[y==i].mean() for i in range(10)])))
def probabilities(state,features,base):
    z={k:q.detach().cpu().numpy() for k,q in state.items()};x=(features-z['mean'])/z['std'];h=np.maximum(x@z['net.0.weight'].T+z['net.0.bias'],0);logits=h@z['net.2.weight'].T+z['net.2.bias']+np.log(base.clip(1e-8));q=np.exp(logits-logits.max(1,keepdims=True));return q/q.sum(1,keepdims=True)
def close(a,b,tol=2e-5):
    assert a.keys()==b.keys()
    for k in a:
        if isinstance(a[k],float):assert math.isclose(a[k],b[k],abs_tol=tol,rel_tol=tol),(k,a[k],b[k])
        else:assert a[k]==b[k],k
def audit(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (out/'audit.json').exists();claim=s.read(out/'claim.json');assert claim['sources']==run.sources();s.engine.verify_sources(claim['sources'],claim['git_commit']);rec=s.read(out/'producer_exit.json');assert rec['returncode']==0 and rec['actual_exit'] and run.optical.base.exited(rec['pid']);fit=s.read(out/'fit_result.json');case=fit['case'];parent=out.parent;c=run.P['parents'][case];assert fit['sources']==run.sources() and fit['head_parameters']==8554 and fit['head_contract']==run.CONTRACT and fit['optical_optimizer_steps']==0 and fit['optical_weights_unchanged'] and fit['optical_checkpoint_sha256']==c['sha']==s.sha(c['path']) and fit['optical_model_digest']==c['digest'];proof=s.read(run.CPU/'checks.json');assert proof['passed'] and fit['initial_head_net_digest']==proof['initial_head_net_digest'];gate=s.read(parent/'segment0001/numeric_gate.json');assert gate['passed'] and gate['optimizer_steps']==0 and gate['head_parameters']==8554
    cfg=c['config'];m=run.optical.manifest(Path(cfg['data']));cache={};pool_errors={}
    for role in ('train','val','test'):
        path=parent/('cache_'+role+'.npz');receipt=s.read(path.with_suffix('.receipt.json'));assert receipt['cache_sha256']==s.sha(path) and receipt['optical_checkpoint_sha256']==c['sha'] and receipt['optical_model_digest']==c['digest'] and receipt['contract']==run.CONTRACT and receipt['sources']==run.sources();z=dict(np.load(path,allow_pickle=False));assert z['ids'].tolist()==m['split_ids'][role] and np.bincount(z['labels'],minlength=10).tolist()==m['supports'][role] and z['features'].shape==(len(z['labels']),256) and np.isfinite(z['features']).all();cache[role]=z
        raw=z['probe_ccd'].astype(np.float64);value=np.log1p(raw/np.maximum(raw.mean((-2,-1),keepdims=True),1e-6));feat=np.stack([value[:,int(np.floor(i*478/16)):int(np.ceil((i+1)*478/16)),int(np.floor(j*478/16)):int(np.ceil((j+1)*478/16))].mean((1,2)) for i in range(16) for j in range(16)],1);error=float(np.max(np.abs(feat-z['features'][:2])));assert error<2e-5;pool_errors[role]=error
    assert not set(cache['train']['ids'])&set(cache['val']['ids']) and not set(cache['train']['ids'])&set(cache['test']['ids']) and not set(cache['val']['ids'])&set(cache['test']['ids'])
    history=s.read(out/'history.json');assert len(history)==fit['actual_head_epochs'] and [h['epoch'] for h in history]==list(range(1,len(history)+1));assert min(history,key=lambda h:h['val']['balanced_nll'])['epoch']==fit['selected_head_epoch'];assert fit['selected_head_epoch']>=1
    for h in history:
        order=torch.randperm(761,generator=torch.Generator().manual_seed(17+h['epoch']));assert h['order_sha256']==hashlib.sha256(order.numpy().astype('<i8').tobytes()).hexdigest() and h['optimizer_steps']==12*h['epoch']
    best=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert best['epoch']==fit['selected_head_epoch'] and last['epoch']==fit['actual_head_epochs'] and all(int(q['step'])==12*len(history) for q in last['optimizer']['state'].values());assert run.digest(best['head_state'])==fit['head_digest'] and sum(v.numel() for k,v in best['head_state'].items() if k.startswith('net.'))==8554
    mean=cache['train']['features'].mean(0,dtype=np.float64).astype(np.float32);std=cache['train']['features'].std(0,dtype=np.float64).astype(np.float32).clip(.001)
    for ck in (best,last):
        assert np.array_equal(ck['head_state']['mean'].numpy(),mean) and np.array_equal(ck['head_state']['std'].numpy(),std) and ck['optical_checkpoint_sha256']==c['sha'] and ck['contract']==run.CONTRACT
        for role,key in (('train','train'),('val','validation')):z=cache[role];p=probabilities(ck['head_state'],z['features'],z['base']);close(metric(z['labels'],p),ck[key])
    td=Path(fit['test_path']);test=s.read(td/'result.json');pred=dict(np.load(td/'predictions.npz',allow_pickle=False));z=cache['test'];assert test['head_digest']==fit['head_digest'] and test['prediction_sha256']==s.sha(td/'predictions.npz') and np.array_equal(pred['ids'],z['ids']) and np.array_equal(pred['labels'],z['labels']);independent=probabilities(best['head_state'],z['features'],z['base']);error=float(np.max(np.abs(independent-pred['probability'])));assert error<2e-5 and np.array_equal(independent.argmax(1),pred['probability'].argmax(1));close(metric(z['labels'],pred['probability']),fit['test'],tol=1e-9);assert fit['test']==test['test'];s.save(out/'audit.json',dict(passed=True,sources=run.sources(),case=case,head_parameters=8554,head_digest=fit['head_digest'],test_result_sha256=s.sha(td/'result.json'),prediction_sha256=s.sha(td/'predictions.npz'),independent_probability_max_error=error,independent_pooling_errors=pool_errors,optical_weights_frozen=True,model_forward_calls=0,optical_models_constructed=0,selected_head_epoch=best['epoch'],actual_head_epochs=last['epoch'],initial_head_net_digest=fit['initial_head_net_digest'],same_sim_hardware_contract=run.CONTRACT),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a.out)
    except BaseException as error:
        mark=a.out/'audit_failure.json'
        if not mark.exists():s.save(mark,dict(error=repr(error),automatic_retry=False),exclusive=True)
        raise
