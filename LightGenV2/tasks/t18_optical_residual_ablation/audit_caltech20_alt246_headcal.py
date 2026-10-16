"""Independent saved linear-head audit; no optical/model inference."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_headcal as study
from audit_caltech20_alt246_biasfree import verify_predictions,agree

def softmax(logits):
    x=logits-logits.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();p=study.admit();meta=s.read(base/'metadata.json');assert meta['sources']==study.sources() and meta['config']==p;study.engine.verify_sources(meta['sources'],meta['git_commit'])
    fit=base/'fit';f=s.read(fit/'fit_result.json');m=s.read(fit/'metadata.json');assert m['sources']==meta['sources'] and m['config']==p;study.engine.verify_sources(m['sources'],m['git_commit']);assert f['history_sha256']==s.sha(fit/'head_history.pt') and f['best_sha256']==s.sha(f['best_path'])
    parent=torch.load(p['parent']['best_path'],map_location='cpu',weights_only=False);checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);original=parent['model'];state=checkpoint['model'];assert set(state)==set(original)
    assert parent['contract']['main_depth']==6 and parent['contract']['rho']==.3 and parent['contract']['terminal_linear_parameters']==5140 and parent['contract']==checkpoint['contract']
    for name,v in original.items():
        if not name.startswith('linear.'):assert torch.equal(v,state[name]),('Unfrozen',name)
    assert study.frozen_digest(state)==study.frozen_digest(original)==f['frozen_digest']==m['frozen_digest']
    assert state['linear.weight'].shape==(20,256) and state['linear.bias'].shape==(20,) and sum(v.numel() for n,v in state.items() if n.startswith('linear.'))==5140
    manifest=study.g.b.old.manifest(Path(study.g.b.config()['data']));checks={}
    if phase=='fit':
        ex=s.read(base/'fit_exit.json');proc=Path('/proc',str(ex['pid']),'cmdline');assert ex['returncode']==0 and ex['actual_exit'] and (not proc.exists() or study.RUN.encode() not in proc.read_bytes())
        with np.load(p['parent']['train_path'],allow_pickle=False) as z:X=z['features'].astype(np.float64);y=z['labels']
        assert np.bincount(y,minlength=20).tolist()==manifest['supports']['train'];mu=X.mean(0);sigma=np.maximum(X.std(0),p['feature_std_floor']);Z=(X-mu)/sigma
        W=original['linear.weight'].numpy().astype(np.float64);B=original['linear.bias'].numpy().astype(np.float64);w0=W*sigma[None];b0=B+W@mu;weights=len(y)/(20*np.bincount(y,minlength=20));T=(np.eye(20)[y]*(1-p['label_smoothing'])+p['label_smoothing']/20)*weights[None]
        logits=Z@w0.T+b0;shift=logits-logits.max(1,keepdims=True);logp=shift-np.log(np.exp(shift).sum(1,keepdims=True));ce=float(-(T*logp).sum(1).mean());regularization=float(np.clip(2*p['initial_penalty_fraction']*ce/(w0*w0).sum(),*p['l2_bounds']))
        assert abs(regularization-f['regularization'])<1e-10 and abs(ce-m['initial_weighted_ce'])<1e-9
        saved=torch.load(fit/'head_history.pt',map_location='cpu',weights_only=False);assert np.allclose(saved['mu'].numpy(),mu,atol=1e-12) and np.allclose(saved['sigma'].numpy(),sigma,atol=1e-12) and np.allclose(saved['initial_w'].numpy(),w0,atol=1e-12) and np.allclose(saved['initial_b'].numpy(),b0,atol=1e-12)
        history=saved['history'];assert len(history)==f['head_updates']<=p['maximum_head_updates'];ema_w=w0;ema_b=b0;selected=0;best=parent['metrics']['val']['balanced_nll'];chosen_w=W;chosen_b=B
        for number,row in enumerate(history,1):
            assert row['step']==number;raw_w=row['raw_w'].numpy();raw_b=row['raw_b'].numpy();ema_w=p['head_ema']*ema_w+(1-p['head_ema'])*raw_w;ema_b=p['head_ema']*ema_b+(1-p['head_ema'])*raw_b
            assert np.allclose(ema_w,row['ema_w'].numpy(),atol=1e-12) and np.allclose(ema_b,row['ema_b'].numpy(),atol=1e-12)
            raw=Z@raw_w.T+raw_b;raw-=raw.max(1,keepdims=True);logp=raw-np.log(np.exp(raw).sum(1,keepdims=True));objective=float(-(T*logp).sum(1).mean()+regularization*.5*(raw_w*raw_w).sum());assert abs(objective-row['raw_objective'])<1e-8
            native=(ema_w/sigma[None]).astype(np.float32);bias=(ema_b-(ema_w/sigma[None])@mu).astype(np.float32)
            for role in ('train','val'):
                with np.load(p['parent'][role+'_path'],allow_pickle=False) as z:F=z['features'].astype(np.float64);Y=z['labels']
                logits=F@native.astype(np.float64).T+bias.astype(np.float64);prob=softmax(logits);actual=row[role+'_probabilities'].numpy().astype(np.float64)
                assert np.abs(prob-actual).max()<3e-6;metrics=study.engine.metrics(Y,actual);agree(metrics,row[role])
            if row['val']['balanced_nll']<best-p['selection_tie_tolerance']:best=row['val']['balanced_nll'];selected=number;chosen_w=native;chosen_b=bias
        assert f['selected_head_step']==selected and np.array_equal(state['linear.weight'].numpy(),chosen_w.astype(np.float32)) and np.array_equal(state['linear.bias'].numpy(),chosen_b.astype(np.float32))
        if selected==0:assert f['best_path']==p['parent']['best_path']
        else:assert checkpoint['head_step']==selected and checkpoint['sources']==meta['sources']
        opt=saved['optimizer']['state'];assert len(opt)==1 and next(iter(opt.values()))['func_evals']==f['closure_calls']
        for role in ('train','val'):
            metrics,checks[role]=verify_predictions(Path(f['selected_'+role+'_path']),state,role,manifest);agree(metrics,f['metrics'][role])
        assert f['optical_model_forwards']==f['optical_updates']==0 and f['trainable_parameters']==5140
    else:
        receipt=s.read(base/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(base/'science/audit.json')
        result=s.read(base/'test/result.json');assert result['best_sha256']==f['best_sha256'] and result['selected_head_step']==f['selected_head_step'] and result['optical_forwards']==0 and result['test_used_for_update_selection'] is False
        path=Path(result['predictions_path']);assert s.sha(path)==result['predictions_sha256'];metrics,checks['test']=verify_predictions(path,state,'test',manifest);agree(metrics,result['metrics'])
        assert result['new_test_evaluations']==(0 if f['selected_head_step']==0 else 1)
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=meta['sources'],actual_source_commit=meta['git_commit'],checks=checks,non_head_parameters_frozen=True,inference_parameters_added=0,linear_parameters=5140,optical_model_forwards=0,test_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        path=a.base/(a.phase+'_audit_failure.json')
        if not path.exists():study.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
