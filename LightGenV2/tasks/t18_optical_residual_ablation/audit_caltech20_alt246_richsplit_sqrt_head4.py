"""Independent fixed-backbone/head-only objective and saved inference audit."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_sqrt_head4 as study
from audit_caltech20_alt246_richsplit import verify_predictions,agree,metrics

def softmax(logits):
    z=logits-logits.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists() and base.name==study.RUN;parent=study.admit();cfg=study.config();meta=s.read(base/'metadata.json');assert meta['sources']==study.sources() and meta['config']==cfg and meta['parent']==parent;study.engine.verify_sources(meta['sources'],meta['git_commit']);f=s.read(base/'fit/fit_result.json');fm=s.read(base/'fit/metadata.json');assert fm['sources']==meta['sources'] and fm['parent']==parent and fm['config']==cfg and f['best_sha256']==s.sha(f['best_path']) and f['last_sha256']==s.sha(base/'fit/last.pt')
    original=torch.load(parent['best'],map_location='cpu',weights_only=False);best=torch.load(f['best_path'],map_location='cpu',weights_only=False);state=best['model'];old=original['model'];assert original['contract']==best['contract'] and original['contract']['main_depth']==4 and original['contract']['terminal_linear_parameters']==5140
    for name,value in old.items():
        if not name.startswith('linear.'):assert torch.equal(value,state[name]),name
    assert study.frozen_digest(state)==study.frozen_digest(old)==f['frozen_digest']==fm['frozen_digest'];assert sum(value.numel() for name,value in state.items() if name.startswith('linear.'))==5140
    ex=s.read(base/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(ex['pid']),'cmdline');assert not proc.exists() or str(base/phase).encode() not in proc.read_bytes();m=study.b.dataset.manifest();checks={}
    if phase=='fit':
        saved=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);history=saved['history'];assert len(history)==f['head_updates']<=20 and f['head_attempts'] in (len(history),len(history)+1) and f['optical_epochs_spent']==100 and f['selected_optical_epoch']==parent['selected_epoch'] and f['optical_forwards']==f['optical_updates']==0 and f['fit_seconds']<=cfg['maximum_seconds']
        w0=old['linear.weight'].numpy().astype(np.float64);b0=old['linear.bias'].numpy().astype(np.float64);assert np.array_equal(saved['initial_w'].numpy(),w0) and np.array_equal(saved['initial_b'].numpy(),b0);data={}
        for role in ('train','val'):
            with np.load(parent[role],allow_pickle=False) as z:data[role]={k:z[k].copy() for k in z.files}
        y=data['train']['labels'];Y=(np.eye(20)[y]*(1-cfg['label_smoothing'])+cfg['label_smoothing']/20)*np.array(cfg['class_weights'])[None];best_key=(-parent['metrics']['val']['accuracy'],parent['metrics']['val']['nll'],0);selected=0;chosen_w=w0.astype(np.float32);chosen_b=b0.astype(np.float32)
        for i,row in enumerate(history,1):
            assert row['step']==i and row['optimizer_n_iter']>=i and row['closure_calls']>=i;w=row['raw_w'].numpy();bias=row['raw_b'].numpy();logits=data['train']['features'].astype(np.float64)@w.T+bias;z=logits-logits.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True));expected=float(-(Y*logp).sum(1).mean()+.5*cfg['anchor_l2']*(np.square(w-w0).sum()+np.square(bias-b0).sum()));assert abs(expected-row['objective'])<1e-8
            native=w.astype(np.float32);nb=bias.astype(np.float32)
            for role in ('train','val'):
                arrays=data[role];prob=softmax(arrays['features'].astype(np.float64)@native.astype(np.float64).T+nb.astype(np.float64));actual=row[role+'_probabilities'].numpy().astype(np.float64);assert np.abs(prob-actual).max()<=3e-6;agree(metrics(arrays['labels'],actual),row[role])
            if (-row['val']['accuracy'],row['val']['nll'],i)<best_key:best_key=(-row['val']['accuracy'],row['val']['nll'],i);selected=i;chosen_w=native;chosen_b=nb
        assert f['selected_head_step']==selected and np.array_equal(chosen_w,state['linear.weight'].numpy()) and np.array_equal(chosen_b,state['linear.bias'].numpy())
        opt=saved['optimizer']['state'];assert len(opt)==1 and next(iter(opt.values()))['func_evals']==f['closure_calls']
        if selected==0:assert f['best_path']==parent['best']
        else:assert best['head_step']==selected and best['sources']==study.sources() and best['epoch']==parent['selected_epoch']
        for role in ('train','val'):
            path=Path(f['selected_'+role+'_path']);actual,checks[role]=verify_predictions(path,state,role,m);agree(actual,f['metrics'][role]);arrays=data[role]
            with np.load(path,allow_pickle=False) as z:
                for name in ('features','probe_amplitudes','labels','ids'):assert np.array_equal(z[name],arrays[name]),(role,name)
    else:
        study.v.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['best_sha256']==started['best_sha256']==f['best_sha256'] and t['selected_head_step']==started['selected_head_step']==f['selected_head_step'] and t['model_digest']==started['model_digest']==s.digest(state) and t['test_used_for_epoch_selection'] is False and t['predictions_sha256']==s.sha(t['predictions_path'])
        actual,checks['test']=verify_predictions(Path(t['predictions_path']),state,'test',m);agree(actual,t['metrics'])
        if t['reused']:assert t['new_test_evaluations']==t['new_optical_TEST_evaluations']==0 and t['model_digest'] in parent['test_registry'] and t['predictions_path']==parent['test_registry'][t['model_digest']]['predictions_path']
        else:assert t['new_test_evaluations']==t['new_optical_TEST_evaluations']==1 and t['model_digest'] not in parent['test_registry']
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=study.sources(),training_commit=meta['git_commit'],checks=checks,all_non_head_tensors_frozen=True,inference_parameters_added=0,terminal_linear_parameters=5140,model_forwards=0,TEST_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
