"""Independent inverse-temperature root and saved-state audit; no model forwards."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
import caltech20_alt246_richsplit_tempcal as study
from audit_caltech20_alt246_richsplit import verify_predictions,metrics,agree

def probability(features,W,bias):
    z=features.astype(np.float64)@W.astype(np.float64).T+bias.astype(np.float64);z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists() and base.name==study.RUN;parent=study.parent();meta=s.read(base/'metadata.json');assert meta['parent']==parent and meta['sources']==study.sources() and meta['config']==study.config();study.engine.verify_sources(meta['sources'],meta['git_commit']);f=s.read(base/'fit/fit_result.json');assert f['best_sha256']==s.sha(f['best_path']) and f['last_sha256']==s.sha(base/'fit/last.pt');original=study.parent_state(parent);best=torch.load(f['best_path'],map_location='cpu',weights_only=False);state=best['model'];T=f['temperature'];assert .5<=T<=3 and best['epoch']==f['actual_epochs']==f['selected_epoch']==120 and best['contract']==parent['contract'] and best['sources']==study.sources() and best['config']==study.config() and best['calibration_uses_VAL_labels'] is True and f['additional_optical_epochs']==f['optical_updates']==0 and f['trainable_degrees_of_freedom']==1 and f['inference_parameters_added']==0
    assert set(state)==set(original) and study.head.frozen_digest(state)==study.head.frozen_digest(original)==f['frozen_digest']
    for name,value in original.items():
        if not name.startswith('linear.'):assert torch.equal(value,state[name]),name
    for name in ('linear.weight','linear.bias'):assert np.array_equal(state[name].numpy(),(original[name].numpy().astype(np.float64)/T).astype(np.float32)),name
    assert sum(q.numel() for name,q in state.items() if name.startswith('linear.'))==5140;ex=s.read(base/(phase+'_exit.json'));proc=Path('/proc',str(ex['pid']),'cmdline');assert ex['returncode']==0 and ex['actual_exit'] and (not proc.exists() or str(base/phase).encode() not in proc.read_bytes());m=study.b.dataset.manifest();checks={}
    if phase=='fit':
        extraction=s.read(base/'extract/result.json');assert extraction['parent']==parent and extraction['sources']==meta['sources'] and extraction['model_digest']==parent['EMA_digest'] and extraction['TEST_images']==extraction['optimizer_updates']==0;exit=s.read(base/'extract_exit.json');proc=Path('/proc',str(exit['pid']),'cmdline');assert exit['returncode']==0 and exit['actual_exit'] and (not proc.exists() or str(base/'extract').encode() not in proc.read_bytes());gate=s.read(base/'extract/numeric_gate.json');assert gate['passed'] and gate['micro1_vs4_probability_capture_maxabs']<=2e-6 and gate['parameters_frozen'];data={}
        for role in ('train','val'):
            rawpath=base/'extract'/(role+'.npz');assert s.sha(rawpath)==extraction['prediction_sha256'][role];actual,checks['unscaled_'+role]=verify_predictions(rawpath,original,role,m);agree(actual,extraction['metrics'][role])
            with np.load(rawpath,allow_pickle=False) as z:data[role]={k:z[k].copy() for k in z.files}
            selected=Path(f['selected_'+role+'_path']);actual,checks[role]=verify_predictions(selected,state,role,m);agree(actual,f['metrics'][role]);agree(actual,best['metrics'][role])
            with np.load(selected,allow_pickle=False) as z:
                for name in ('features','probe_amplitudes','ids','labels'):assert np.array_equal(z[name],data[role][name])
                assert np.array_equal(z['probabilities'].argmax(1),data[role]['probabilities'].argmax(1))
        W=original['linear.weight'].numpy().astype(np.float64);bias=original['linear.bias'].numpy().astype(np.float64);z=data['val']['features'].astype(np.float64)@W.T+bias;labels=data['val']['labels'];weight=1/(20*np.bincount(labels,minlength=20)[labels])
        def objective(logT):
            x=z/np.exp(logT);x-=x.max(1,keepdims=True);return float(np.sum(weight*(np.log(np.exp(x).sum(1))-x[np.arange(len(x)),labels])))
        def derivative_inverse(inverse):
            x=z*inverse;x-=x.max(1,keepdims=True);p=np.exp(x);p/=p.sum(1,keepdims=True);return float(np.sum(weight*((p*z).sum(1)-z[np.arange(len(z)),labels])))
        low,high=1/3,2.;d0,d1=derivative_inverse(low),derivative_inverse(high);inverse=low if d0>=0 else high if d1<=0 else brentq(derivative_inverse,low,high,xtol=1e-12);reference_T=1/inverse;last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);solver=last['solver'];assert solver['success'] and len(last['scalar_trace'])==solver['nfev']+1==f['scalar_objective_calls'] and abs(reference_T-np.exp(solver['log_temperature']))<1e-5
        for row in last['scalar_trace']:assert abs(objective(row['log_temperature'])-row['balanced_VAL_NLL'])<1e-9
        assert abs(objective(0)-last['baseline_float64_balanced_VAL_NLL'])<1e-9 and objective(np.log(reference_T))<=objective(0)+1e-9;candidate_T=float(np.exp(solver['log_temperature']));native_W=(W/candidate_T).astype(np.float32);native_b=(bias/candidate_T).astype(np.float32);candidate=metrics(labels,probability(data['val']['features'],native_W,native_b));baseline=metrics(labels,probability(data['val']['features'],W.astype(np.float32),bias.astype(np.float32)))
        # The decision is made using native float32 head probabilities; audit its value independently.
        if T!=1:assert abs(T-candidate_T)<1e-12 and candidate['balanced_nll']<=baseline['balanced_nll']+1e-6
        else:assert candidate['balanced_nll']>=baseline['balanced_nll']-1e-6
        checks['temperature']=dict(selected=T,independent_inverse_root_temperature=reference_T,baseline_float64_balanced_VAL_NLL=objective(0),best_float64_balanced_VAL_NLL=objective(np.log(reference_T)),argmax_unchanged_TRAIN_VAL=True,calibration_used_VAL_labels=True)
    else:
        study.previous.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['best_sha256']==started['best_sha256']==f['best_sha256'] and t['model_digest']==started['model_digest']==s.digest(state) and t['temperature']==started['temperature']==T and t['selected_epoch']==started['selected_epoch']==120 and t['test_used_for_selection'] is False and t['VAL_used_for_calibration'] is True and t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=verify_predictions(Path(t['predictions_path']),state,'test',m);agree(actual,t['metrics'])
        if t['reused']:assert t['new_test_evaluations']==0 and t['model_digest'] in parent['test_registry'] and t['predictions_path']==parent['test_registry'][t['model_digest']]['predictions_path']
        else:assert t['new_test_evaluations']==1 and t['model_digest'] not in parent['test_registry']
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=study.sources(),training_commit=meta['git_commit'],checks=checks,all_non_head_tensors_frozen=True,inference_parameters_added=0,terminal_linear_parameters=5140,VAL_used_for_calibration=True,model_forwards=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
