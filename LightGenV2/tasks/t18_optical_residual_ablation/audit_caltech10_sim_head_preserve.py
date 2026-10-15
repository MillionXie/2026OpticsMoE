"""Independent saved-feature/gradient-objective/selected-readout verification."""
import argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech10_sim_head_preserve as run
from audit_caltech10_sim_head import probabilities,metric,close
s=run.s
def audit(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (out/'audit.json').exists();meta=s.read(out.parent/'metadata.json');assert meta['sources']==run.sources();s.engine.verify_sources(meta['sources'],meta['git_commit']);assert s.read(out/'audit_claim.json')['sources']==run.sources();case=meta['case'];c=run.P['cases'][case];op=run.P['optical_parents'][case];assert s.sha(op['path'])==op['sha'] and s.sha(c['head_path'])==c['head_sha256'];fit=s.read(out/'fit_result.json');last=torch.load(out/'last_checkpoint.pt',map_location='cpu',weights_only=False);best=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);hist=s.read(out/'history.json');expected=max(hist,key=run.select_key);assert expected['head_epoch']==fit['selected_head_epoch']==best['epoch'] and hist[0]['head_epoch']==3 and [h['phase_epoch'] for h in hist]==list(range(len(hist)));assert last['epoch']==fit['actual_head_epochs'] and all(int(q['step'])==12*fit['new_phase_epochs'] for q in last['optimizer']['state'].values()) and fit['head_parameters']==8554 and fit['optical_optimizer_steps']==0 and fit['contract']==run.CONTRACT;assert run.prior.digest(best['head_state'])==fit['head_digest'] and sum(q.numel() for k,q in best['head_state'].items() if k.startswith('net.'))==8554
    parent=torch.load(c['head_path'],map_location='cpu',weights_only=False);old=Path(run.P['cache_root'])/case;cache={role:dict(np.load(old/('cache_'+role+'.npz'),allow_pickle=False)) for role in ('train','val','test')};manifest=run.prior.optical.manifest(Path(op['config']['data']))
    for role,z in cache.items():assert z['ids'].tolist()==manifest['split_ids'][role] and np.bincount(z['labels'],minlength=10).tolist()==manifest['supports'][role] and s.read((old/('cache_'+role+'.npz')).with_suffix('.receipt.json'))['cache_sha256']==s.sha(old/('cache_'+role+'.npz'))
    for ck in (last,best):
        for k in ('mean','std'):assert torch.equal(ck['head_state'][k],parent['head_state'][k])
        for role,key in (('train','train'),('val','validation')):z=cache[role];p=probabilities(ck['head_state'],z['features'],z['base']);close(metric(z['labels'],p),ck[key])
    loss_errors=[];z=cache['train']
    for path in sorted(out.glob('loss_epoch*.npz')):
        a=dict(np.load(path,allow_pickle=False));order=a['order'];epoch=int(path.stem[-4:]);expected_order=torch.randperm(761,generator=torch.Generator().manual_seed(17+epoch)).numpy();assert np.array_equal(order,expected_order);logits=a['logits'].astype(np.float64);base=z['base'][order].astype(np.float64);labels=z['labels'][order];correct=base.argmax(1)==labels;lp=logits-logits.max(1,keepdims=True);lp-=np.log(np.exp(lp).sum(1,keepdims=True));ce=-lp[np.arange(761),labels];kl=(base*(np.log(base.clip(1e-8))-lp)).sum(1);terms=np.where(correct,run.P['correct_CE_weight'],1.)*ce+run.P['correct_KL_weight']*correct*kl;independent=np.array([terms[i:i+64].mean() for i in range(0,761,64)]);error=float(np.max(np.abs(independent-a['loss'])));assert error<2e-5;loss_errors.append(error)
    assert len(loss_errors)==3
    td=Path(fit['test_path']);test=s.read(td/'result.json');saved=dict(np.load(td/'predictions.npz',allow_pickle=False));z=cache['test'];assert test['head_digest']==fit['head_digest'] and test['test']==fit['test'] and test['prediction_sha256']==s.sha(td/'predictions.npz') and np.array_equal(saved['ids'],z['ids']) and np.array_equal(saved['labels'],z['labels']);p=probabilities(best['head_state'],z['features'],z['base']);error=float(np.max(np.abs(p-saved['probability'])));assert error<2e-5 and np.array_equal(p.argmax(1),saved['probability'].argmax(1));close(metric(z['labels'],saved['probability']),test['test'],tol=1e-9);assert run.P['controls'][case]==fit['control'];gap=100*(fit['test']['accuracy']-fit['control']['test']['accuracy']);assert math.isclose(gap,fit['test_gap_pp'],abs_tol=1e-12);s.save(out/'audit.json',dict(passed=True,sources=run.sources(),head_parameters=8554,parent_head_sha256=c['head_sha256'],optical_checkpoint_sha256=op['sha'],head_digest=fit['head_digest'],independent_loss_max_error=max(loss_errors),independent_probability_max_error=error,test_result_sha256=s.sha(td/'result.json'),head_optimizer_steps=12*fit['new_phase_epochs'],selected_head_epoch=best['epoch'],head_optimizer_paused_during_synchronous_audit=True,producer_final_exit_verified_separately=True,optical_forward_calls=0,optical_models_constructed=0,test_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a.out)
    except BaseException as error:
        if not (a.out/'audit_failure.json').exists():s.save(a.out/'audit_failure.json',dict(error=repr(error),automatic_retry=False),exclusive=True)
        raise
