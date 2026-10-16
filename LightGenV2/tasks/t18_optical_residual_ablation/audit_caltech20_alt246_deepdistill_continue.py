"""Independent saved audit of the single six-layer KD method; zero forwards."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_deepdistill_continue as study
from audit_caltech20_alt246_biasfree import plans,verify_predictions,agree,history_metric
from audit_caltech20_alt246_genreg import masks,moment_digest

def exited(pid):
    p=Path('/proc',str(pid),'cmdline');return not p.exists() or study.RUN.encode() not in p.read_bytes()

def audit(base,out,phase):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();s=study.s;p=study.admit();cfg=study.config();meta=s.read(base/'metadata.json');assert meta['sources']==study.sources() and meta['config']==cfg and meta['profile']==p;study.engine.verify_sources(meta['sources'],meta['git_commit']);checks={};torch.set_num_threads(4)
    folder=base/'fit';f=s.read(folder/'fit_result.json');fm=s.read(folder/'metadata.json');assert fm['sources']==meta['sources'] and fm['config']==cfg and fm['profile']==p;study.engine.verify_sources(fm['sources'],fm['git_commit'])
    assert s.sha(f['best_path'])==f['best_sha256'] and s.sha(folder/'last.pt')==f['last_sha256'] and s.sha(folder/'history.json')==f['history_sha256'];best=torch.load(f['best_path'],map_location='cpu',weights_only=False);manifest=study.b.old.manifest(Path(study.b.config()['data']))
    ex=s.read(base/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and exited(ex['pid']);assert not (base/phase/'failure.json').exists()
    if phase=='fit':
        parent=torch.load(p['parent_last']['path'],map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);teacher=torch.load(p['teacher']['path'],map_location='cpu',weights_only=False)
        assert parent['epoch']==140 and parent['steps']==12040 and fm['parent_raw_digest']==s.digest(parent['model']) and fm['parent_ema_digest']==s.digest(parent['ema']) and fm['initial_optimizer_moment_digest']==moment_digest(parent['optimizer'])
        assert fm['teacher_digest']==s.digest(teacher['model'])==f['teacher_digest'] and f['teacher_frozen'] and teacher['contract']['main_depth']==2 and all(.7<a<=.95 for a in teacher['metrics']['val']['alphas'])
        h=s.read(folder/'history.json');assert len(h)==f['additional_epochs']==20 and f['actual_epochs']==last['epoch']==160 and f['steps']==last['steps']==13760
        model=study.RegularizedMoE(6);model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True);assert model.contract()==best['contract'] and len(last['optimizer']['state'])==len(list(model.parameters())) and all(float(v['step'])==13760 for v in last['optimizer']['state'].values())
        assert [q['weight_decay'] for q in last['optimizer']['param_groups']]==[0,cfg['electronic_weight_decay']] and last['config']==cfg and last['sources']==meta['sources']
        for name,v in last['previous_final_ema'].items():
            expected=v.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][name].numpy().astype(np.float64)*(1-cfg['ema_decay']);assert np.allclose(expected,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6)
        loss=p['prior_metrics']['val']['balanced_nll'];selected=p['prior_selected_epoch'];headstep=p['prior_selected_head_step'];selected_digest=None;lr=parent['optimizer']['param_groups'][0]['lr'];scheduler_best=parent['scheduler']['best'];bad=parent['scheduler']['num_bad_epochs']
        for epoch,row in enumerate(h,141):
            assert row['epoch']==epoch and row['adam_steps']==86*epoch and row['teacher_samples']==1372 and 0<=row['teacher_eligible']<=1372 and np.isfinite(row['mean_temperature_scaled_kl']) and row['mean_temperature_scaled_kl']>=-1e-6 and row['teacher_digest']==fm['teacher_digest']
            assert tuple(row[k] for k in ('order_sha256','affine_sha256','mix_sha256'))==plans(epoch,cfg,1372) and (row['flip_sha256'],row['mask_sha256'])==masks(epoch,cfg)
            assert np.allclose(row['lr'],[lr,lr],atol=1e-12,rtol=0)
            for role in ('train','val'):history_metric(row[role],manifest['supports'][role])
            if row['val']['balanced_nll']<loss-1e-6:loss=row['val']['balanced_nll'];selected=epoch;headstep=0;selected_digest=row['ema_digest']
            if row['val']['balanced_nll']<scheduler_best-.002:scheduler_best=row['val']['balanced_nll'];bad=0
            else:bad+=1
            if bad>3:lr=max(lr*.5,2e-5);bad=0
        assert f['selected_epoch']==selected==best['epoch'] and f['selected_head_step']==headstep and best.get('head_step',0)==headstep
        if selected_digest is None:assert f['best_sha256']==p['prior_best']['sha256']
        else:assert s.digest(best['model'])==selected_digest and best['sources']==meta['sources']
        assert last['scheduler']['last_epoch']==40 and last['scheduler']['num_bad_epochs']==bad and abs(last['scheduler']['best']-scheduler_best)<1e-10 and np.allclose([q['lr'] for q in last['optimizer']['param_groups']],[lr,lr],atol=1e-12,rtol=0)
        assert s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest']
        for role in ('train','val'):
            metrics,checks[role]=verify_predictions(Path(f['selected_'+role+'_path']),best['model'],role,manifest);agree(metrics,f['metrics'][role])
        gate=s.read(folder/'numeric_gate.json');assert gate['passed'] and gate['initial_digest']==fm['parent_raw_digest'] and gate['teacher_digest']==fm['teacher_digest'] and gate['teacher_frozen'] and gate['optimizer_steps']==gate['test_arrays_read']==0 and {q['samples'] for q in gate['checks']}=={'all20','macro16','tail12'}
        for row in gate['checks']:assert set(row['gradient_norms'])==set(dict(model.named_parameters())) and all(np.isfinite(v) and v>0 for v in row['gradient_norms'].values()) and row['gradient_relative_l2']<=.02 and row['gradient_cosine']>=.999 and row['maxabs']<2e-6
        checks['teacher_coverage_per_epoch']=[q['teacher_eligible']/1372 for q in h];checks['final_student_contract']=model.contract()
    else:
        receipt=s.read(base/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(base/'science/audit.json');t=s.read(base/'test/result.json');assert t['best_sha256']==f['best_sha256'] and t['selected_epoch']==f['selected_epoch'] and t['teacher_in_inference'] is False and t['test_used_for_epoch_selection'] is False and s.sha(t['predictions_path'])==t['predictions_sha256']
        prior=torch.load(p['prior_best']['path'],map_location='cpu',weights_only=False);original=torch.load(study.s.read(study.HERE/'caltech20_alt246_deepdistill_profile.json')['prior_best']['path'],map_location='cpu',weights_only=False);known={s.digest(prior['model']),s.digest(original['model']),study.h.config()['parent']['model_digest']};assert t['model_digest']==s.digest(best['model']) and t['new_test_evaluations']==(0 if t['model_digest'] in known else 1);metrics,checks['test']=verify_predictions(Path(t['predictions_path']),best['model'],'test',manifest);agree(metrics,t['metrics'])
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=meta['sources'],actual_training_commit=meta['git_commit'],checks=checks,teacher_only_in_training=True,inference_parameters_added=0,model_forward_calls=0,test_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        path=a.base/(a.phase+'_audit_failure.json')
        if not path.exists():study.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
