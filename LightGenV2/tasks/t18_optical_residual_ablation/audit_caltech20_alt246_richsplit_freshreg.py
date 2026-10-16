"""Independent complete100 saved-math audit, zero optical forwards/TEST images."""
import argparse,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_freshreg as study
import audit_caltech20_alt246_richsplit as previous
from audit_caltech20_alt246_richsplit_cachedkd import reference_kd
from audit_caltech20_alt246_richsplit_viewreg import view_hashes

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();meta=s.read(base/'metadata.json');case=meta['case'];cfg=study.config(case);assert base==study.expected_root(case) and meta['sources']==study.sources() and meta['config']==cfg;study.engine.verify_sources(meta['sources'],meta['git_commit']);study.cpu_ready();m=study.b.dataset.manifest();torch.set_num_threads(4);checks={};f=s.read(base/'fit/fit_result.json');ex=s.read(base/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(ex['pid']),'cmdline');assert not proc.exists() or str(base/phase).encode() not in proc.read_bytes();best=torch.load(f['best_path'],map_location='cpu',weights_only=False);assert f['best_sha256']==s.sha(f['best_path']) and best['sources']==study.sources() and best['config']==cfg
    if phase=='fit':
        fm=s.read(base/'fit/metadata.json');last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);h=s.read(base/'fit/history.json');assert len(h)==100 and [q['epoch'] for q in h]==list(range(1,101)) and f['actual_epochs']==f['additional_epochs']==100 and f['fresh_restart_not_resume'] and f['stop_reason']=='maximum_epochs';assert f['steps']==last['steps']==19700 and last['epoch']==100 and last['sources']==study.sources() and last['config']==cfg
        assert fm['sources']==meta['sources'] and fm['config']==cfg and fm['versions']==study.versions() and fm['optimizer_state_restored'] is False and fm['teacher']==last['teacher']==study.k.teacher_info();assert f['last_sha256']==s.sha(base/'fit/last.pt') and f['history_sha256']==s.sha(base/'fit/history.json');assert f['fit_seconds']<=cfg['maximum_seconds']+max(q['epoch_seconds'] for q in h)+60
        fresh=study.initial_state();initial=s.digest(fresh.state_dict());assert fm['initial_digest']==best['initial_digest']==last['initial_digest']==initial==s.read(study.BASE/'L6_rho03/fit_result.json')['initial_digest'];assert fresh.contract()==best['contract'];fresh.load_state_dict(last['model'],strict=True);fresh.load_state_dict(best['model'],strict=True);assert all(.7<a<=.95 for a in fresh.alphas()) and len(list(fresh.parameters()))==30
        optimizer=last['optimizer'];assert len(optimizer['state'])==30 and all(float(q['step'])==19700 for q in optimizer['state'].values()) and [q['weight_decay'] for q in optimizer['param_groups']]==[0,cfg['electronic_weight_decay']];expected_lr=cfg['minimum_lr'];assert np.allclose([q['lr'] for q in optimizer['param_groups']],expected_lr,atol=1e-12,rtol=0) and last['scheduler']['last_epoch']==100 and last['scheduler']['T_max']==100 and last['scheduler']['eta_min']==expected_lr
        teacher_info=study.k.teacher_info();teacher=torch.load(teacher_info['best'],map_location='cpu',weights_only=False);previous.verify_predictions(Path(teacher_info['train']),teacher['model'],'train',m)
        with np.load(teacher_info['train'],allow_pickle=False) as z:probs=z['probabilities'].copy();labels=z['labels'].copy()
        with np.load(cfg['data'],allow_pickle=False) as z:images=z['train_images'].copy()
        n=len(labels);assert sorted(last['kd_student_probabilities'])==list(range(1,101))
        for row in h:
            epoch=row['epoch'];assert row['adam_steps']==197*epoch and tuple(row[q] for q in ('order_sha256','affine_sha256','mix_sha256'))==previous.plans(epoch,cfg,n) and tuple(row[q] for q in ('flip_sha256','mask_sha256'))==previous.mask_plans(epoch,cfg,n);lr=cfg['minimum_lr']+(cfg['lr']-cfg['minimum_lr'])*(1+math.cos(math.pi*(epoch-1)/100))/2;assert np.allclose(row['lr'],lr,atol=1e-12,rtol=0) and row['effective_KD_alpha']==cfg['kd_alpha']*min(epoch/10,1.)
            order=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013])).permutation(n).astype('<i8');views=view_hashes(images,order,epoch,cfg)
            for kind,value in views.items():assert row[kind+'_sha256']==value,(epoch,kind)
            record={q:[] for q in ('teacher_target','teacher_gate')};count=0;support=np.zeros(20,np.int64);kl=0.;student=last['kd_student_probabilities'][epoch].numpy().astype(np.float64);assert student.shape==(n,20) and np.isfinite(student).all() and (student>=0).all() and np.allclose(student.sum(1),1,atol=2e-6)
            for batch,off in enumerate(range(0,n,16)):
                idx=order[off:off+16];q,eligible=reference_kd(probs,labels,idx,epoch,batch,cfg);record['teacher_target'].append(hashlib.sha256(q.tobytes()).hexdigest());record['teacher_gate'].append(hashlib.sha256(eligible.tobytes()).hexdigest());count+=int(eligible.sum());support+=np.bincount(labels[idx],weights=eligible,minlength=20).astype(np.int64);p=student[off:off+len(idx)];log=np.log(np.maximum(p,1e-12))/cfg['kd_temperature'];temp=np.exp(log-log.max(1,keepdims=True));temp/=temp.sum(1,keepdims=True);kl+=float(cfg['kd_temperature']**2*np.sum((q.astype(np.float64)*(np.log(np.maximum(q,1e-12))-np.log(np.maximum(temp,1e-12)))).sum(1)*eligible))
            for kind,values in record.items():assert row[kind+'_sha256']==hashlib.sha256(''.join(values).encode()).hexdigest()
            assert count==row['kd_eligible_count'] and support.tolist()==row['kd_eligible_support'] and abs(kl-row['kd_KL_T2_sum'])<1e-6;checks['epoch'+str(epoch)]=dict(KD_eligible=count,KL_T2_sum=kl,TRAIN_view_replay=n)
            for role in ('train','val'):previous.history_metric(row[role],m['supports'][role])
        assert set(last['previous_final_ema'])==set(last['model'])==set(last['ema'])
        for name,earlier in last['previous_final_ema'].items():
            expected=earlier.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][name].numpy().astype(np.float64)*(1-cfg['ema_decay']);assert np.allclose(expected,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6),name
        # Independent lexicographic selector: higher accuracy, then NLL, then time.
        candidates=sorted(h,key=lambda q:(-float(q['val']['accuracy']),float(q['val']['nll']),int(q['epoch'])));selected=candidates[0];assert selected['epoch']==f['selected_epoch']==best['epoch'] and s.digest(best['model'])==selected['ema_digest'] and s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest']
        for role in ('train','val'):
            actual,checks[role]=previous.verify_predictions(Path(f['selected_'+role+'_path']),best['model'],role,m);previous.agree(actual,f['metrics'][role]);previous.agree(actual,selected[role]);previous.agree(actual,best['metrics'][role])
        gate=s.read(base/'fit/numeric_gate.json');assert gate['passed'] and gate['initial_digest']==initial and gate['optimizer_steps']==0 and len(gate['checks'])==6
        for q in gate['checks']:assert q['maxabs']<=2e-6 and q['gradient_relative_l2']<=.02 and q['gradient_cosine']>=.999 and set(q['gradient_norms'])==set(dict(fresh.named_parameters())) and all(np.isfinite(x) and x>0 for x in q['gradient_norms'].values())
        checks.update(fresh_same_original_initialization=True,new_TRAIN_photos_independently_replayed=100*n,full_epochs=100,Adam_updates=19700,parameter_tensors=30,terminal_parameters=5140)
    else:
        study.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['selected_epoch']==started['selected_epoch']==f['selected_epoch'] and t['model_digest']==started['model_digest']==s.digest(best['model']) and t['best_sha256']==started['best_sha256']==f['best_sha256'] and started['sources']==study.sources() and t['test_used_for_epoch_selection'] is False;assert t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=previous.verify_predictions(Path(t['predictions_path']),best['model'],'test',m);previous.agree(actual,t['metrics']);registry=study.prior.parent()['test_registry']
        if t['reused']:
            assert t['new_test_evaluations']==0 and t['model_digest'] in registry;known=registry[t['model_digest']];assert t['predictions_path']==known['predictions_path'] and t['predictions_sha256']==known['predictions_sha256']
        else:assert t['new_test_evaluations']==1 and t['model_digest'] not in registry and (base/('digest_'+t['model_digest']+'.claim.json')).exists()
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,run=str(base),sources=study.sources(),training_commit=meta['git_commit'],config=cfg,checks=checks,model_forwards=0,test_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
