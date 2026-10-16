"""Own prefix-state, new20 independent TRAIN replay and saved prediction audit."""
import argparse,hashlib,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_emastable as m
import audit_caltech20_alt246_richsplit as previous
from audit_caltech20_alt246_richsplit_cachedkd import reference_kd
from audit_caltech20_alt246_richsplit_viewreg import view_hashes
from check_caltech20_alt246_richsplit_emastable import independent_TV

def audit(base,out,phase):
    s=m.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();meta=s.read(base/'metadata.json');case=meta['case'];p=m.parent();cfg=m.config(case);assert base==m.expected_root(case) and meta['sources']==m.sources() and meta['parent']==p and meta['config']==cfg;m.engine.verify_sources(meta['sources'],meta['git_commit']);m.cpu_ready();torch.set_num_threads(4);f=s.read(base/'fit/fit_result.json');best=torch.load(f['best_path'],map_location='cpu',weights_only=False);assert f['best_sha256']==s.sha(f['best_path']);ex=s.read(base/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(ex['pid']),'cmdline');assert not proc.exists() or str(base/phase).encode() not in proc.read_bytes();manifest=m.b.dataset.manifest();checks={}
    if phase=='fit':
        fm=s.read(base/'fit/metadata.json');old=torch.load(p['last'],map_location='cpu',weights_only=False);last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);h=s.read(base/'fit/history.json');prefix=s.read(p['history_path']);assert h[:100]==prefix and len(h)==120 and f['actual_epochs']==120 and f['additional_epochs']==20 and f['steps']==last['steps']==23640 and last['epoch']==120 and f['stop_reason']=='registered20_epoch_EMA_horizon_review';assert f['last_sha256']==s.sha(base/'fit/last.pt') and f['history_sha256']==s.sha(base/'fit/history.json');assert f['fit_seconds']<=3600+max(q['epoch_seconds'] for q in h[100:])+60;assert fm['sources']==last['sources']==m.sources() and fm['config']==last['config']==cfg and fm['parent']==last['parent']==p and fm['versions']==m.versions();assert fm['initial_raw_digest']==s.digest(old['model']) and fm['initial_EMA_digest']==s.digest(old['ema']) and fm['initial_optimizer_moment_digest']==m.g.optimizer_moment_digest(old['optimizer']);assert last['parent_scheduler']==old['scheduler'] and last['scheduler']==dict(type='constant_after_cosine100',last_epoch=120,epochs_after_parent=20,lr=cfg['minimum_lr']);assert last['initial_digest']==old['initial_digest']==s.read(m.BASE/'L6_rho03/fit_result.json')['initial_digest'];assert len(last['optimizer']['state'])==30 and all(float(q['step'])==23640 for q in last['optimizer']['state'].values()) and np.allclose([q['lr'] for q in last['optimizer']['param_groups']],cfg['minimum_lr'],atol=1e-12,rtol=0) and [q['weight_decay'] for q in last['optimizer']['param_groups']]==[0,cfg['electronic_weight_decay']]
        model=m.RegularizedMoE(6,.3);assert best['contract']==model.contract();model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True);assert all(.7<a<=.95 for a in model.alphas());assert sorted(last['kd_student_probabilities'])==list(range(1,121))
        for e in range(1,101):assert s.digest(last['kd_student_probabilities'][e])==s.digest(old['kd_student_probabilities'][e])
        info=m.k.teacher_info();assert fm['teacher']==last['teacher']==info;teacher=torch.load(info['best'],map_location='cpu',weights_only=False);previous.verify_predictions(Path(info['train']),teacher['model'],'train',manifest)
        with np.load(info['train'],allow_pickle=False) as z:probs=z['probabilities'].copy();labels=z['labels'].copy()
        with np.load(cfg['data'],allow_pickle=False) as z:images=z['train_images'].copy()
        n=len(labels)
        for row in h[100:]:
            epoch=row['epoch'];assert row['adam_steps']==197*epoch and row['effective_KD_alpha']==cfg['kd_alpha'] and np.allclose(row['lr'],cfg['minimum_lr'],atol=1e-12,rtol=0);assert tuple(row[q] for q in ('order_sha256','affine_sha256','mix_sha256'))==previous.plans(epoch,cfg,n) and tuple(row[q] for q in ('flip_sha256','mask_sha256'))==previous.mask_plans(epoch,cfg,n);order=np.random.default_rng(np.random.SeedSequence([17,epoch,20261013])).permutation(n).astype('<i8');views=view_hashes(images,order,epoch,cfg)
            for kind,value in views.items():assert row[kind+'_sha256']==value,(epoch,kind)
            student=last['kd_student_probabilities'][epoch].numpy().astype(np.float64);assert student.shape==(n,20) and np.isfinite(student).all() and (student>=0).all() and np.allclose(student.sum(1),1,atol=2e-6);record={q:[] for q in ('teacher_target','teacher_gate')};count=0;support=np.zeros(20,np.int64);kl=0.
            for batch,off in enumerate(range(0,n,16)):
                idx=order[off:off+16];q,eligible=reference_kd(probs,labels,idx,epoch,batch,cfg);record['teacher_target'].append(hashlib.sha256(q.tobytes()).hexdigest());record['teacher_gate'].append(hashlib.sha256(eligible.tobytes()).hexdigest());count+=int(eligible.sum());support+=np.bincount(labels[idx],weights=eligible,minlength=20).astype(np.int64);log=np.log(np.maximum(student[off:off+len(idx)],1e-12))/cfg['kd_temperature'];temp=np.exp(log-log.max(1,keepdims=True));temp/=temp.sum(1,keepdims=True);kl+=float(cfg['kd_temperature']**2*np.sum((q.astype(np.float64)*(np.log(np.maximum(q,1e-12))-np.log(np.maximum(temp,1e-12)))).sum(1)*eligible))
            for kind,values in record.items():assert row[kind+'_sha256']==hashlib.sha256(''.join(values).encode()).hexdigest()
            assert count==row['kd_eligible_count'] and support.tolist()==row['kd_eligible_support'] and abs(kl-row['kd_KL_T2_sum'])<1e-6
            for role in ('train','val'):previous.history_metric(row[role],manifest['supports'][role])
        for name,value in last['previous_final_ema'].items():assert np.allclose(value.numpy().astype(np.float64)*.999+last['model'][name].numpy().astype(np.float64)*.001,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6),name
        for state_key,field in [('model','raw_phase_smoothness'),('ema','EMA_phase_smoothness')]:
            value,_=independent_TV(last[state_key],1.);assert abs(value-h[-1][field])<1e-6;checks[field]=value
        selected=sorted(h,key=lambda q:(-q['val']['accuracy'],q['val']['nll'],q['epoch']))[0];assert selected['epoch']==f['selected_epoch']==best['epoch'] and s.digest(best['model'])==selected['ema_digest'] and s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest'];assert best['sources']==(m.fresh.sources() if selected['epoch']<=100 else m.sources())
        for role in ('train','val'):
            actual,checks[role]=previous.verify_predictions(Path(f['selected_'+role+'_path']),best['model'],role,manifest);previous.agree(actual,f['metrics'][role]);previous.agree(actual,selected[role]);previous.agree(actual,best['metrics'][role])
        gate=s.read(base/'fit/numeric_gate.json');assert gate['passed'] and gate['initial_digest']==s.digest(old['model']) and gate['optimizer_steps']==0 and len(gate['checks'])==6
        for q in gate['checks']:assert q['maxabs']<=2e-6 and q['gradient_relative_l2']<=.02 and q['gradient_cosine']>=.999 and set(q['gradient_norms'])==set(dict(model.named_parameters())) and all(np.isfinite(z) and z>0 for z in q['gradient_norms'].values())
        checks.update(exact_parent100_prefix=True,additional_epochs=20,new_TRAIN_pixels_replayed=20*n,optimizer_updates=23640,new_EMA_decay=.999)
    else:
        m.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['model_digest']==started['model_digest']==s.digest(best['model']) and t['best_sha256']==started['best_sha256']==f['best_sha256'] and t['selected_epoch']==started['selected_epoch']==f['selected_epoch'] and started['sources']==m.sources() and t['test_used_for_epoch_selection'] is False and t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=previous.verify_predictions(Path(t['predictions_path']),best['model'],'test',manifest);previous.agree(actual,t['metrics'])
        if t['reused']:
            registry=m.known_registry(case);assert t['new_test_evaluations']==0 and t['model_digest'] in registry;known=registry[t['model_digest']];assert t['predictions_path']==known['predictions_path'] and t['predictions_sha256']==known['predictions_sha256']
        else:
            assert t['new_test_evaluations']==1 and t['model_digest'] not in p['test_registry'] and (base/('digest_'+t['model_digest']+'.claim.json')).exists();owner=s.read(m.HERE/'runs/simulation/caltech20_rich20_EMA999_shared_digest_claims_20261017'/(t['model_digest']+'.json'));assert owner['owner_root']==str(base) and owner['model_digest']==t['model_digest'] and owner['sources']==m.sources()
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,run=str(base),sources=m.sources(),training_commit=meta['git_commit'],config=cfg,parent=p,checks=checks,optical_model_forwards=0,TEST_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
