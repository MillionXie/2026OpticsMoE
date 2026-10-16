"""Independent saved-math and complete new TRAIN pixel replay; zero forwards."""
import argparse,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import caltech20_alt246_richsplit_cachedkd as study
import audit_caltech20_alt246_richsplit as previous








def reference_kd(probs,labels,indices,epoch,batch,cfg):
    source=probs[indices];soft=np.power(np.maximum(source,1e-12),1/cfg['kd_temperature']);soft/=np.sum(soft,axis=1,keepdims=True);valid=np.array([int(np.argmax(row))==int(label) and float(np.max(row))>=cfg['teacher_confidence'] for row,label in zip(source,labels[indices])]);rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=lam if lam>=.5 else 1-lam;permutation=rng.permutation(len(indices));return np.asarray(lam*soft+(1-lam)*soft[permutation],np.float32),np.logical_and(valid,valid[permutation])

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();meta=s.read(base/'metadata.json');depth=meta['depth'];parent_root=Path(meta['parent_root']);assert base==study.expected_root(depth,parent_root) and meta['sources']==study.sources() and meta['config']==study.config();study.engine.verify_sources(meta['sources'],meta['git_commit']);parent=study.parent_info(parent_root,depth);assert parent==meta['parent'];cfg=study.config();m=study.b.dataset.manifest();torch.set_num_threads(4);checks={};f=s.read(base/'fit/fit_result.json')
    ex=s.read(base/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(ex['pid']),'cmdline');assert not proc.exists() or str(base/phase).encode() not in proc.read_bytes();best=torch.load(f['best_path'],map_location='cpu',weights_only=False);assert f['best_sha256']==s.sha(f['best_path'])
    if phase=='fit':
        metadata=s.read(base/'fit/metadata.json');assert metadata['sources']==meta['sources'] and metadata['config']==cfg and metadata['parent']==parent and metadata['versions']==study.versions()
        assert metadata['parent_last_sha256']==s.sha(parent['last']) and metadata['parent_history_sha256']==s.sha(parent['history_path'])
        old=torch.load(parent['last'],map_location='cpu',weights_only=False);last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);h=s.read(base/'fit/history.json');prefix=s.read(parent['history_path']);start=parent['actual_epochs'];end=start+20
        assert h[:start]==prefix and len(h)==end and f['actual_epochs']==end and f['additional_epochs']==end-100 and f['block_epochs']==20 and f['steps']==last['steps']==end*cfg['steps_per_epoch'] and f['stop_reason']=='registered20_epoch_method_review'
        assert f['fit_seconds']<=cfg['maximum_seconds']+max(q['epoch_seconds'] for q in h[start:])+60
        assert f['last_sha256']==s.sha(base/'fit/last.pt') and f['history_sha256']==s.sha(base/'fit/history.json') and last['epoch']==end and last['config']==cfg and last['sources']==study.sources()
        assert metadata['parent_raw_digest']==s.digest(old['model']) and metadata['parent_ema_digest']==s.digest(old['ema']) and metadata['initial_optimizer_moment_digest']==study.g.optimizer_moment_digest(old['optimizer'])
        model=study.RegularizedMoE(depth);assert best['contract']==model.contract();model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True)
        optimizer=last['optimizer'];assert len(optimizer['state'])==len(list(model.parameters())) and all(float(v['step'])==last['steps'] for v in optimizer['state'].values()) and [v['weight_decay'] for v in optimizer['param_groups']]==[0,cfg['electronic_weight_decay']]
        dummy=[torch.nn.Parameter(torch.zeros(())),torch.nn.Parameter(torch.zeros(()))];opt=torch.optim.AdamW([dict(params=[dummy[0]],weight_decay=0),dict(params=[dummy[1]],weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,threshold=.002,threshold_mode='abs',min_lr=cfg['minimum_lr']);baseline=study.s.read(study.BASE/f'L{depth}_rho03/fit_result.json');scheduler.best=baseline['metrics']['val']['balanced_nll']
        info=study.teacher_info();assert metadata['teacher']==last['teacher']==info;teacher=torch.load(info['best'],map_location='cpu',weights_only=False);previous.verify_predictions(Path(info['train']),teacher['model'],'train',m)
        with np.load(info['train'],allow_pickle=False) as z:teacher_probs=z['probabilities'].copy();teacher_labels=z['labels'].copy()
        n=len(teacher_labels);assert sorted(last['kd_student_probabilities'])==list(range(101,121))
        for row in h[100:]:
            assert np.allclose(row['lr'],[q['lr'] for q in opt.param_groups],atol=1e-12,rtol=0);scheduler.step(row['val']['balanced_nll'])
        for name in ('best','num_bad_epochs','cooldown_counter','last_epoch','_last_lr'):assert np.allclose(last['scheduler'][name],scheduler.state_dict()[name],atol=1e-12,rtol=0),name
        for row in h[start:]:
            epoch=row['epoch'];assert row['adam_steps']==cfg['steps_per_epoch']*epoch and tuple(row[k] for k in ('order_sha256','affine_sha256','mix_sha256'))==previous.plans(epoch,cfg,n) and tuple(row[k] for k in ('flip_sha256','mask_sha256'))==previous.mask_plans(epoch,cfg,n)
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(n).astype('<i8');record={'teacher_target':[],'teacher_gate':[]};count=0;support=np.zeros(20,np.int64);kl_sum=0.;student=last['kd_student_probabilities'][epoch].numpy().astype(np.float64);assert student.shape==(n,20) and np.isfinite(student).all() and (student>=0).all() and np.allclose(student.sum(1),1,atol=2e-6)
            for batch,off in enumerate(range(0,n,16)):
                idx=order[off:off+16];q,eligible=reference_kd(teacher_probs,teacher_labels,idx,epoch,batch,cfg);record['teacher_target'].append(hashlib.sha256(q.tobytes()).hexdigest());record['teacher_gate'].append(hashlib.sha256(eligible.tobytes()).hexdigest());count+=int(eligible.sum());support+=np.bincount(teacher_labels[idx],weights=eligible,minlength=20).astype(np.int64);p=student[off:off+len(idx)];z=np.log(np.maximum(p,1e-12))/cfg['kd_temperature'];temp=np.exp(z-z.max(1,keepdims=True));temp/=temp.sum(1,keepdims=True);kl_sum+=float(cfg['kd_temperature']**2*np.sum((q.astype(np.float64)*(np.log(np.maximum(q,1e-12))-np.log(np.maximum(temp,1e-12)))).sum(1)*eligible))
            for kind,values in record.items():assert row[kind+'_sha256']==hashlib.sha256(''.join(values).encode()).hexdigest()
            assert count==row['kd_eligible_count'] and support.tolist()==row['kd_eligible_support'] and abs(kl_sum-row['kd_KL_T2_sum'])<1e-6;checks['KD_epoch'+str(epoch)]=dict(eligible=count,eligible_support=support.tolist(),KL_T2_sum=kl_sum)
            for role in ('train','val'):previous.history_metric(row[role],m['supports'][role])
        assert set(last['previous_final_ema'])==set(last['model'])==set(last['ema'])
        for name,earlier in last['previous_final_ema'].items():
            expected=earlier.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][name].numpy().astype(np.float64)*(1-cfg['ema_decay']);assert np.allclose(expected,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6),name
        selected=min(h,key=lambda q:q['val']['balanced_nll']);assert selected['epoch']==f['selected_epoch']==best['epoch'] and s.digest(best['model'])==selected['ema_digest'] and s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest']
        assert best['sources']==(study.b.sources() if selected['epoch']<=100 else study.sources())
        for role in ('train','val'):
            actual,checks[role]=previous.verify_predictions(Path(f['selected_'+role+'_path']),best['model'],role,m);previous.agree(actual,f['metrics'][role]);previous.agree(actual,selected[role]);previous.agree(actual,best['metrics'][role])
        gate=s.read(base/'fit/numeric_gate.json');assert gate['passed'] and gate['initial_digest']==s.digest(old['model']) and gate['optimizer_steps']==0 and len(gate['checks'])==6
        for c in gate['checks']:assert c['maxabs']<=2e-6 and c['gradient_relative_l2']<=.02 and c['gradient_cosine']>=.999 and set(c['gradient_norms'])==set(dict(model.named_parameters())) and all(np.isfinite(v) and v>0 for v in c['gradient_norms'].values())
        checks['cached_teacher_targets_replayed']=20*n;checks['new_epochs']=20;checks['parameter_count']=sum(v.numel() for v in model.parameters())
    else:
        study.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['selected_epoch']==started['selected_epoch']==f['selected_epoch'] and t['model_digest']==started['model_digest']==s.digest(best['model']) and t['best_sha256']==started['best_sha256']==f['best_sha256'] and started['sources']==study.sources() and t['test_used_for_epoch_selection'] is False
        assert t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=previous.verify_predictions(Path(t['predictions_path']),best['model'],'test',m);previous.agree(actual,t['metrics'])
        if t['reused']:
            assert t['new_test_evaluations']==0 and t['model_digest'] in parent['test_registry'];known=parent['test_registry'][t['model_digest']];assert t['predictions_path']==known['predictions_path'] and t['predictions_sha256']==known['predictions_sha256']
        else:assert t['new_test_evaluations']==1 and t['model_digest'] not in parent['test_registry'] and (base/('digest_'+t['model_digest']+'.claim.json')).exists()
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,run=str(base),sources=study.sources(),training_commit=meta['git_commit'],config=cfg,checks=checks,model_forwards=0,test_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
