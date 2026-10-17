"""Own saved-source/parent/new-optimizer/TRAIN-view/KD/EMA/prediction audit."""
import argparse,hashlib,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_selfview2 as m
import audit_caltech20_alt246_richsplit as independent
from audit_caltech20_alt246_richsplit_cachedkd import reference_kd
from audit_caltech20_alt246_richsplit_viewreg import view_hashes
def audit(base,out,phase):
    assert base==m.OUT and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();m.cpu_ready();s=m.s;p=m.parent();cfg=m.config();meta=s.read(base/'metadata.json');assert meta['sources']==m.sources() and meta['config']==cfg and meta['parent']==p;m.engine.verify_sources(meta['sources'],meta['git_commit']);f=s.read(base/'fit/fit_result.json');best=torch.load(f['best_path'],map_location='cpu',weights_only=False);assert f['best_sha256']==s.sha(f['best_path']);ex=s.read(base/(phase+'_exit.json'));proc=Path('/proc',str(ex['pid']),'cmdline');assert ex['returncode']==0 and ex['actual_exit'] and (not proc.exists() or str(base/phase).encode() not in proc.read_bytes());manifest=m.b.dataset.manifest();checks={}
    if phase=='fit':
        old=torch.load(p['best'],map_location='cpu',weights_only=False);last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);h=s.read(base/'fit/history.json');fm=s.read(base/'fit/metadata.json');assert len(h)==21 and h[0]['tune_epoch']==0 and h[0]['train']==p['metrics']['train'] and h[0]['val']==p['metrics']['val'];assert h[0]['EMA_digest']==fm['initial_digest']==s.digest(old['model']) and fm['initial_optimizer_states']==0 and fm['optimizer_restored'] is False and fm['teacher_forward_count']==0;assert fm['sources']==last['sources']==m.sources() and fm['config']==last['config']==cfg and fm['parent']==last['parent']==p;assert f['actual_optical_epochs_spent']==120 and f['additional_epochs']==20 and f['new_adam_steps']==last['new_steps']==3940 and f['optimizer_restored'] is False and last['tune_epoch']==20 and f['parent_selected_optical_epoch']==86 and f['inherited_head_updates']==40 and f['inherited_selected_head_step']==27 and f['stop_reason']=='registered20_epoch_selfview_review';assert f['last_sha256']==s.sha(base/'fit/last.pt') and f['history_sha256']==s.sha(base/'fit/history.json') and f['fit_seconds']<=3600+max(a['epoch_seconds'] for a in h[1:])+60;opt=last['optimizer'];assert len(opt['state'])==12 and all(float(a['step'])==3940 for a in opt['state'].values());assert np.allclose([a['lr'] for a in opt['param_groups']],cfg['lr'],atol=1e-12,rtol=0) and [a['weight_decay'] for a in opt['param_groups']]==[0,cfg['electronic_weight_decay']]
        model=m.RegularizedMoE(2,.3);assert model.contract()==old['contract']==best['contract'];model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True);assert len(dict(model.named_parameters()))==12 and all(.7<a<=.95 for a in model.alphas());assert sorted(last['student_probabilities'])==list(range(1,21));independent.verify_predictions(Path(p['train']),old['model'],'train',manifest)
        with np.load(p['train'],allow_pickle=False) as z:probs=z['probabilities'].copy();labels=z['labels'].copy()
        with np.load(cfg['data'],allow_pickle=False) as z:images=z['train_images'].copy()
        n=len(labels)
        for tune,row in enumerate(h[1:],1):
            epoch=100+tune;assert row['tune_epoch']==tune and row['augmentation_epoch']==epoch and row['adam_steps']==197*tune and np.allclose(row['lr'],cfg['lr'],atol=1e-12,rtol=0);assert tuple(row[a] for a in ('order_sha256','affine_sha256','mix_sha256'))==independent.plans(epoch,cfg,n);assert tuple(row[a] for a in ('flip_sha256','mask_sha256'))==independent.mask_plans(epoch,cfg,n);order=np.random.default_rng(np.random.SeedSequence([17,epoch,20261013])).permutation(n).astype('<i8');views=view_hashes(images,order,epoch,cfg)
            for name,value in views.items():assert row[name+'_sha256']==value
            student=last['student_probabilities'][tune].numpy().astype(np.float64);assert student.shape==(n,20) and np.isfinite(student).all() and (student>=0).all() and np.allclose(student.sum(1),1,atol=2e-6);record={a:[] for a in ('teacher_target','teacher_gate')};count=0;support=np.zeros(20,np.int64);kl=0.
            for batch,off in enumerate(range(0,n,16)):
                idx=order[off:off+16];q,valid=reference_kd(probs,labels,idx,epoch,batch,cfg);record['teacher_target'].append(hashlib.sha256(q.tobytes()).hexdigest());record['teacher_gate'].append(hashlib.sha256(valid.tobytes()).hexdigest());count+=int(valid.sum());support+=np.bincount(labels[idx],weights=valid,minlength=20).astype(np.int64);z=np.log(np.maximum(student[off:off+len(idx)],1e-12))/2;T=np.exp(z-z.max(1,keepdims=True));T/=T.sum(1,keepdims=True);kl+=float(4*np.sum((q*(np.log(np.maximum(q,1e-12))-np.log(np.maximum(T,1e-12)))).sum(1)*valid))
            for name,values in record.items():assert row[name+'_sha256']==hashlib.sha256(''.join(values).encode()).hexdigest()
            assert count==row['kd_eligible_count'] and support.tolist()==row['kd_eligible_support'] and abs(kl-row['kd_KL_T2_sum'])<1e-6
            for role in ('train','val'):independent.history_metric(row[role],manifest['supports'][role])
        for name,value in last['previous_final_ema'].items():assert np.allclose(value.numpy().astype(np.float64)*.999+last['model'][name].numpy().astype(np.float64)*.001,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6),name
        selected=min(h,key=lambda a:(-a['val']['accuracy'],a['val']['nll'],a['tune_epoch']));assert selected['tune_epoch']==f['selected_tune_epoch'] and s.digest(best['model'])==selected['EMA_digest'] and s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['EMA_digest'];assert best['sources']==(p['parent_sources'] if selected['tune_epoch']==0 else m.sources())
        for role in ('train','val'):
            actual,checks[role]=independent.verify_predictions(Path(f['selected_'+role+'_path']),best['model'],role,manifest);independent.agree(actual,f['metrics'][role]);independent.agree(actual,selected[role]);independent.agree(actual,best['metrics'][role])
        gate=s.read(base/'fit/numeric_gate.json');assert gate['passed'] and gate['optimizer_steps']==0 and gate['initial_digest']==s.digest(old['model']) and len(gate['checks'])==6
        for a in gate['checks']:assert a['maxabs']<=2e-6 and a['gradient_relative_l2']<=.02 and a['gradient_cosine']>=.999 and set(a['gradient_norms'])==set(dict(model.named_parameters())) and all(np.isfinite(z) and z>0 for z in a['gradient_norms'].values())
        checks.update(new_epochs=20,parent_selected_optical_epoch=86,inherited_head_step=27,new_optimizer=True,new_Adam_steps=3940,independently_replayed_TRAIN_images=20*n,teacher_forwards=0)
    else:
        m.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');start=s.read(base/'test/started.json');assert t['best_sha256']==start['best_sha256']==f['best_sha256'] and t['model_digest']==start['model_digest']==s.digest(best['model']) and t['selected_tune_epoch']==start['selected_tune_epoch']==f['selected_tune_epoch'] and start['sources']==m.sources() and t['test_used_for_epoch_selection'] is False and t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=independent.verify_predictions(Path(t['predictions_path']),best['model'],'test',manifest);independent.agree(actual,t['metrics'])
        if t['reused']:assert t['new_test_evaluations']==0 and t['model_digest'] in p['test_registry'] and t['predictions_path']==p['test_registry'][t['model_digest']]['predictions_path']
        else:assert t['new_test_evaluations']==1 and t['model_digest'] not in p['test_registry']
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=m.sources(),training_commit=meta['git_commit'],config=cfg,parent=p,checks=checks,optical_forwards=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
