"""Independent saved math and per-macro SAM/input-mask evidence; no forward/test."""
import argparse,hashlib,inspect,math,os,textwrap
from pathlib import Path
import numpy as np
import torch
import caltech10_error_distill as study
import audit_resisc20 as sealed
from audit_resisc10 import predictions
from audit_resisc20_input_pair import scheduler_budget_checks
s=study.s

def independent_verifier(case):
    depth=case['depth'];phase=479364 if depth==2 else 908552;tensors=6 if depth==2 else 11
    mix="(.2 if epoch<=100 else (.3 if epoch<=325 else .1))" if depth==2 else "(.2 if epoch<=100 else (.3 if epoch<=575 else .4))"
    src=textwrap.dedent(inspect.getsource(sealed.verify));changes={"<=30":"==cfg['maximum_epochs']","s.r.CorrectedMoE(4,f['rho'])":"study.parent.SubsetMoE(f['depth'],f['rho'])","==908552":f'=={phase}',"len(last['optimizer']['state'])==11":f"len(last['optimizer']['state'])=={tensors}","labels=z[role+'_labels'].copy()":"labels=study.parent.subset_labels(z[role+'_labels'].copy())",'s.eligibility(f,hist)':'study.parent.eligibility(f,hist)',"cfg['mixup_alpha'],cfg['mixup_alpha']":mix+','+mix}
    for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
    original="lam=float(draw.beta("+mix+','+mix+"));lam=max(lam,1-lam);permutation=draw.permutation(count)"
    assert src.count(original)==1
    replacement=f"lam=1. if epoch>1090 else float(draw.beta("+mix+','+mix+f"));lam=max(lam,1-lam);permutation=np.arange(count) if epoch>1090 else draw.permutation(count)"
    src=src.replace(original,replacement)
    ns=dict(sealed.__dict__,study=study,predictions=predictions);exec(compile(src,__file__+':independent_verify','exec'),ns);return ns['verify']

def independent_scheduler(case):
    src=textwrap.dedent(inspect.getsource(scheduler_budget_checks));old='for i,h in enumerate(hist):';assert src.count(old)==1
    prior=(326,.0005) if case['depth']==2 else (576,.0001);second=(726,.0003) if case['depth']==2 else (651,.0001);third=(951,.0003) if case['depth']==2 else (726,.0001);fourth=(1041,.0003);fifth=(1091,.0003)
    inject=f"\n        if h['epoch']=={prior[0]}:rate={prior[1]};best=float('inf');bad=0\n        if h['epoch']=={second[0]}:rate={second[1]};best=float('inf');bad=0\n        if h['epoch']=={third[0]}:rate={third[1]};best=float('inf');bad=0\n        if h['epoch']=={fourth[0]}:rate={fourth[1]};best=float('inf');bad=0\n        if h['epoch']=={fifth[0]}:rate={fifth[1]};best=float('inf');bad=0\n        if h['epoch']=={case['parent_epoch']+1}:rate={case['restart_lr']};best=float('inf');bad=0"
    src=src.replace(old,old+inject);ns=dict(scheduler_budget_checks.__globals__);exec(compile(src,__file__+':LR_replay','exec'),ns);return ns['scheduler_budget_checks']

def verify(root,case):
    c=study.CASES[case];depth=c['depth'];cfg=study.config(case);m=study.parent.manifest(Path(cfg['data']));fit=independent_verifier(c)(root,case,Path(cfg['data']),m);assert cfg['mixup_alpha']==cfg['label_smoothing']==0 and fit['config']==cfg and fit['sources']==study.sources() and fit['stop_reason']=='maximum_epochs';arm=root/case;par=study.PARENT;prefix=s.read(par/'history.json');hist=s.read(arm/'history.json');grads=s.read(arm/'gradients.json');assert len(hist)==len(prefix)+10==study.TARGET and hist[:len(prefix)]==prefix and grads[:len(prefix)]==s.read(par/'gradients.json');old=torch.load(par/'last_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);receipt=s.read(arm/'resume_receipt.json');first=len(prefix)==c['parent_epoch']
    assert receipt['last_checkpoint_sha256']==s.sha(par/'last_checkpoint.pt') and receipt['prefix_history_sha256']==s.sha(par/'history.json') and receipt['parent_epoch']==len(prefix) and receipt['first_new_epoch']==len(prefix)+1 and receipt['restored_state_sha256']=={k:s.digest(old[k]) for k in ('model','ema')}
    assert receipt['optimizer_reset']==receipt['ema_reset']==False and receipt['adam_moments_preserved'] and receipt['scheduler_reset']==first and receipt['scheduler_restart_epoch']==(c['parent_epoch']+1 if first else None) and receipt['scheduler_restart_lr']==(c['restart_lr'] if first else None);gate=s.read(arm/'smoke.json');assert gate['initial_parameter_sha256']==s.digest(old['model']) and gate['model_state_preserved'] and len(gate['method_checks'])==4
    if first:done=s.read(par.parent/'audit_completed.json');assert done['audit_sha256']==c['parent_audit_sha256']==s.sha(par.parent/'science/audit.json') and s.sha(par/'last_checkpoint.pt')==c['parent_last_sha256'] and s.sha(par/'history.json')==c['parent_history_sha256']
    else:done=s.read(par.parent/'audit_completed.json');assert done['audit_sha256']==s.sha(par.parent/'science/audit.json')
    assert done['passed'] and done['actual_audit_returncode']==0
    previous=prefix[-1]['seconds'];offset=previous;steps=0;passes=0
    train_data=study.parent.load_fit(Path(cfg['data']),'train',m)
    for row in hist[len(prefix):]:
        assert previous-offset<1800 and row['seconds']>=previous;previous=row['seconds'];record=s.read(arm/('method_epoch%04d.json'%row['epoch']));assert record['epoch']==row['epoch'] and len(record['records'])==48
        for batch,v in enumerate(record['records']):
            assert v['epoch']==row['epoch'] and v['batch']==batch and v['shape'][0]==(9 if batch==47 else 16) and v['exact_restore'] and v['dropout_probability']==c['input_dropout'] and v['seed']==17+row['epoch']*1000003+batch*97
            assert v['sam']==(c['sam_radius']>0) and v['gradient_passes']==(2 if c['sam_radius'] else 1)
            if c['sam_radius']:assert .95*c['sam_radius']<=v['perturbation_norm']<=1.05*c['sam_radius'] and math.isfinite(v['first_grad_norm']) and v['first_grad_norm']>0 and math.isfinite(v['second_loss'])
            if c['input_dropout']:
                b,h,w=v['shape'];generator=torch.Generator().manual_seed(v['seed']);coarse=torch.rand((b,math.ceil(h/16),math.ceil(w/16)),generator=generator)>=c['input_dropout'];mask=coarse.repeat_interleave(16,1).repeat_interleave(16,2)[:,:h,:w];assert hashlib.sha256(mask.numpy().tobytes()).hexdigest()==v['mask_sha256'] and float((~mask).float().mean())==v['zero_fraction']
            assert v['focal_gamma']==c['focal_gamma'] and v['color_strength']==c['color_strength'] and v['color_seed']==17+row['epoch']*1000003+batch*97+73
            order,_=s.engine.epoch_plan(row['epoch'],sum(m['supports']['train']),cfg);idx=order[batch*16:batch*16+v['shape'][0]];assert hashlib.sha256(np.asarray(idx,dtype='<i8').tobytes()).hexdigest()==v['source_indices_sha256']
            if c['color_strength']:
                image=train_data[0][idx];generator=torch.Generator().manual_seed(v['color_seed']);strength=c['color_strength'];gains=1-strength+2*strength*torch.rand((len(idx),3,1,1),generator=generator);contrast=1-strength+2*strength*torch.rand((len(idx),1,1,1),generator=generator);z=image.float()/255.;mean=z.mean(dim=(-2,-1),keepdim=True);colored=(((z-mean)*contrast+mean)*gains).clamp(0,1).mul(255).round().to(torch.uint8);assert hashlib.sha256(colored.numpy().tobytes()).hexdigest()==v['color_images_sha256']
            else:assert v['color_images_sha256'] is None
            assert v['error_teacher_multiplier']==c['error_teacher_multiplier'] and len(v['teacher_evidence'])==math.ceil(v['shape'][0]/8)
            active=0;focused=0;weighted=0.;kl_sum=0.;evidence_offset=0
            for ev in v['teacher_evidence']:
                tp=np.asarray(ev['teacher_probs'],np.float64);sp=np.asarray(ev['student_probs'],np.float64);labels=np.asarray(ev['labels'],np.int64);count=min(8,v['shape'][0]-evidence_offset);assert tp.shape==sp.shape==(count,10) and np.isfinite(tp).all() and np.isfinite(sp).all() and (tp>=0).all() and (sp>=0).all() and np.allclose(tp.sum(1),1,atol=1e-5,rtol=0) and np.allclose(sp.sum(1),1,atol=1e-5,rtol=0) and np.array_equal(labels,train_data[1][idx[evidence_offset:evidence_offset+count]].numpy());gate=(tp.argmax(1)==labels)&(tp.max(1)>=.5);wrong=sp.argmax(1)!=labels;wg=gate*(1+(c['error_teacher_multiplier']-1)*wrong);soft=np.sqrt(tp.clip(1e-12));soft/=soft.sum(1,keepdims=True);student=np.sqrt(sp.clip(1e-12));student/=student.sum(1,keepdims=True);kl=(soft*(np.log(soft.clip(1e-12))-np.log(student.clip(1e-12)))).sum(1)*4;active+=int(gate.sum());focused+=int((gate&wrong).sum());weighted+=float(wg.sum());kl_sum+=float((kl*wg).sum());evidence_offset+=count
            assert evidence_offset==v['shape'][0] and active==v['teacher_active_samples'] and focused==v['teacher_error_samples'] and weighted==v['teacher_weighted_active_samples'] and math.isclose(kl_sum/v['shape'][0],v['teacher_kl'],rel_tol=1e-5,abs_tol=1e-5)
            assert v['teacher_alpha']==c['teacher_alpha']
            if c['teacher_alpha']:assert v['teacher_batches']==math.ceil(v['shape'][0]/8) and 0<=v['teacher_active_samples']<=v['shape'][0] and math.isfinite(v['teacher_kl']) and v['teacher_kl']>=-1e-6
            else:assert v['teacher_batches']==v['teacher_active_samples']==0
            steps+=1;passes+=v['gradient_passes']
    assert steps==480 and passes==(960 if c['sam_radius'] else 480)
    for row in hist:
        for role in ('train','val'):
            if role not in row:continue
            metric=row[role];cm=np.asarray(metric['confusion_matrix']);support=np.asarray(m['supports'][role]);assert cm.shape==(10,10) and np.issubdtype(cm.dtype,np.integer) and (cm>=0).all() and np.array_equal(cm.sum(1),support) and np.allclose(metric['recall'],cm.diagonal()/support,rtol=0,atol=1e-12) and math.isclose(metric['accuracy'],cm.trace()/support.sum(),abs_tol=1e-12) and math.isclose(metric['balanced_accuracy'],(cm.diagonal()/support).mean(),abs_tol=1e-12)
    budget=independent_scheduler(c)(dict(cfg,maximum_seconds=float('inf'),minimum_epochs=float('inf')),hist,last,'maximum_epochs');assert last['scheduler']['last_epoch']==len(hist) and all(int(v['step'])==48*len(hist) for v in last['optimizer']['state'].values());return fit,budget,dict(Adam_updates=steps,gradient_passes=passes,method_records_verified=True,deterministic_dropout_replayed=True,color_bytes_independently_replayed=True,focal_gamma_verified=True,error_focus_and_KL_independently_recomputed=True,SAM_exact_restore_records=True)

def audit(a):
    torch.set_num_threads(4)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==a.root/'science' and not a.out.exists();meta=s.read(a.root/'metadata.json');study.TARGET=meta['target_epoch'];study.PARENT=Path(meta['parent']);assert meta['case']==a.case and meta['sources']==study.sources();s.engine.verify_sources(meta['sources'],meta['git_commit']);claim=s.read(a.root/'audit_claim.json');assert claim['sources']==meta['sources'] and claim['case']==a.case;rec=s.read(a.root/a.case/'producer_exit.json');assert rec['returncode']==0 and rec['actual_exit'] and study.parent.base.exited(rec['pid']) and not (a.root/a.case/'failure.json').exists();study.bind(a.case);fit,budget,method=verify(a.root,a.case);a.out.mkdir();s.save(a.out/'audit.json',dict(passed=True,sources=meta['sources'],fit=fit,budget=budget,method=method,teacher_checkpoint_sha256=study.P['teacher_checkpoint_sha256'],teacher_model_digest=study.P['teacher_model_digest'],model_forward_calls=0,test_image_arrays_read=0,test_informed_development=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        mark=a.root/'cpu_report_failure.json'
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
