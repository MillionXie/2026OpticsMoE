"""Independent saved math with one documented phase326 LR/statistics restart."""
import argparse,copy,hashlib,inspect,math,os,textwrap
from pathlib import Path
import numpy as np
import torch
import caltech10_l2_refit as study
import audit_resisc20 as sealed
from audit_resisc10 import predictions
from audit_resisc20_input_pair import scheduler_budget_checks
s=study.s
def independent_verifier():
    src=textwrap.dedent(inspect.getsource(sealed.verify));changes={"<=30":"==cfg['maximum_epochs']","s.r.CorrectedMoE(4,f['rho'])":"study.parent.SubsetMoE(f['depth'],f['rho'])","==908552":"==479364","len(last['optimizer']['state'])==11":"len(last['optimizer']['state'])==6","labels=z[role+'_labels'].copy()":"labels=study.parent.subset_labels(z[role+'_labels'].copy())",'s.eligibility(f,hist)':'study.parent.eligibility(f,hist)',"cfg['mixup_alpha'],cfg['mixup_alpha']":"(.2 if epoch<=100 else (.3 if epoch<=325 else .1)),(.2 if epoch<=100 else (.3 if epoch<=325 else .1))"}
    for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
    ns=dict(sealed.__dict__,study=study,predictions=predictions);exec(compile(src,__file__+':independent_L2_verify','exec'),ns);return ns['verify']
def independent_scheduler():
    src=textwrap.dedent(inspect.getsource(scheduler_budget_checks));old='for i,h in enumerate(hist):';assert src.count(old)==1;src=src.replace(old,old+"\n        if h['epoch']==326:rate=.0005;best=float('inf');bad=0");ns=dict(scheduler_budget_checks.__globals__);exec(compile(src,__file__+':independent_restarted_LR','exec'),ns);return ns['scheduler_budget_checks']
def verify(root):
    case='L2_rho03';cfg=study.config(case);m=study.parent.manifest(Path(cfg['data']));fit=independent_verifier()(root,case,Path(cfg['data']),m);cfg.update(depth=2,residual_rho=.3,classes=m['classes'],contract=study.parent.contract(2,.3));assert fit['config']==cfg and fit['sources']==study.sources() and fit['stop_reason']=='maximum_epochs';arm=root/case;hist=s.read(arm/'history.json');grads=s.read(arm/'gradients.json');par=study.PARENT;prefix=s.read(par/'history.json');oldgrad=s.read(par/'gradients.json');ck=torch.load(par/'last_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);receipt=s.read(arm/'resume_receipt.json');first=len(prefix)==325
    assert len(hist)==len(prefix)+25==study.TARGET and hist[:len(prefix)]==prefix and grads[:len(prefix)]==oldgrad
    assert receipt['parent_epoch']==len(prefix) and receipt['first_new_epoch']==len(prefix)+1 and receipt['last_checkpoint_sha256']==s.sha(par/'last_checkpoint.pt') and receipt['prefix_history_sha256']==s.sha(par/'history.json') and receipt['restored_state_sha256']=={k:s.digest(ck[k]) for k in ('model','ema')}
    assert receipt['optimizer_reset']==receipt['ema_reset']==False and receipt['scheduler_reset']==first and receipt['adam_moments_preserved'];assert receipt['scheduler_restart_epoch']==(326 if first else None) and receipt['scheduler_restart_lr']==(.0005 if first else None) and s.read(arm/'smoke.json')['initial_parameter_sha256']==s.digest(ck['model'])
    done=s.read(par.parent/'L2_audit_completed.json');assert done['passed'] and done['audit_sha256']==s.sha(par.parent/'L2_science/audit.json')
    if first:assert s.sha(par/'last_checkpoint.pt')==study.P['parent_last_sha256'] and s.sha(par/'history.json')==study.P['parent_history_sha256'] and done['audit_sha256']==study.P['parent_audit_sha256']
    offset=prefix[-1]['seconds'];previous=offset
    for row in hist[len(prefix):]:assert previous-offset<1800 and row['seconds']>=previous;previous=row['seconds']
    for row in hist:
        for role in ('train','val'):
            if role not in row:continue
            v=row[role];cm=np.asarray(v['confusion_matrix']);support=np.asarray(m['supports'][role]);assert cm.shape==(10,10) and np.issubdtype(cm.dtype,np.integer) and (cm>=0).all() and np.array_equal(cm.sum(1),support);recall=cm.diagonal()/support;assert np.allclose(v['recall'],recall,rtol=0,atol=1e-12) and math.isclose(v['accuracy'],float(cm.trace()/support.sum()),abs_tol=1e-12) and math.isclose(v['balanced_accuracy'],float(recall.mean()),abs_tol=1e-12)
    budget=independent_scheduler()(dict(cfg,maximum_seconds=float('inf'),minimum_epochs=float('inf')),hist,last,'maximum_epochs');assert last['scheduler']['last_epoch']==len(hist) and all(int(v['step'])==48*len(hist) for v in last['optimizer']['state'].values());return fit,budget
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==a.root/'L2_science' and not a.out.exists();meta=s.read(a.root/'metadata.json');study.TARGET=meta['target_epoch'];study.PARENT=Path(meta['parent']);assert meta['case']=='L2_rho03' and meta['sources']==study.sources();s.engine.verify_sources(meta['sources'],meta['git_commit']);claim=s.read(a.root/'L2_audit_claim.json');assert claim['sources']==meta['sources'] and claim['cases']==['L2_rho03'];rec=s.read(a.root/'L2_rho03/producer_exit.json');assert rec['returncode']==0 and rec['actual_exit'] and study.parent.base.exited(rec['pid']) and not (a.root/'L2_rho03/failure.json').exists();study.bind('L2_rho03');fit,budget=verify(a.root);a.out.mkdir();s.save(a.out/'audit.json',dict(passed=True,sources=meta['sources'],fit=fit,budget=budget,independent_MixUp_02_03_01_schedule=True,independent_epoch326_LR_statistics_restart=True,raw_EMA_and_Adam_moments_preserved=True,actual_Adam_steps=48*fit['epochs_completed'],model_forward_calls=0,test_arrays_read=0,test_informed_development=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        out=a.root/'L2_cpu_report_failure.json'
        if not out.exists():s.save(out,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
