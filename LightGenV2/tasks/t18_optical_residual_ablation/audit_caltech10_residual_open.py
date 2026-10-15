"""Independent saved-state review of each open residual continuation block."""
import argparse,copy,hashlib,inspect,math,os,textwrap
from pathlib import Path
import numpy as np
import torch
import caltech10_residual_open as study
import audit_resisc20 as sealed
from audit_resisc10 import predictions
from audit_resisc20_input_pair import scheduler_budget_checks
s=study.s
def verify(root,case,m):
    src=textwrap.dedent(inspect.getsource(sealed.verify));changes={"<=30":"==cfg['maximum_epochs']","s.r.CorrectedMoE(4,f['rho'])":"study.parent.SubsetMoE(f['depth'],f['rho'])","==908552":"==study.parameter_count(f['depth'])","len(last['optimizer']['state'])==11":"len(last['optimizer']['state'])==len(shapes)","labels=z[role+'_labels'].copy()":"labels=study.parent.subset_labels(z[role+'_labels'].copy())",'s.eligibility(f,hist)':'study.parent.eligibility(f,hist)',"cfg['mixup_alpha'],cfg['mixup_alpha']":"(.2 if epoch<=100 else .3),(.2 if epoch<=100 else .3)"}
    for old,new in changes.items():assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    ns=dict(sealed.__dict__,study=study,predictions=predictions);exec(compile(src,__file__+':independent_verify','exec'),ns);cfg=study.config(case);f=ns['verify'](root,case,Path(cfg['data']),m);depth,rho=study.CASES[case];cfg.update(depth=depth,residual_rho=rho,classes=m['classes'],contract=study.parent.contract(depth,rho));assert f['config']==cfg and f['sources']==study.sources() and f['stop_reason']=='maximum_epochs'
    hist=s.read(root/case/'history.json');grad=s.read(root/case/'gradients.json');parent=study.PARENT;prefix=s.read(parent/'history.json');receipt=s.read(root/case/'resume_receipt.json');ck=torch.load(parent/'last_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(root/case/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert len(hist)==len(prefix)+25==study.TARGET and hist[:len(prefix)]==prefix and grad[:len(prefix)]==s.read(parent/'gradients.json')
    assert receipt['parent_epoch']==len(prefix) and receipt['first_new_epoch']==len(prefix)+1 and receipt['last_checkpoint_sha256']==s.sha(parent/'last_checkpoint.pt') and receipt['prefix_history_sha256']==s.sha(parent/'history.json')
    assert receipt['restored_state_sha256']=={k:s.digest(ck[k]) for k in ('model','ema')} and not any(receipt[k] for k in ('optimizer_reset','ema_reset','scheduler_reset')) and s.read(root/case/'smoke.json')['initial_parameter_sha256']==s.digest(ck['model'])
    done=parent.parent/(case[:2]+'_audit_completed.json');audit=parent.parent/(case[:2]+'_science')/'audit.json'
    prev=s.read(done);assert prev['passed'] and prev['audit_sha256']==s.sha(audit)
    offset=prefix[-1]['seconds'];previous=offset
    for row in hist[len(prefix):]:assert previous-offset<1800 and row['seconds']>=previous;previous=row['seconds']
    for row in hist:
        for role in ('train','val'):
            if role not in row:continue
            v=row[role];cm=np.asarray(v['confusion_matrix']);support=np.asarray(m['supports'][role]);assert cm.shape==(10,10) and np.issubdtype(cm.dtype,np.integer) and (cm>=0).all() and np.array_equal(cm.sum(1),support);recall=cm.diagonal()/support;assert np.allclose(v['recall'],recall,rtol=0,atol=1e-12) and math.isclose(v['accuracy'],float(cm.trace()/support.sum()),abs_tol=1e-12) and math.isclose(v['balanced_accuracy'],float(recall.mean()),abs_tol=1e-12)
    budget=scheduler_budget_checks(dict(cfg,maximum_seconds=float('inf'),minimum_epochs=float('inf')),hist,last,'maximum_epochs');return f,budget
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and a.out==a.root/(a.stage+'_science');meta=s.read(a.root/'metadata.json');study.TARGET=meta['target_epoch'];study.PARENT=Path(meta['parent']);case=meta['case'];assert case==a.stage+'_rho03' and meta['sources']==study.sources();s.engine.verify_sources(meta['sources'],meta['git_commit']);claim=s.read(a.root/(a.stage+'_audit_claim.json'));assert claim['sources']==meta['sources'] and claim['cases']==[case]
    rec=s.read(a.root/case/'producer_exit.json');assert rec['returncode']==0 and rec['actual_exit'] and study.parent.base.exited(rec['pid']) and not (a.root/case/'failure.json').exists();study.bind(case);m=study.parent.manifest(Path(study.parent.config()['data']));f,budget=verify(a.root,case,m);a.out.mkdir();s.save(a.out/'audit.json',dict(passed=True,sources=meta['sources'],fit=f,budget_checks=budget,actual_adam_steps=f['epochs_completed']*48,restored_prefix_preserved=True,independent_MixUp_schedule_02_then_03=True,model_forward_calls=0,test_arrays_read=0,test_informed_development=True,scope='Residual extra budget versus permanently frozen100 control'),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        fail=a.root/(a.stage+'_cpu_report_failure.json')
        if not fail.exists():s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
