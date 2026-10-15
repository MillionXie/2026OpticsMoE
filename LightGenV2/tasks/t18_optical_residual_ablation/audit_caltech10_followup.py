"""Independent saved math for one continuation and matched shallow pairs."""
import argparse,copy,inspect,math,os,textwrap
from pathlib import Path
import numpy as np
import torch
import caltech10_followup as study
import audit_resisc20 as sealed
from audit_resisc10 import predictions
from audit_resisc20_input_pair import scheduler_budget_checks
s=study.s
def verify(root,case,m):
    source=textwrap.dedent(inspect.getsource(sealed.verify));changes={"<=30":"==cfg['maximum_epochs']","s.r.CorrectedMoE(4,f['rho'])":"study.parent.SubsetMoE(f['depth'],f['rho'])","==908552":"==study.parameter_count(f['depth'])","len(last['optimizer']['state'])==11":"len(last['optimizer']['state'])==len(shapes)","labels=z[role+'_labels'].copy()":"labels=study.parent.subset_labels(z[role+'_labels'].copy())",'s.eligibility(f,hist)':'study.parent.eligibility(f,hist)'}
    if case=='regularize_L4_rho03':changes["cfg['mixup_alpha'],cfg['mixup_alpha']"]="(.2 if epoch<=150 else .4),(.2 if epoch<=150 else .4)"
    for old,new in changes.items():assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    ns=dict(sealed.__dict__,study=study,predictions=predictions);exec(compile(source,__file__+':sealed_independent_verify','exec'),ns)
    cfg=study.config(case);data=Path(cfg['data']);f=ns['verify'](root,case,data,m);depth,rho=study.CASES[case];cfg.update(depth=depth,residual_rho=rho,classes=m['classes'],contract=study.parent.contract(depth,rho));assert f['config']==cfg and f['sources']==study.sources() and f['stop_reason']=='maximum_epochs'
    h=s.read(root/case/'history.json');last=torch.load(root/case/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    for row in h:
        for role in ('train','val'):
            if role not in row:continue
            v=row[role];cm=np.asarray(v['confusion_matrix']);support=np.asarray(m['supports'][role]);assert cm.shape==(10,10) and np.issubdtype(cm.dtype,np.integer) and (cm>=0).all() and np.array_equal(cm.sum(1),support)
            recall=cm.diagonal()/support;assert np.allclose(v['recall'],recall,rtol=0,atol=1e-12) and math.isclose(v['accuracy'],float(np.trace(cm)/support.sum()),abs_tol=1e-12) and math.isclose(v['balanced_accuracy'],float(recall.mean()),abs_tol=1e-12)
    if case=='regularize_L4_rho03':
        arm,parent=study.parent_state();assert len(h)==200 and h[:150]==s.read(arm/'history.json') and s.read(root/case/'gradients.json')[:150]==s.read(arm/'gradients.json')
        receipt=s.read(root/case/'resume_receipt.json');assert receipt['parent_epoch']==150 and receipt['first_new_epoch']==151 and receipt['last_checkpoint_sha256']==s.sha(arm/'last_checkpoint.pt') and receipt['prefix_history_sha256']==s.sha(arm/'history.json')
        assert receipt['restored_state_sha256']=={k:s.digest(parent[k]) for k in ('model','ema')} and not any(receipt[k] for k in ('optimizer_reset','ema_reset','scheduler_reset'))
        assert s.read(root/case/'smoke.json')['initial_parameter_sha256']==s.digest(parent['model']);offset=h[149]['seconds'];prev=offset
        for row in h[150:]:assert prev-offset<1800 and row['seconds']>=prev;prev=row['seconds']
        budget=scheduler_budget_checks(dict(cfg,maximum_seconds=offset+1800),h,last,'maximum_epochs')
    else:assert len(h)==150;budget=scheduler_budget_checks(cfg,h,last,'maximum_epochs')
    return f,budget
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.root.name==study.RUN and a.out==a.root/(a.stage+'_science') and not a.out.exists()
    meta=s.read(a.root/'metadata.json');claim=s.read(a.root/(a.stage+'_audit_claim.json'));assert meta['sources']==claim['sources']==study.sources();s.engine.verify_sources(meta['sources'],meta['git_commit']);expected=['regularize_L4_rho03'] if a.stage=='regularize' else [a.stage+'_rho03',a.stage+'_rho0'];assert claim['cases']==expected
    m=study.parent.manifest(Path(study.parent.config()['data']));fits={};budgets={}
    for case in expected:
        rec=s.read(a.root/case/'producer_exit.json');assert rec['returncode']==0 and rec['actual_exit'] and study.parent.base.exited(rec['pid']) and not (a.root/case/'failure.json').exists()
        study.bind(case);fits[case],budgets[case]=verify(a.root,case,m)
    if len(fits)==2:
        assert len({f['initial_parameter_sha256'] for f in fits.values()})==len({tuple(f['orders']) for f in fits.values()})==len({tuple(f['transforms']) for f in fits.values()})==1
        cfgs=[]
        for f in fits.values():v=copy.deepcopy(f['config']);v.pop('residual_rho');v.pop('contract');cfgs.append(v)
        assert cfgs[0]==cfgs[1]
    a.out.mkdir(exist_ok=False);s.save(a.out/'audit.json',dict(passed=True,fits=fits,budget_checks=budgets,sources=meta['sources'],stage=a.stage,actual_adam_steps={k:f['epochs_completed']*48 for k,f in fits.items()},parameters_per_depth={k:study.parameter_count(f['depth']) for k,f in fits.items()},phase_tensors_per_depth={k:study.tensor_count(f['depth']) for k,f in fits.items()},firstminimum_validation_EMA_balancedNLL=True,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,scope='Residual continuation extra budget; shallow pairs matched150; exploratory fixed10 development'),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['regularize','L3','L2'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        fail=a.root/(a.stage+'_cpu_report_failure.json')
        if not fail.exists():s.save(fail,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
