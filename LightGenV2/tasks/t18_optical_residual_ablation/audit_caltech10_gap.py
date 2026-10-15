"""Independent ten-output saved science for frozen category development pair."""
import argparse,copy,os,math
import numpy as np
from pathlib import Path
import caltech10_gap as study
import audit_resisc10 as sealed
s=study.s
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.base.name==study.RUN and a.out==a.base/'cpu_report' and not a.out.exists()
    study.bind();sealed.study=study;m=study.manifest(a.data);meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json')
    assert study.base.exited(meta['pid']) and meta['sources']==study.sources() and s.read(a.base/'audit_claim.json')['sources']==meta['sources'];assert not state['failures'] and state['active']==state['pending']==0
    fits={};budgets={}
    for j in state['completed']:
        assert j['exit_code']==0 and study.base.exited(j['pid']);f,b=sealed.verify(a.base,j['case'],a.data,m);assert f['epochs_completed']==150 and f['stop_reason']=='maximum_epochs' and f['sources']==meta['sources'];fits[j['case']]=f;budgets[j['case']]=b
    for key in fits:
        for row in s.read(a.base/key/'history.json'):
            for role in ('train','val'):
                if role not in row:continue
                v=row[role];cm=np.asarray(v['confusion_matrix']);support=np.asarray(m['supports'][role]);assert cm.shape==(10,10) and np.issubdtype(cm.dtype,np.integer) and (cm>=0).all() and np.array_equal(cm.sum(1),support)
                recall=cm.diagonal()/support;assert np.allclose(v['recall'],recall,rtol=0,atol=1e-12) and math.isclose(v['accuracy'],float(np.trace(cm)/support.sum()),abs_tol=1e-12) and math.isclose(v['balanced_accuracy'],float(recall.mean()),abs_tol=1e-12)
    assert set(fits)==set(study.CASES) and len({f['initial_parameter_sha256'] for f in fits.values()})==1
    configs=[]
    for f in fits.values():
        v=copy.deepcopy(f['config']);v.pop('residual_rho');v.pop('contract');configs.append(v)
    assert configs[0]==configs[1] and len({tuple(f['orders']) for f in fits.values()})==len({tuple(f['transforms']) for f in fits.values()})==1
    a.out.mkdir(exist_ok=False);s.save(a.out/'audit.json',dict(passed=True,fits=fits,budget_checks=budgets,sources=meta['sources'],class_count=10,common_pair_epochs=150,train_samples=sum(m['supports']['train']),val_samples=sum(m['supports']['val']),optimizer_steps_per_epoch=48,actual_adam_steps_per_arm=7200,tail_batch=9,validation_informed_class_selection=True,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,scope='New selected10 development task, not original20 performance or untouched independent generalization'),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out','data'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        p=a.base/'cpu_report_failure.json'
        if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
