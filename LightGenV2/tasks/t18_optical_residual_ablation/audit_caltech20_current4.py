"""Independent saved20-class verifier for new current4 Caltech equal100 pair."""
import argparse,copy,inspect,math,os,textwrap
from pathlib import Path
import torch
import caltech20_current4 as study
import audit_resisc20 as sealed
from audit_resisc20_input_pair import scheduler_budget_checks,history_metrics

def verifier():
    src=textwrap.dedent(inspect.getsource(sealed.verify));old="f['epochs_completed']<=30";assert src.count(old)==1
    ns=dict(sealed.__dict__);exec(compile(src.replace(old,"f['epochs_completed']==100"),__file__+':sealed20verify','exec'),ns)
    return ns['verify']

def audit(a):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.base.name==study.RUN and a.out==a.base/'cpu_report'
    assert not a.out.exists() and not (a.base/'cpu_report_failure.json').exists();study.bind();cfg=study.prerequisites();m=study.manifest(a.data)
    meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json');assert meta['sources']==study.sources() and s.read(a.base/'audit_claim.json')['sources']==meta['sources']
    assert study.exited(meta['pid']) and s.read(a.base/'queue_exit.json')['returncode']==0
    assert state['active']==state['pending']==0 and not state['failures'] and {j['case'] for j in state['completed']}==set(study.CASES)
    fits={};schedule={};verify=verifier()
    for j in state['completed']:
        assert j['exit_code']==0 and study.exited(j['pid']);key=j['case'];f=verify(a.base,key,a.data,m)
        expected=dict(cfg,depth=4,residual_rho=study.CASES[key],classes=m['classes'],contract=study.contract(4,study.CASES[key]))
        assert f['config']==expected and f['sources']==meta['sources'] and f['stop_reason']=='maximum_epochs'
        hist=s.read(a.base/key/'history.json');last=torch.load(a.base/key/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        schedule[key]=scheduler_budget_checks(expected,hist,last,f['stop_reason'])
        for h in hist:
            history_metrics(h['val'],m['supports']['val'])
            if 'train' in h:history_metrics(h['train'],m['supports']['train'])
        fits[key]=f
    normalized=[]
    for f in fits.values():
        v=copy.deepcopy(f['config'])
        for k in ('residual_rho','contract'):v.pop(k)
        normalized.append(v)
    assert normalized[0]==normalized[1] and len({f['initial_parameter_sha256'] for f in fits.values()})==1
    assert len({tuple(f['orders']) for f in fits.values()})==len({tuple(f['transforms']) for f in fits.values()})==1
    a.out.mkdir(exist_ok=False);s.save(a.out/'audit.json',dict(passed=True,fits=fits,sources=meta['sources'],training_commit=meta['git_commit'],scheduler_budget_checks=schedule,common_pair_epochs=100,actual_adam_steps_per_arm=8600,train_samples=1372,val_samples=283,tail_batch=12,cpu_models_for_shape_seed=2,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,new_audit_source_sha256=s.sha(__file__),scope='Current four experts, frozen screened Caltech20, equal100; no old nine-expert gain or test claim'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out','data'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        mark=a.base/'cpu_report_failure.json'
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
