"""Independent saved two-arm continuation audit, no forward/test."""
import argparse,copy,inspect,math,os,textwrap
from pathlib import Path
import torch
import caltech20_continue200 as study
import audit_resisc20 as sealed
from audit_resisc20_input_pair import scheduler_budget_checks,history_metrics
s=study.s
def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.base.name==study.RUN and a.out==a.base/'cpu_report' and not a.out.exists()
    meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json');assert meta['sources']==study.sources() and study.base.exited(meta['pid']);assert s.read(a.base/'queue_exit.json')['returncode']==0
    assert not state['failures'] and state['active']==state['pending']==0 and {j['case'] for j in state['completed']}==set(study.CASES)
    source=textwrap.dedent(inspect.getsource(sealed.verify));assert source.count("f['epochs_completed']<=30")==1
    ns=dict(sealed.__dict__);exec(compile(source.replace("f['epochs_completed']<=30","f['epochs_completed']==200"),__file__+':independentverify','exec'),ns)
    fits={};schedules={}
    for key in study.CASES:
        study.bind(key);arm,parent=study.parent(key);f=ns['verify'](a.base,key,a.data,study.base.manifest(a.data));cfg=study.config(key)
        assert f['config']==cfg and f['sources']==meta['sources'] and f['stop_reason']=='maximum_epochs'
        h=s.read(a.base/key/'history.json');g=s.read(a.base/key/'gradients.json');assert h[:100]==s.read(arm/'history.json') and g[:100]==s.read(arm/'gradients.json')
        last=torch.load(a.base/key/'last_checkpoint.pt',map_location='cpu',weights_only=False);offset=h[99]['seconds'];prev=offset
        for row in h[100:]:assert prev-offset<3600 and row['seconds']>=prev;prev=row['seconds']
        schedules[key]=scheduler_budget_checks(dict(cfg,maximum_seconds=offset+3600),h,last,'maximum_epochs')
        receipt=s.read(a.base/key/'resume_receipt.json');assert receipt['parent_epoch']==100 and receipt['first_new_epoch']==101 and receipt['last_checkpoint_sha256']==s.sha(arm/'last_checkpoint.pt')
        assert receipt['prefix_history_sha256']==s.sha(arm/'history.json') and receipt['restored_state_sha256']=={k:s.digest(parent[k]) for k in ('model','ema')}
        assert not receipt['optimizer_reset'] and not receipt['ema_reset'] and not receipt['scheduler_reset']
        smoke=s.read(a.base/key/'smoke.json');assert smoke['passed'] and smoke['initial_parameter_sha256']==s.digest(parent['model'])
        m=study.base.manifest(a.data)
        for row in h:
            history_metrics(row['val'],m['supports']['val'])
            if 'train' in row:history_metrics(row['train'],m['supports']['train'])
        fits[key]=f
    norm=[]
    for f in fits.values():
        v=copy.deepcopy(f['config'])
        for k in ('residual_rho','contract'):v.pop(k)
        norm.append(v)
    assert norm[0]==norm[1] and len({tuple(f['orders']) for f in fits.values()})==len({tuple(f['transforms']) for f in fits.values()})==len({f['initial_parameter_sha256'] for f in fits.values()})==1
    a.out.mkdir(exist_ok=False);s.save(a.out/'audit.json',dict(passed=True,fits=fits,sources=meta['sources'],scheduler_budget_checks=schedules,common_pair_epochs=200,actual_adam_steps_per_arm=17200,preserved_prefix_epochs=100,optimizer_ema_scheduler_restored=True,model_forward_calls=0,test_arrays_read=0,test_evaluations=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','out','data'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        q=a.base/'cpu_report_failure.json'
        if not q.exists():s.save(q,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
