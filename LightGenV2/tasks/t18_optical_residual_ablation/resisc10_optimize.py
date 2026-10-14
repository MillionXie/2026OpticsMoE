"""One conditional pure-optical MixUp correction; original160 sources never edited."""
import argparse,inspect,os,textwrap
from pathlib import Path
import resisc10 as base
from audit_caltech_repair234 import exited
s=base.s;PROFILE=base.HERE/'resisc10_optimize_profile.json'
_PARENT_RUN=base.RUN;_SOURCE=base.sources;_CONFIG=base.config
FILES=('resisc10_optimize.py','resisc10_optimize_profile.json','check_resisc10_optimize.py','audit_resisc10_optimize.py','RESISC10_NIGHT_OPTIMIZATION_20261015.md')
def admit_parent():
    cfg=s.read(PROFILE);b=base.HERE/'runs/simulation'/cfg['parent_run'];meta=s.read(b/'metadata.json');status=s.read(b/'status.json');proof=s.read(b/'cpu_report/audit.json');done=s.read(b/'saved_cpu_audit_dispatch_completed_20261015.json')
    assert meta['git_commit']==cfg['parent_source_commit'] and meta['sources']==_SOURCE() and len(meta['sources'])==160
    s.engine.verify_sources(meta['sources'],meta['git_commit'])
    assert exited(meta['pid']) and status['active']==status['pending']==0 and not status['failures'] and len(status['completed'])==2 and all(exited(j['pid']) and j['exit_code']==0 for j in status['completed'])
    assert proof['passed'] and done['passed'] and done['audit_sha256']==s.sha(b/'cpu_report/audit.json') and proof['class_count']==10
    assert not (b/'cpu_report_failure.json').exists() and proof['test_evaluations']==0
    f=proof['fits']['L4_10_rho03'];assert f['metrics']['val']['accuracy']<cfg['target_validation_accuracy'],'Pure optical target already reached: prioritize acceptance, do not search'
    return dict(parent=str(b),audit_sha256=s.sha(b/'cpu_report/audit.json'),parent_selected_checkpoint_sha256=f['checkpoint_sha256'],parent_selected_epoch=f['selected_epoch'],parent_validation_accuracy=f['metrics']['val']['accuracy'])
def sources():
    v=_SOURCE()
    for name in FILES:v[(base.HERE/name).relative_to(base.ROOT).as_posix()]=s.sha(base.HERE/name)
    return v
def config():
    parent=admit_parent();c=_CONFIG();p=s.read(PROFILE)
    c.update(mixup_alpha=p['mixup_alpha'],study_case='mixup02',optimization_profile_sha256=s.sha(PROFILE),optimization_parent=parent,scope='Validation selected top10 development; fresh pure optical paired MixUp .2 optimization, no test')
    return c
base.RUN='resisc10_s17_mainL4_input100_rho_pair120_mixup02_micro8_gpu25_20261015'
base.CPU=base.HERE/'runs/smoke/resisc10_mixup02_cpu_native_20261015'
base.NUMERIC=base.HERE/'runs/smoke/resisc10_mixup02_gpu2_numeric_20261015'
base.CASES={'L4_10opt_rho03':.3,'L4_10opt_rho0':0.}
base.sources=sources;base.config=config
_q=textwrap.dedent(inspect.getsource(base.queue));assert _q.count("HERE/'resisc10.py'")==1
_ns=dict(base.__dict__);exec(compile(_q.replace("HERE/'resisc10.py'","HERE/'resisc10_optimize.py'"),__file__+':sealed_opt_queue','exec'),_ns);queue=_ns['queue']
def __getattr__(name):return getattr(base,name)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['numeric','fit','queue'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--data',type=Path);a=p.parse_args()
    try:
        admit_parent();(queue if a.phase=='queue' else getattr(base,a.phase))(a)
    except BaseException as e:
        if a.out.exists() and not (a.out/'failure.json').exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
