"""Human fixed two replacements, fresh equal100 pure-optical pair; old sources immutable."""
import argparse,copy,inspect,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import resisc10 as base
from audit_caltech_repair234 import exited
s=base.s;HERE=base.HERE;ROOT=base.ROOT
PROFILE=HERE/'resisc10_reselect_profile.json'
FILES=('resisc10_reselect.py','resisc10_reselect_profile.json','audit_resisc10_reselect.py','RESISC10_RESELECT_PROTOCOL_20261015.md')
_SOURCES=base.sources;_CONTRACT=base.contract;_FORWARD=base.SubsetMoE.forward
INDICES=(3,7,8,9,10,11,14,17,18,19)
RUN='resisc10_reselect_station_terrace_s17_pair100_mixup02_gpu25_20261015'
CASES={'L4_reselect_rho03':.3,'L4_reselect_rho0':0.}
CPU=HERE/'runs/smoke/resisc10_reselect_cpu_native_20261015'
NUMERIC=HERE/'runs/smoke/resisc10_reselect_gpu2_numeric_20261015'

def plan():
    p=s.read(PROFILE)
    assert tuple(p['original_indices'])==INDICES and p['classes']==[p['parent_classes'][i] for i in INDICES]
    assert set(p['removed_classes'])=={'bridge','tennis_court'} and set(p['added_classes'])=={'railway_station','terrace'}
    assert p['maximum_epochs']==100 and p['minimum_epochs']>100 and p['mixup_alpha']==.2
    assert s.sha(ROOT/p['selection_source'])==p['selection_source_sha256']
    return p

class SubsetMoE(s.r.CorrectedMoE):
    def __init__(self,depth,rho):
        super().__init__(depth,rho)
        self.detectors=self.detectors[list(INDICES)].clone()
SubsetMoE.forward=_FORWARD

def contract(depth,rho):
    v=_CONTRACT(depth,rho)
    v.update(original_detector_indices=list(INDICES),readout='Human fixed replacement10 original-coordinate windows; no electronic classifier')
    return v

def sources():
    v=_SOURCES()
    for name in FILES:v[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    assert len(v)==164
    return v

def config():
    p=plan();c=s.read(base.parent.PROFILE);c.pop('cases')
    c.update(dataset=p['dataset'],input_tile=100,subset_plan_sha256=s.sha(PROFILE),subset_manifest_sha256=s.sha(base.private_manifest()),class_count=10,original_indices=list(INDICES),maximum_epochs=100,maximum_seconds=p['maximum_seconds'],minimum_epochs=p['minimum_epochs'],mixup_alpha=.2,maximum_gpu_children=2,maximum_gpu_children_per_card=1,numeric_gate=NUMERIC.relative_to(ROOT).as_posix()+'/checks.json',cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',scope='Human confirmed two class replacements, fresh equal100 pair; validation-informed development, no test')
    return c

base.PLAN=PROFILE;base.plan=plan;base.INDICES=INDICES;base.SubsetMoE=SubsetMoE;base.contract=contract
base.private_manifest=lambda:Path(plan()['out'])/'data_manifest.json'
base.RUN=RUN;base.CASES=CASES;base.CPU=CPU;base.NUMERIC=NUMERIC;base.sources=sources;base.config=config

def claim(path,phase):
    assert not path.exists()
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    s.save(path,dict(pid=os.getpid(),phase=phase,sources=src,git_commit=commit,automatic_retry=False,test_evaluations=0),exclusive=True)

def cpu(a):
    assert a.out==CPU and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    base.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'new_replacement_cpu')
    import check_resisc10 as check
    import export_resisc10 as export
    export.INDICES=INDICES;export.contract=contract;export._ns['deployment_contract']=contract
    check.check(a.out)
    v=s.read(a.out/'checks.json');v.update(class_selection_fixed_validation_top10=False,human_two_replacements_fixed=True,original_detector_indices=list(INDICES),run=RUN)
    s.save(a.out/'checks.json',v)

def prepare(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path(plan()['out']).exists()
    value=base.prepare();print(__import__('json').dumps(value),flush=True)

def numeric(a):
    assert a.out==NUMERIC
    claim(a.out.with_name(a.out.name+'.claim.json'),'new_replacement_numeric')
    base.numeric(a)

def fit(a):
    base.fit(a)
    f=s.read(a.out/'fit_result.json');f['scope']='Human confirmed replacement10 fresh pair; no early stop, firstminimum validation EMA NLL, no test'
    s.save(a.out/'fit_result.json',f)
    assert f['epochs_completed']==100 and f['stop_reason']=='maximum_epochs','Safety budget or failure before100: preserve and do not call complete'

_q=textwrap.dedent(inspect.getsource(base.queue));assert _q.count("HERE/'resisc10.py'")==1
_ns=dict(base.__dict__);exec(compile(_q.replace("HERE/'resisc10.py'","HERE/'resisc10_reselect.py'"),__file__+':sealed_replacement_queue','exec'),_ns)
queue=_ns['queue']

def campaign(a):
    assert a.out==HERE/'runs/simulation'/RUN and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    base.prerequisites();base.slots();claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'new_replacement_campaign')
    with a.out.with_name(a.out.name+'.queue.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(HERE/'resisc10_reselect.py'),'--phase','queue','--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        rc=child.wait()
    assert a.out.exists();s.save(a.out/'queue_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=exited(child.pid)),exclusive=True)
    state=s.read(a.out/'status.json')
    assert rc==0 and exited(child.pid) and not state['failures'] and state['active']==state['pending']==0 and len(state['completed'])==2,'Pair failed: no audit/retry'
    assert all(j['exit_code']==0 and exited(j['pid']) for j in state['completed'])
    assert all(s.read(a.out/j['case']/'fit_result.json')['epochs_completed']==100 for j in state['completed'])
    claim(a.out/'audit_claim.json','new_replacement_saved_science')
    with (a.out/'audit.log').open('x') as log:
        rc=subprocess.run([sys.executable,str(HERE/'audit_resisc10_reselect.py'),'--base',str(a.out),'--out',str(a.out/'cpu_report'),'--data',config()['data']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,'Independent saved science failed: preserve, no retry'
    assert s.read(a.out/'cpu_report/audit.json')['passed']
    s.save(a.out/'audit_completed.json',dict(passed=True,audit_sha256=s.sha(a.out/'cpu_report/audit.json'),actual_audit_returncode=rc),exclusive=True)
    s.save(a.out/'status.json',dict(state, state='replacement10_complete_science_passed'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','cpu','numeric','fit','queue','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--data',type=Path);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        dest=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not dest.exists():s.save(dest,dict(error=repr(e),pid=os.getpid(),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
