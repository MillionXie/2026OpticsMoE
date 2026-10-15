"""Correct assistant's150-epoch budget deviation with exact100 fresh pairs."""
import argparse,copy,inspect,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import caltech10_followup as prior
s=prior.s;parent=prior.parent;baseline=prior.baseline;resources=prior.resources
HERE=prior.HERE;ROOT=prior.ROOT;PROFILE=HERE/'caltech10_equal100_profile.json'
FILES=('caltech10_equal100.py','caltech10_equal100_profile.json','audit_caltech10_equal100.py','CALTECH10_EQUAL100_CORRECTION_PROTOCOL_20261015.md')
RUN=s.read(PROFILE)['run'];CPU=HERE/'runs/smoke/caltech10_equal100_contract_20261015';gate=prior.gate;parameter_count=prior.parameter_count;tensor_count=prior.tensor_count
wait_slot=prior.wait_slot
CASES={f'L{d}_rho'+('03' if rho==.3 else '0'):(d,rho) for d in (4,3,2) for rho in (.3,0.)}
def sources():
    v=prior.sources()
    for n in FILES:v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return v
def config(case):
    assert case in CASES;c=parent.config();p=s.read(PROFILE);c.update(maximum_epochs=p['maximum_epochs'],maximum_seconds=p['maximum_fit_seconds_per_arm'],minimum_epochs=p['minimum_epochs'],maximum_gpu_children=4,cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',numeric_gate=prior.NUMERIC.relative_to(ROOT).as_posix()+'/checks.json',scope=p['scope']);return c
# Reuse the unchanged fresh fitting body in this process; the old modules stay frozen.
_src=textwrap.dedent(inspect.getsource(prior.bind));exec(compile(_src,__file__+':fresh100_bind','exec'),globals())
def cpu(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==CPU and not a.out.exists();resources.slots();proofs={}
    for module,out in ((parent,parent.CPU),(parent,parent.NUMERIC),(prior,prior.CPU),(prior,prior.NUMERIC)):
        v=s.read(out/'checks.json');assert v['passed'] and v['sources']==module.sources();s.engine.verify_sources(v['sources'],v['git_commit']);proofs[out.relative_to(ROOT).as_posix()]=s.sha(out/'checks.json')
    for case in CASES:
        cfg,fn=bind(case);assert cfg['maximum_epochs']==100 and cfg['minimum_epochs']>100 and cfg['maximum_seconds']==1800 and cfg['mixup_alpha']==.2
    import audit_caltech10_equal100 as auditor
    assert callable(auditor.audit) and callable(auditor.verify)
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,git_commit=commit,automatic_retry=False),exclusive=True);a.out.mkdir(parents=True,exist_ok=False)
    s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,epochs_per_arm=100,optimizer_steps_per_arm=4800,tail_batch=9,reused_immutable_graph_proof_sha256=proofs,new_producer_own_gates_required=True,models_constructed=0,forward_calls=0,images_read=0,test_arrays_read=0),exclusive=True)
def prerequisites():
    v=s.read(CPU/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
def fit(a):
    assert a.case in CASES and (a.depth,a.rho)==CASES[a.case] and a.gpu in resources.GPU.values();prerequisites();resources.slots();cfg,fn=bind(a.case);a.data=Path(cfg['data']);a.resume_parent=None;fn(a);f=s.read(a.out/'fit_result.json');f['scope']=cfg['scope'];s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',parent.eligibility(f,s.read(a.out/'history.json')));assert f['epochs_completed']==100 and f['stop_reason']=='maximum_epochs'
# Separate claims/auditor/test paths; no old one-shot entry point is rerun.
for name,changes in (('audit_stage',{"'audit_caltech10_followup.py'":"'audit_caltech10_equal100.py'"}),('test',{}),('test_once',{"'caltech10_followup.py'":"'caltech10_equal100.py'"})):
    src=textwrap.dedent(inspect.getsource(getattr(prior,name)))
    for old,new in changes.items():assert src.count(old)==1;src=src.replace(old,new)
    exec(compile(src,__file__+':new100_'+name,'exec'),globals())
def campaign(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==HERE/'runs/simulation'/RUN and not a.out.exists();prerequisites();resources.slots();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'claim.json',dict(pid=os.getpid(),sources=src,git_commit=commit,automatic_retry=False,contract='Bothrho100; correction of assistant150 deviation'),exclusive=True);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=s.read(PROFILE),command=sys.argv),exclusive=True)
    pending=list(CASES);active=[];completed=[]
    while pending or active:
        for j in list(active):
            if j['process'].poll() is not None:completed.append(prior.finish_job(a.out,j));active.remove(j)
        for index in (0,2,4,5):
            if not pending:break
            gpu=resources.GPU[index];live=resources.slots()
            if live['total']>=4 or live['counts'][gpu] or any(j['gpu_uuid']==gpu for j in active):continue
            s.gpu_budget.ALLOWED.update(resources.GPU.values())
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            case=pending.pop(0);depth,rho=CASES[case];out=a.out/case;assert not out.exists();log=(a.out/(case+'.log')).open('x');cmd=[sys.executable,str(HERE/'caltech10_equal100.py'),'--phase','fit','--case',case,'--depth',str(depth),'--rho',str(rho),'--gpu',gpu,'--out',str(out)];child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=case,depth=depth,rho=rho,pid=child.pid,gpu_uuid=gpu,process=child,log=log))
        s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending,completed=completed));s.save(a.out/'status.json',dict(state='equal100_training' if active or pending else 'equal100_awaiting_science',active=len(active),pending=len(pending),completed=completed,failures=[],test_used_for_selection=False))
        if active or pending:time.sleep(3)
    for stage in ('L4','L3','L2'):
        audit_stage(a.out,stage,[stage+'_rho03',stage+'_rho0']);vals=[test_once(a.out,stage+suffix,index) for suffix,index in (('_rho03',4),('_rho0',2))];gap=100*(vals[0]['test']['accuracy']-vals[1]['test']['accuracy']);s.save(a.out/(stage+'_paired_test.json'),dict(residual_minus_control_pp=gap,macro_difference_pp=100*(vals[0]['test']['balanced_accuracy']-vals[1]['test']['balanced_accuracy']),target_gap10pp_achieved=gap>=10,test_used_for_selection=False,test_evaluations_per_model=1),exclusive=True)
    s.save(a.out/'current_processes.json',dict(active=[],pending=[],completed=completed));s.save(a.out/'status.json',dict(state='equal100_complete_science_and_frozen_tests_passed',active=0,pending=0,completed=completed,failures=[],test_used_for_selection=False,automatic_further_training=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','fit','test','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
