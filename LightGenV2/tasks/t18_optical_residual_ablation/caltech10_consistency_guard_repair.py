"""Auditable exact-self resource guard repair; original failed205 stays frozen."""
import argparse,inspect,os,subprocess,textwrap
from pathlib import Path
import caltech10_consistency_diagnosis as failed
s=failed.s;HERE=failed.HERE;ROOT=failed.ROOT
FILES=('caltech10_consistency_guard_repair.py','CALTECH10_CONSISTENCY_GUARD_REPAIR_PROTOCOL_20261015.md')
FAILED=HERE/'runs/smoke/caltech10_consistency_train_diagnosis_20261015'
FAILED_SHA='6b7bccd08d2350e07b2116ce6b3b21a8707a9d105475f4342095c39c99cbd0bb'
CPU=HERE/'runs/smoke/caltech10_consistency_self_guard_cpu_20261015'
OUT=HERE/'runs/smoke/caltech10_consistency_guard_repaired_train_20261015'
def sources():
 src=failed.sources()
 for name in FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
 return src
def exclusive_self(jobs,pid,gpu):
 assert gpu==failed.prior.resources.GPU[4]
 assert len(jobs)==1 and jobs[0]==dict(pid=pid,gpu_uuid=gpu),jobs
 return True
def preserved_failure():
 path=FAILED.with_name(FAILED.name+'.failure.json');assert s.sha(path)==FAILED_SHA and not FAILED.exists() and not FAILED.with_name(FAILED.name+'.claim.json').exists();return path
def checked_run():
 src=textwrap.dedent(inspect.getsource(failed.run));a="HERE/'runs/smoke/caltech10_consistency_train_diagnosis_20261015'";b="and prior.resources.slots()['total']==0"
 assert src.count(a)==src.count(b)==1
 src=src.replace(a,'OUT').replace(b,"and exclusive_self(prior.resources.slots()['jobs'],os.getpid(),a.gpu)")
 original='dict(passed=True,sources=src';assert src.count(original)==1
 src=src.replace(original,"dict(passed=True,guard_repair_cpu_sha256=s.sha(CPU/'checks.json'),preserved_prior_failure_sha256=FAILED_SHA,sources=src")
 ns=dict(failed.__dict__,sources=sources,OUT=OUT,CPU=CPU,FAILED_SHA=FAILED_SHA,exclusive_self=exclusive_self);exec(compile(src,__file__+':unchanged_diagnostic_with_exact_self_guard','exec'),ns);return ns['run']
def cpu(a):
 assert a.out==CPU and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not CPU.exists() and not CPU.with_name(CPU.name+'.claim.json').exists();preserved_failure();checked_run();gpu=failed.prior.resources.GPU[4];assert exclusive_self([dict(pid=123,gpu_uuid=gpu)],123,gpu)
 rejected=0
 for jobs,requested in [([],gpu),([dict(pid=124,gpu_uuid=gpu)],gpu),([dict(pid=123,gpu_uuid=failed.prior.resources.GPU[2])],gpu),([dict(pid=123,gpu_uuid=gpu),dict(pid=124,gpu_uuid=gpu)],gpu),([dict(pid=123,gpu_uuid=gpu),dict(pid=124,gpu_uuid=failed.prior.resources.GPU[2])],gpu),([dict(pid=123,gpu_uuid=failed.prior.resources.A100)],gpu),([dict(pid=123,gpu_uuid=failed.prior.resources.A100)],failed.prior.resources.A100)]:
  try:exclusive_self(jobs,123,requested)
  except AssertionError:rejected+=1
  else:raise AssertionError('Resource guard accepted a non-exclusive or prohibited allocation')
 assert rejected==7;src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);assert len(src)==207;s.save(CPU.with_name(CPU.name+'.claim.json'),dict(sources=src,git_commit=commit,automatic_retry=False),exclusive=True);CPU.mkdir(parents=True);s.save(CPU/'checks.json',dict(passed=True,sources=src,git_commit=commit,preserved_prior_failure_sha256=FAILED_SHA,accept_exact_self=True,rejected_cases=rejected,global_resource_limits_unchanged=True,real_models=0,forward_calls=0,optimizer_steps=0,test_arrays_read=0),exclusive=True)
def diagnose(a):
 assert a.out==OUT and not OUT.exists() and not OUT.with_name(OUT.name+'.claim.json').exists() and not OUT.with_name(OUT.name+'.failure.json').exists();preserved_failure();proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources() and proof['preserved_prior_failure_sha256']==FAILED_SHA;s.engine.verify_sources(proof['sources'],proof['git_commit']);checked_run()(a);v=s.read(OUT/'diagnosis.json');assert v['passed'] and v['sources']==sources() and v['guard_repair_cpu_sha256']==s.sha(CPU/'checks.json');preserved_failure()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','diagnose'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');a=p.parse_args()
 try:cpu(a) if a.phase=='cpu' else diagnose(a)
 except BaseException as exc:
  mark=a.out.with_name(a.out.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(exc),automatic_retry=False),exclusive=True)
  raise
