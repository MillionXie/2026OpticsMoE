"""Execution-only RESISC continuation: preserve full data/macro16/optimizer budget."""
import argparse,inspect,json,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import resisc20 as s
import resisc20_batch_probe as probe
import resisc20_mask_batch as batch
HERE=s.HERE;ROOT=s.ROOT;PARENT=probe.BASE
GATE=HERE/'runs/smoke/resisc20_scalar_mask_batch_gpu5_20261014/checks.json'
STOP=PARENT.with_suffix('.speed_stop_receipt_20261014.json')
FILES=('resisc20_batch_probe.py','resisc20_mask_batch.py','resisc20_microbatch.py','audit_resisc20_microbatch.py','RESISC20_SPEED_PROTOCOL_20261014.md')
def sources():
 src=s.sources()
 for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return src

def prerequisites():
 proof=s.read(GATE);assert proof['passed'];s.engine.verify_sources(proof['sources'],proof['git_commit']);micro=proof['chosen_microbatch']
 assert micro in (2,4,8) and micro in proof['common_eligible_microbatches']
 assert all(next(v for v in c['checks'] if v['microbatch']==micro)['speedup_vs_micro1']>=1.1 for c in proof['checks'])
 assert proof['physical_graph_unchanged'] and proof['original_micro1_unchanged']
 return micro

def gate(depth,rho,data,cfg,device,micro,state=None):
 assert depth==4 and micro==prerequisites();s.engine.seed(cfg['seed']);model=s.r.CorrectedMoE(depth,rho).to(device)
 if state is not None:model.load_state_dict(state,strict=True)
 initial=s.digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
 _,theta=s.engine.epoch_plan(1,len(data[1]),cfg);groups=[np.asarray([int((data[1]==k).nonzero()[0]) for k in range(20)]),np.arange(14)]
 checks=[]
 for idx in groups:
  x,y,_=s.engine.mixed_batch(data,idx,theta,1,0,cfg,device);v=probe.compare(probe.compute(model,x,y,weights,cfg,1),probe.compute(model,x,y,weights,cfg,micro));assert v['passed'],v;checks.append(v)
 norms={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert len(norms)==11
 del model;torch.cuda.empty_cache()
 return dict(passed=True,depth=depth,rho=rho,micro=micro,classes_checked=20,tail_checked=14,checks=checks,all_phase_gradients_finite_nonzero=True,phase_gradients=norms,initial_parameter_sha256=initial,parameters=908552,test_evaluations=0)

def bind(micro):
 s.bind();batch.install();original_read=s.read
 def read(path):
  v=original_read(path)
  return dict(v,microbatch=micro) if Path(path)==s.PROFILE else v
 s.engine.read=read;s.engine.sources=sources;s.engine.fresh_gate=gate
 # Compile the sealed fit with three explicit execution/resume bookkeeping
 # substitutions. Model/objective/order/MixUp/Adam/EMA/scheduler code is unchanged.
 source=textwrap.dedent(inspect.getsource(s.engine.fit))
 replacements={
  "for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib');v.pop('microbatch')",
  "offset=history[-1]['seconds']":"offset=max(history[-1]['seconds'],a.resume_elapsed_seconds)",
  "old_memory_gib=3,new_memory_gib=2":"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],old_microbatch=resume['config']['microbatch'],new_microbatch=cfg['microbatch'],conservative_prior_seconds=offset"}
 for old,new in replacements.items():assert source.count(old)==1;source=source.replace(old,new)
 namespace=dict(s.engine.__dict__);exec(compile(source,str(HERE/'resisc20_microbatch.py')+':sealed_fit_execution_bind','exec'),namespace)
 return namespace['fit']

def stopped():
 stop=s.read(STOP);assert stop['passed'] and stop['user_authorized_speed_migration'] and stop['pending']==0
 for j in stop['children']:
  p=Path('/proc')/str(j['pid']);assert not p.exists() or not (p/'cmdline').read_bytes()
 meta=s.read(PARENT/'metadata.json');p=Path('/proc')/str(meta['pid']);assert not p.exists() or not (p/'cmdline').read_bytes()
 s.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['git_commit']=='513f0e884a16be5f37846974deaeea55fac95f94'
 return stop

def fit(a):
 micro=prerequisites();stop=stopped();s.slots();a.resume_parent=PARENT/a.case;a.resume_elapsed_seconds=stop['conservative_prior_fit_seconds'];function=bind(micro)
 function(a);f=s.read(a.out/'fit_result.json');f['scope']='Full-resolution RESISC validation pair, same macro16/full data and total30/14400 budget; disclosed microbatch execution continuation, not a new candidate or independent Caltech recovery.'
 f['execution_migration']=dict(parent=str(a.resume_parent),old_microbatch=1,new_microbatch=micro,prefix_epochs=next(j['epoch'] for j in stop['children'] if j['case']==a.case),stop_receipt_sha256=s.sha(STOP),probe_sha256=s.sha(GATE),no_completed_epoch_replayed=True,optimizer_ema_scheduler_reset=False)
 s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',s.eligibility(f,s.read(a.out/'history.json')))

def queue(a):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();micro=prerequisites();stop=stopped();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
 a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,config=dict(s.read(s.PROFILE),microbatch=micro),command=sys.argv,resume_parent=str(PARENT),stop_receipt_sha256=s.sha(STOP),phase='user_authorized_execution_speed_continuation_not_extra_fit_budget',test_evaluations=0))
 pending=[.3,0.];active=[];complete=[];failures=[]
 while pending or active:
  for j in list(active):
   rc=j['process'].poll()
   if rc is not None:
    j['log'].close();record={k:v for k,v in j.items() if k not in ('process','log')};record['exit_code']=rc;(complete if rc==0 else failures).append(record);active.remove(j)
  if failures:pending=[]
  for index in (6,2,5):
   if not pending or failures:break
   gpu=s.ALLOWED[index];slots=s.slots()
   if slots['total']>=6 or slots['counts'][gpu]>=2 or any(j['gpu_uuid']==gpu for j in active):continue
   s.gpu_budget.ALLOWED.update(s.ALLOWED.values())
   try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
   except AssertionError:continue
   rho=pending.pop(0);case='L4_rho03' if rho else 'L4_rho0';out=a.out/case;assert not out.exists();log=(a.out/(case+'.log')).open('x');cmd=[sys.executable,str(HERE/'resisc20_microbatch.py'),'--phase','fit','--depth','4','--rho',str(rho),'--case',case,'--gpu',gpu,'--data',s.read(s.PROFILE)['data'],'--out',str(out)]
   child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=case,rho=rho,pid=child.pid,gpu_uuid=gpu,process=child,log=log))
  s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending));s.save(a.out/'status.json',dict(state='speed_pair_failed' if failures else 'speed_pair_running' if active or pending else 'speed_pair_complete_awaiting_CPU_audit',completed=complete,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
  if active or pending:time.sleep(5)
 s.engine.verify_sources(src,commit)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['queue','fit'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--case');p.add_argument('--gpu');p.add_argument('--data',type=Path);a=p.parse_args()
 try:globals()[a.phase](a)
 except BaseException as e:
  if a.out.exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0))
  raise
