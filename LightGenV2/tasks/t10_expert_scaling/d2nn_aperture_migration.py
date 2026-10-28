"""Release A100 and resume N16 on GPU0; preserve healthy N25/N49 workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .d2nn_aperture_rerun import sha,save,read,geometry
TASK=Path(__file__).resolve().parent;REPO=TASK.parents[2]
PROFILE=TASK/'configs/kather_d2nn_aperture_migration_20261029.json'
def command(pid):
 p=Path('/proc',str(pid),'cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
def verify(plan):
 for p,h in plan['source_files'].items():assert sha(REPO/p)==h,p

def worker(root):
 import argparse,torch
 from . import train
 plan=read(root/'plan.json');verify(plan);r=next(r for r in plan['runs'] if r['experts']==16);out=Path(r['run_dir'])
 assert os.environ['CUDA_VISIBLE_DEVICES']==r['training_gpu_uuid'] and not (out/'metadata.json').exists()
 resume=read(out/'migration_resume_identity.json')
 assert sha(out/'last_checkpoint.pt')==resume['last_checkpoint_sha256'] and sha(out/'best_checkpoint.pt')==resume['best_checkpoint_sha256']
 a=argparse.Namespace(data=Path(plan['data']),out=out,arch='d2nn_same_aperture',experts=16,top_k=16,layers=4,epochs=60,lr=.002,seed=17,microbatch=2,padding=2,smoke_limit=0,resume=True,profile=None)
 save(out/'run_identity.json',dict(source_commit=plan['source_commit'],source_files=plan['source_files'],pid=os.getpid(),training_gpu_uuid=r['training_gpu_uuid'],resume=resume,metadata_git_commit_scope='host HEAD; exact source overlay in this identity',timestamp=time.time()))
 try:train.run(a)
 except BaseException:
  import traceback
  save(out/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
 finally:torch.cuda.empty_cache()

def scheduler(root):
 cfg=read(PROFILE);source=os.environ['T10_D2NN_MIGRATION_SOURCE_COMMIT'];oldroot=REPO/cfg['old_root'];old=read(oldroot/'launch.json')
 assert not command(old['pid'])
 rows=[]
 for orig in old['runs']:
  r=orig.copy();n=r['experts'];r['training_gpu_uuid']=r['gpu_uuid'];r['test_gpu_uuid']=cfg['TEST_assignments'][str(n)]
  if n==16:r.update(run_dir=str(root/'d2nn_same_aperture_N16_s17_resume'),source_run=orig['run_dir'],training_gpu_uuid=cfg['N16_resume_GPU'])
  rows.append(r)
 for orig in old['runs']:
  if orig['experts'] in [25,49]:assert '--worker --experts '+str(orig['experts'])+' --root '+str(oldroot) in command(orig['train_pid']) or read(Path(orig['run_dir'])/'status.json')['state']=='complete'
 files=old['source_files'].copy()
 for p in [Path(__file__),PROFILE]:files[str(p.relative_to(REPO)).replace('\\','/')]=sha(p)
 for p,h in files.items():assert __import__('hashlib').sha256(subprocess.check_output(['git','show',source+':'+p],cwd=REPO)).hexdigest()==h,p
 root.mkdir(exist_ok=True);assert not (root/'plan.json').exists()
 plan=dict(source_commit=source,original_source_commit=old['source_commit'],source_files=files,runs=rows,data=old['data'],profile=cfg,pid=os.getpid(),timestamp=time.time(),A100_use_forbidden=True)
 save(root/'plan.json',plan);r=next(r for r in rows if r['experts']==16)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=r['training_gpu_uuid'],T10_SOURCE_COMMIT=source,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 log=open(root/'N16_resume_train.log','a');p=subprocess.Popen([sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.d2nn_aperture_migration','--root',str(root),'--resume-worker'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['train_pid']=p.pid
 save(root/'launch.json',plan)
 while True:
  states={r['experts']:read(Path(r['run_dir'])/'status.json') if (Path(r['run_dir'])/'status.json').exists() else {} for r in rows}
  failures=[n for n,s in states.items() if s.get('state')=='failed']
  if p.poll() not in [None,0]:failures.append(16)
  active=[r['experts'] for r in rows if states[r['experts']].get('state')!='complete']
  save(root/'status.json',dict(state='training_after_migration',active=active,failures=sorted(set(failures)),pid=os.getpid(),timestamp=time.time()))
  if failures:
   save(root/'status.json',dict(state='failed',failures=sorted(set(failures)),healthy_workers_not_terminated=True,pid=os.getpid()));return
  if not active and all(not command(r['train_pid']) for r in rows):break
  time.sleep(20)
 verify(plan);locks=[]
 for r in rows:
  out=Path(r['run_dir']);h=read(out/'history.json');res=read(out/'result.json');assert len(h)==60 and not (out/'test_result.json').exists()
  assert sha(out/'best_checkpoint.pt')==res['checkpoint_sha256'];locks.append(dict(run_dir=str(out),checkpoint_sha256=res['checkpoint_sha256']))
 save(root/'all_models_test_lock.json',dict(source_commit=source,runs=locks,timestamp=time.time(),selection='VAL macroNLL minimum EMA; all four completed and locked before TEST'))
 children=[]
 for r in rows:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=r['test_gpu_uuid'],T10_SOURCE_COMMIT=source)
  f=open(root/f'N{r["experts"]}_test.log','a')
  cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test','--run',r['run_dir'],'--data',plan['data'],'--selection-lock',str(root/'all_models_test_lock.json'),'--batch','2','--cache-locked-result']
  c=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);r['test_pid']=c.pid;children.append((c,f))
 save(root/'launch.json',plan);save(root/'status.json',dict(state='evaluating',pid=os.getpid(),timestamp=time.time()))
 codes=[c.wait() for c,f in children]
 for c,f in children:f.close()
 log.close();verify(plan)
 if not any(codes):save(root/'summary.json',dict(status='complete',source_commit=source,rows=[dict(**r,test=read(Path(r['run_dir'])/'test_result.json'),training=read(Path(r['run_dir'])/'result.json')) for r in rows],seed=17,not_three_seed_mean_or_SD=True,timestamp=time.time()))
 save(root/'status.json',dict(state='failed' if any(codes) else 'complete',test_returncodes=codes,pid=os.getpid(),all_children_exited=True,A100_unused=True,foreign_untouched=True,timestamp=time.time()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--resume-worker',action='store_true');a=ap.parse_args()
 if a.resume_worker:worker(a.root)
 else:scheduler(a.root)
if __name__=='__main__':main()
