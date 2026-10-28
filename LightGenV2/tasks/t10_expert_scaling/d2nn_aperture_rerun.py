"""Four single-seed D2NN runs with every phase covering the LG global aperture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parent
REPO=TASK.parents[2]
PROFILE=TASK/'configs/kather_d2nn_same_aperture_s17_20261029.json'
DEPENDENCIES=['d2nn_aperture_rerun.py','model.py','train.py','plan.py','configs/study.json',
              'evaluate_test.py','locked_test_cache.py','configs/kather_d2nn_same_aperture_s17_20261029.json']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def geometry(n):
 g=int(n**.5)*224+(int(n**.5)-1)*30
 return dict(experts=n,global_side=g,expert_layer_pixels=n*224**2,global_layer_pixels=g*g,
             LG_parameters=224**2+2*n*224**2+2*g*g,D2NN_parameters=4*g*g)
def verify(plan):
 for path,h in plan['source_files'].items():assert sha(REPO/path)==h,path

def worker(args):
 import torch
 from . import train
 plan=read(args.root/'plan.json');verify(plan);row=next(r for r in plan['runs'] if r['experts']==args.experts)
 out=Path(row['run_dir']);out.mkdir(exist_ok=False)
 assert os.environ['CUDA_VISIBLE_DEVICES']==row['gpu_uuid']
 assert torch.cuda.get_device_name() in ['NVIDIA A100-PCIE-40GB','NVIDIA GeForce RTX 4090']
 save(out/'run_identity.json',dict(source_commit=plan['source_commit'],source_files=plan['source_files'],pid=os.getpid(),
     expected=row,host_head=plan['host_head'],metadata_git_commit_scope='train.py records host HEAD; this identity records exact published source overlay',timestamp=time.time()))
 a=argparse.Namespace(data=Path(plan['data']),out=out,arch='d2nn_same_aperture',experts=args.experts,top_k=args.experts,
                      layers=4,epochs=60,lr=.002,seed=17,microbatch=2,padding=2,smoke_limit=0,resume=False,profile=None)
 try:
  train.run(a)
  meta=read(out/'metadata.json');assert meta['parameter_count']==row['D2NN_parameters'] and meta['geometry']['active_side_px']==row['global_side']
  assert meta['test_read'] is False and len(read(out/'history.json'))==60
 except BaseException:
  import traceback
  save(out/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
 finally:torch.cuda.empty_cache()

def scheduler(args):
 profile=read(PROFILE);source=os.environ['T10_D2NN_APERTURE_SOURCE_COMMIT']
 sources={str((TASK/n).relative_to(REPO)).replace('\\','/'):sha(TASK/n) for n in DEPENDENCIES}
 sources['LightGenV2/common/optical_contract.py']=sha(REPO/'LightGenV2/common/optical_contract.py')
 for p,h in sources.items():assert hashlib.sha256(subprocess.check_output(['git','show',source+':'+p],cwd=REPO)).hexdigest()==h,p
 assert sha(profile['data'])==profile['data_sha256'] and sha(Path(profile['data']).parent/'data_manifest.json')==profile['manifest_sha256']
 # Explicit GPU inventory and capacity admission; no foreign process is modified.
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 free={l.split(',')[0].strip():int(l.split(',')[1]) for l in inventory.splitlines()}
 assert free[profile['assignments']['4']]>=24000
 assert all(free[profile['assignments'][str(n)]]>=12000 for n in [25,49])
 args.root.mkdir(exist_ok=False)
 rows=[dict(**geometry(n),gpu_uuid=profile['assignments'][str(n)],seed=17,run_dir=str(args.root/f'd2nn_same_aperture_N{n}_s17')) for n in [4,16,25,49]]
 plan=dict(source_commit=source,host_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           source_files=sources,data=profile['data'],profile=profile,runs=rows,pid=os.getpid(),timestamp=time.time())
 save(args.root/'plan.json',plan);children={};handles=[];failures=[]
 def launch(row,phase):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=row['gpu_uuid'],T10_SOURCE_COMMIT=source,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  if phase=='train':cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.d2nn_aperture_rerun','--worker','--experts',str(row['experts']),'--root',str(args.root)]
  else:cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test','--run',row['run_dir'],'--data',plan['data'],'--selection-lock',str(args.root/'all_models_test_lock.json'),'--batch','2','--cache-locked-result']
  log=open(args.root/f'N{row["experts"]}_{phase}.log','a');handles.append(log)
  p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row[phase+'_pid']=p.pid;row[phase+'_command']=cmd;return p
 for row in rows:children[row['experts']]=launch(row,'train')
 save(args.root/'launch.json',plan)
 while children:
  for n,p in list(children.items()):
   rc=p.poll()
   if rc is not None:
    if rc:failures.append(dict(experts=n,phase='train',returncode=rc))
    del children[n]
  save(args.root/'status.json',dict(state='training',active=list(children),failures=failures,timestamp=time.time(),pid=os.getpid()))
  if children:time.sleep(20)
 if failures:save(args.root/'status.json',dict(state='failed',failures=failures,pid=os.getpid()));return
 verify(plan)
 locks=[]
 for row in rows:
  out=Path(row['run_dir']);result=read(out/'result.json');assert read(out/'status.json')['state']=='complete'
  assert len(read(out/'history.json'))==60 and not (out/'test_result.json').exists()
  assert sha(out/'best_checkpoint.pt')==result['checkpoint_sha256']
  locks.append(dict(run_dir=str(out),checkpoint_sha256=result['checkpoint_sha256']))
 save(args.root/'all_models_test_lock.json',dict(source_commit=source,runs=locks,timestamp=time.time(),selection='VAL macroNLL minimum EMA; all four models locked before TEST'))
 children={r['experts']:launch(r,'test') for r in rows};save(args.root/'launch.json',plan)
 while children:
  for n,p in list(children.items()):
   rc=p.poll()
   if rc is not None:
    if rc:failures.append(dict(experts=n,phase='test',returncode=rc))
    del children[n]
  save(args.root/'status.json',dict(state='evaluating',active=list(children),failures=failures,pid=os.getpid(),timestamp=time.time()))
  if children:time.sleep(20)
 verify(plan)
 results=[]
 if not failures:
  for row in rows:
   test=read(Path(row['run_dir'])/'test_result.json');results.append(dict(**row,test=test,training=read(Path(row['run_dir'])/'result.json')))
  save(args.root/'summary.json',dict(status='complete',source_commit=source,rows=results,seed=17,not_three_seed_mean_or_SD=True,timestamp=time.time()))
 for h in handles:h.close()
 save(args.root/'status.json',dict(state='failed' if failures else 'complete',failures=failures,pid=os.getpid(),timestamp=time.time(),all_children_exited=True,foreign_untouched=True))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--worker',action='store_true');ap.add_argument('--experts',type=int);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 if a.check:
  p=read(PROFILE);assert p['seed']==17 and p['epochs']==60 and p['layers']==4
  for n in [4,16,25,49]:
   g=geometry(n);assert g['D2NN_parameters']>g['LG_parameters'];assert g['global_side']==p['geometry'][str(n)]['global_side'];assert g==p['geometry'][str(n)]
  print(json.dumps(p['geometry']));return
 if a.worker:worker(a)
 else:scheduler(a)
if __name__=='__main__':main()
