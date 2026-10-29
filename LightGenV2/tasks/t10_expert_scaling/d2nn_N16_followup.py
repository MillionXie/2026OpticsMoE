"""One bounded TRAIN-only continuation of the aperture-matched N16 seed17."""
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
PROFILE=TASK/'configs/kather_d2nn_N16_followup_20261029.json'
DEPENDENCIES=['d2nn_N16_followup.py','configs/kather_d2nn_N16_followup_20261029.json',
 'diagnose_topk.py','train.py','model.py','plan.py','configs/study.json','evaluate_test.py','locked_test_cache.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def command(pid):
 p=Path('/proc',str(pid),'cmdline');return p.read_bytes().replace(b'\0',b' ').decode(errors='replace') if p.exists() else ''
def target_met(correct,c):
 assert type(correct) is int and 0<=correct<=752
 return (correct-c['N4_TEST_correct'])*100>=c['minimum_margin_above_N4_pp']*752 and (c['N25_TEST_correct']-correct)*100>=c['minimum_margin_below_N25_pp']*752
def validate(c):
 assert c['source_epochs']==60 and c['extra_epochs']==30 and c['lr_start']==.0002 and c['lr_end']==.00002
 assert c['microbatch']==2 and c['TEST_gradient_inputs']==0 and c['A100_forbidden']
 assert c['gpu_index']==1 and c['gpu_uuid']=='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'
 assert [v for v in range(753) if target_met(v,c)]==[461,462]
 assert c['target_correct_min']==461 and c['target_correct_max']==462
 assert c['selection']=='VAL_macro_NLL_minimum_EMA_including_source_best'

def verify(files):
 for p,h in files.items():assert sha(REPO/p)==h,p

def admission(c):
 import torch
 torch.set_num_threads(2)
 source=Path(c['source']);m=read(source/'metadata.json');r=read(source/'result.json');h=read(source/'history.json');t=read(source/'test_result.json')
 assert [v['epoch'] for v in h]==list(range(1,61)) and m['arguments']['arch']=='d2nn_same_aperture'
 assert m['arguments']['experts']==m['arguments']['top_k']==16 and m['arguments']['seed']==17 and m['arguments']['layers']==4
 assert m['parameter_count']==3888784 and m['geometry']['active_side_px']==986 and m['test_read'] is False
 assert r['best_epoch']==min(h,key=lambda v:v['val']['macro_nll'])['epoch']==59
 assert sha(source/'test_predictions.npz')==c['source_prediction_sha256']==t['predictions_sha256']
 assert sum(t['test']['confusion_matrix'][i][i] for i in range(8))==457
 assert sha(Path(m['arguments']['data']))==c['data_sha256'] and sha(Path(m['arguments']['data']).parent/'data_manifest.json')==c['manifest_sha256']
 audit={}
 for name,key in [('best_checkpoint.pt','source_best_sha256'),('last_checkpoint.pt','source_last_sha256')]:
  assert sha(source/name)==c[key]
  v=torch.load(source/name,map_location='cpu',weights_only=False)
  for field in ['model','live_model']:
   assert len(v[field])==4 and all(list(t.shape)==[986,986] and torch.isfinite(t).all() for t in v[field].values())
  steps=[]
  for s in v['optimizer']['state'].values():
   assert all(torch.isfinite(x).all() for x in s.values() if isinstance(x,torch.Tensor));steps.append(int(s['step'].item()))
  assert steps==[v['epoch']*219]*4
  audit[name]=dict(sha256=sha(source/name),epoch=v['epoch'],Adam_steps=steps,all_tensors_finite=True)
  if name=='last_checkpoint.pt':assert v['epoch']==60 and steps==[13140]*4
  del v
 assert not torch.cuda.is_initialized()
 return dict(checkpoints=audit,source_metadata_sha256=sha(source/'metadata.json'),source_history_sha256=sha(source/'history.json'),source_result_sha256=sha(source/'result.json'),source_prediction_sha256=sha(source/'test_predictions.npz'),CPU_only_no_forward=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--check',action='store_true');a=ap.parse_args();c=read(PROFILE);validate(c)
 if a.check:print(json.dumps(dict(target_correct=[461,462],extra_epochs=30,TEST_gradient_inputs=0)));return
 source_commit=os.environ['T10_D2NN_N16_FOLLOWUP_SOURCE_COMMIT']
 files={p.relative_to(REPO).as_posix():sha(p) for p in [*[TASK/n for n in DEPENDENCIES],REPO/'LightGenV2/common/optical_contract.py']}
 for p,h in files.items():assert hashlib.sha256(subprocess.check_output(['git','show',source_commit+':'+p],cwd=REPO)).hexdigest()==h,p
 audit=admission(c);root=a.root.resolve();root.mkdir(parents=True,exist_ok=False);out=root/'N16_s17'
 plan=dict(source_commit=source_commit,source_files=files,profile=c,CPU_source_admission=audit,source_host_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip(),pid=os.getpid(),out=str(out),timestamp=time.time())
 save(root/'plan.json',plan)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',T10_SOURCE_COMMIT=source_commit,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 def available():
  inventory=subprocess.check_output(['nvidia-smi','-i',str(c['gpu_index']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert inventory[0].strip()==c['gpu_uuid']
  compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  own=[]
  for line in compute.splitlines():
   if not line.strip():continue
   pid,uuid=line.split(',',1)
   if uuid.strip()==c['gpu_uuid'] and 't10_expert_scaling' in command(int(pid)):own.append(int(pid))
  return int(inventory[1])>=c['minimum_free_mib'] and not own
 while not available():save(root/'status.json',dict(state='waiting_for_authorized_GPU1',pid=os.getpid(),timestamp=time.time()));time.sleep(20)
 verify(files)
 cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.diagnose_topk','--source',c['source'],'--out',str(out),'--epochs','30','--lr','0.0002','--batch','2','--selection-metric','macro_nll','--defer-test']
 log=(root/'train.log').open('x');child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 plan.update(train_pid=child.pid,train_command=cmd);save(root/'launch.json',plan)
 while child.poll() is None:
  status=read(out/'status.json') if (out/'status.json').exists() else {}
  save(root/'status.json',dict(state='training',pid=os.getpid(),train_pid=child.pid,worker=status,timestamp=time.time()));time.sleep(20)
 rc=child.wait();log.close()
 if rc:save(root/'status.json',dict(state='failed',phase='train',returncode=rc,pid=os.getpid()));return
 verify(files);assert not command(child.pid);assert read(out/'status.json')['state']=='validation_complete_test_deferred'
 assert [v['epoch'] for v in read(out/'history.json')]==list(range(61,91)) and not (out/'test_result.json').exists()
 result=read(out/'result.json');assert sha(out/'best_checkpoint.pt')==result['checkpoint_sha256']
 lock=out/'test_selection_lock.json';assert read(lock)['runs'][0]['checkpoint_sha256']==result['checkpoint_sha256']
 # The source best remains eligible; its already locked TEST is reused if selected.
 if result['checkpoint_sha256']==c['source_best_sha256']:
  import shutil
  for name in ['test_predictions.npz','test_result.json']:shutil.copy2(Path(c['source'])/name,out/name)
  save(out/'TEST_reuse_receipt.json',dict(reused_without_inference=True,source=c['source'],original_selection_lock_sha256=read(out/'test_result.json')['selection_lock_sha256'],new_selection_lock_sha256=sha(lock)))
  plan['test_pid']=None
 else:
  while not available():save(root/'status.json',dict(state='waiting_for_TEST_GPU1',pid=os.getpid(),timestamp=time.time()));time.sleep(20)
  cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test','--run',str(out),'--data',read(out/'metadata.json')['arguments']['data'],'--selection-lock',str(lock),'--batch','2','--cache-locked-result']
  log=(root/'test.log').open('x');child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);plan.update(test_pid=child.pid,test_command=cmd);save(root/'launch.json',plan)
  save(root/'status.json',dict(state='evaluating_locked_TEST',pid=os.getpid(),test_pid=child.pid,timestamp=time.time()));rc=child.wait();log.close()
  if rc:save(root/'status.json',dict(state='failed',phase='test',returncode=rc,pid=os.getpid()));return
 verify(files);test=read(out/'test_result.json');correct=sum(test['test']['confusion_matrix'][i][i] for i in range(8));assert sum(map(sum,test['test']['confusion_matrix']))==752
 assert sha(Path(c['source'])/'best_checkpoint.pt')==c['source_best_sha256'] and sha(Path(c['source'])/'last_checkpoint.pt')==c['source_last_sha256']
 summary=dict(status='complete',seed=17,extra_epochs=30,total_epochs=90,best_epoch=result['best_epoch'],TEST_correct=correct,source_TEST_correct=457,paired_N4_correct=457,paired_N25_correct=464,target_interval_met=target_met(correct,c),margin_above_N4_pp=100*(correct-457)/752,margin_below_N25_pp=100*(464-correct)/752,TEST_gradient_inputs=0,extra_TRAIN_gradient_inputs=104880,extra_updates=6570,TEST_used_to_select_checkpoint=False,role=c['role'],out=str(out),source_commit=source_commit,TEST=test,result=result)
 save(root/'summary.json',summary);save(root/'launch.json',plan);save(root/'resource_release.json',dict(all_children_exited=True,foreign_untouched=True,A100_unused=True,timestamp=time.time()));save(root/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time()))

if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  if '--root' in sys.argv:
   root=Path(sys.argv[sys.argv.index('--root')+1])
   if root.exists():save(root/'status.json',dict(state='failed',phase='coordinator',pid=os.getpid(),traceback=traceback.format_exc(),foreign_untouched=True))
  raise
