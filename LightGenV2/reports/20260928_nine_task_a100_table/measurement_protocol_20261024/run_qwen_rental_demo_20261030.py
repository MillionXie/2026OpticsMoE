"""One frozen Qwen3-VL LGVQ temporal demo, original512/B1/four frames.

No training, seed search, warmup, energy measurement or changes to arithmetic.
Full file-to-CPU wall and narrower first-TF diagnostic are kept distinct.
Neither is relabelled as matched-scope physical LightGen speedup.
"""
import argparse,csv,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
ROOT=TASK/'runs/simulation/qwen_rental_demo_20261030_01'
GPU='GPU-30fecb94-0222-a3d5-d952-a91f2cab2f43'
PIPE='/data/tmp/qwen_rental_demo_20261030_01'
ASSETS=Path('/root/data/lgvq_20261029/qwen_demo_assets_20261030_01')
MANIFEST=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
REFERENCE=ASSETS/'original_reference_report.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=True,indent=2,allow_nan=False));tmp.replace(p)
def source_map():
 from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as core
 paths=[Path(__file__),HERE/'serialized_mps_guard_20261029.py']
 return {**core.source_map(),**{p.relative_to(REPO).as_posix():sha(p) for p in paths}}
def verify_sources():
 commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'];sources=source_map()
 for name,digest in sources.items():
  b=subprocess.check_output(['git','show',commit+':'+name.replace('\\','/')],cwd=REPO,timeout=10)
  if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Unpublished science source: '+name)
 return dict(published_commit=commit,sources=sources)
def check_assets():
 if sha(REFERENCE)!='d73c648a40290655051ea7487a797eb9677990ecaa2f25cb199e7f1062f37a31':raise ValueError('Original Qwen reference changed')
 ref=json.loads(REFERENCE.read_text())
 for name,h in ref['assets'].items():
  if sha(ASSETS/name)!=h:raise ValueError('Original Qwen asset differs: '+name)
 if sha(ASSETS/'best_checkpoint.pt')!=ref['checkpoint_sha256']:raise ValueError('Original frozen head differs')
 if sha(MANIFEST)!='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc':raise ValueError('Original manifest differs')
 return ref
def timing(records):
 if len(records)!=512 or len({r['sample_id'] for r in records})!=512:raise ValueError('Exactly512 ordered calls required')
 for i,r in enumerate(records):
  p,d,a,s,e,z=[float(r[k]) for k in ['preprocess_start','prepare_end','model_call_start','first_block_start','gpu_output','call_return']]
  if not p<=d<=a<=s<e<=z or (i and float(records[i-1]['call_return'])>p):raise ValueError('Invalid original serial boundaries')
 full=float(records[-1]['call_return'])-float(records[0]['preprocess_start'])
 tf=sum(float(r['legacy_host_ms']) for r in records)/1000
 return dict(videos=512,groups16=32,file_to_CPU_s=full,file_to_CPU_ms16=full/32*1000,
             first_TF_to_score_work_ms16=tf/32*1000,energy_measured=False,
             first_TF_scope_excludes_frontend_and_inter_request_gaps=True,matched_LG_compute_scope=False)
def worker(a):
 ref=check_assets();frozen=verify_sources();leaf=ROOT/'qwen_cap50_repeat01';leaf.mkdir(exist_ok=False)
 from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as core
 from LightGenV2.tasks.t06_video_quality_assessment import evaluate_temporal_baseline_batches as native
 from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
 from serialized_mps_guard_20261029 import serialized_guard_class
 import torch
 import numpy as np
 import cv2
 engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=ROOT
 guard=serialized_guard_class(engine.MPSGuard)(SimpleNamespace(deadline=a.deadline));timer=None;records=[];evidence=[]
 def status(state,**kw):write(leaf/'status.json',dict(status=state,pid=os.getpid(),unix=time.time(),samples=len(records),**kw))
 try:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='50':raise ValueError('Explicit GPU50 required before CUDA')
  guard.check();status('loading');torch.set_num_threads(8);cv2.setNumThreads(1)
  args=SimpleNamespace(baseline='qwen',model=ASSETS,checkpoint=ASSETS/'best_checkpoint.pt',manifest=MANIFEST,image_size=448)
  model,head,rows,prepare,transfer,forward,timer,handles=native.load_runtime(args)
  selected=rows[:512]
  if core.ordered_id_sha(selected)!='48b7fea193abc07f48bec9b091db0351b949387c9b44fbd228399ce3a03b655a':raise ValueError('Original512 order differs')
  if torch.cuda.get_device_properties(0).multi_processor_count!=54:raise ValueError('Actual50 allocation not54SM')
  guard.bind();before=core.file_identities(selected)
  identity=dict(**frozen,model='Qwen3-VL-2B-Instruct',target='LGVQ temporal',repeat=1,batch=1,frames_per_video=4,
   checkpoint_sha256=ref['checkpoint_sha256'],assets=ref['assets'],gpu_uuid=GPU,cap=50,
   no_training_or_seed_search=True,explicit_warmup_forwards=0,torch_version=torch.__version__,cuda=torch.version.cuda,
   cpu_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads(),model_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
  write(leaf/'identity.json',identity)
  with torch.inference_mode():
   core.collect(selected,prepare,transfer,forward,lambda:timer.state['vision_host_started'],torch.cuda.synchronize,guard.fast,
    records=records,progress=lambda n:status('evaluating',completed=n),
    input_observer=lambda r,c:evidence.append(dict(sample_id=r['sample_id'],**core.cpu_input_evidence(c))))
  cost=timing(records)
  with (ASSETS/'original_reference_calls.csv').open(newline='',encoding='utf-8-sig') as f:old_calls=list(csv.DictReader(f))[:512]
  if len(old_calls)!=512 or any(x['sample_id']!=y['sample_id'] or json.loads(x['frame_positions'])!=json.loads(y['frame_positions']) for x,y in zip(records,old_calls)):
   raise ValueError('Original four-frame selection differs')
  if core.file_identities(selected)!=before:raise ValueError('Original input metadata changed')
  after=core.file_identities(selected,hash_content=True)
  storage=json.loads(Path('/root/data/lgvq_20261029/input_local_overlay_receipt01.json').read_text())
  for old in storage['records']:
   if after[old['sample_id']]['sha256']!=old['sha256']:raise ValueError('Original input bytes differ')
  from LightGenV2.tasks.t06_video_quality_assessment.quality_token_common import metrics
  performance=metrics(np.array([r['target_mos'] for r in records]),np.array([r['prediction'] for r in records]))
  write(leaf/'input_audit.json',dict(samples=evidence,input_files=after))
  guard.check();check_assets()
  write(leaf/'report.json',dict(status='complete',**identity,cost=cost,performance512=performance,
   test_subset_not_full558=True,energy_qualified=False,effective_tokens=sum(x['effective_input_tokens'] for x in evidence)))
  status('complete')
 except BaseException as e:status('failed',error=repr(e));raise
 finally:
  if timer:timer.close()
  with (leaf/'calls.csv').open('x',newline='') as f:
   if records:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  guard.close(leaf/'process_audit.json')
def suite(a):
 import fcntl
 from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
 check_assets();frozen=verify_sources();ROOT.mkdir(exist_ok=False)
 lock=(TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a');child=None;active=False
 state=dict(status='admission',pid=os.getpid(),deadline=a.deadline,plan=['qwen_cap50_repeat01'],**frozen)
 engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=ROOT
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for _ in range(3):
   if engine.base.guard_records(GPU):raise RuntimeError('Foreign GPU process; no automatic retry')
   from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
   validate_idle_memory(GPU);time.sleep(1)
  Path(PIPE).parent.mkdir(exist_ok=True,parents=True);engine.daemon(True);active=True
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50',
   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
  cmd=[sys.executable,'-u',str(Path(__file__)),'--phase','worker','--deadline',str(a.deadline)]
  with (ROOT/'qwen_cap50_repeat01.log').open('x') as f:
   child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   state.update(status='running',child_pid=child.pid);write(ROOT/'suite_status.json',state)
   try:rc=child.wait(timeout=max(1,a.deadline-time.time()))
   except BaseException:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise
  if rc:raise RuntimeError('One demo failed; retain evidence, no retry')
  state.update(status='complete',finished=time.time())
 except BaseException as e:state.update(status='stopped',error=repr(e));raise
 finally:
  if active:engine.daemon(False);state['mps_released']=True
  write(ROOT/'suite_status.json',state);lock.close()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['check','suite','worker'],required=True);p.add_argument('--deadline',type=float)
 a=p.parse_args()
 if a.phase=='check':print(json.dumps(dict(status='CPU_asset_contract_passed',reference=check_assets()['checkpoint_sha256'])));return
 if a.deadline is None or not 0<a.deadline-time.time()<=3600:raise ValueError('Finite demo at most60min')
 (suite if a.phase=='suite' else worker)(a)
if __name__=='__main__':main()
