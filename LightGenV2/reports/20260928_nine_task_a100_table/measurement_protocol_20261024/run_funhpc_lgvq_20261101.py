"""New authorized slow-host LGVQ queue: B1 full-input native DS/Kimi, then LG.

Order explicitly requested: both50 r1, LG r1, both100 r1, remaining repeats.
No training, precision change, synthetic warmup or old-run overwrite. Baseline
energy is not measured here. LG software timing and optical schedule separate.
"""
import argparse,csv,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path[:0]=[str(REPO),str(HERE)]
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
ROOT=TASK/'runs/simulation/funhpc_lgvq_e2e_20261101_01'
GPU='GPU-780326a6-79bc-7a45-36f6-683c5b278bee'
PIPE='/data/tmp/funhpc_lgvq_e2e_20261101_01'
MANIFEST=Path('/data/lgvq/original_manifest.csv')
MANIFEST_SHA='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
IDS_SHA='48b7fea193abc07f48bec9b091db0351b949387c9b44fbd228399ce3a03b655a'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=True,indent=2,allow_nan=False));tmp.replace(p)

def plan():
 first=[('deepseek',50,1),('kimi',50,1),('LG2',100,1),('deepseek',100,1),('kimi',100,1)]
 return first+[(m,c,r) for r in (2,3) for c in (50,100) for m in ('deepseek','kimi')]+[('LG2',100,2),('LG2',100,3)]

def name(point):
 m,c,r=point
 return f'{m}_cap{c}_repeat{r:02d}'

def timing(rows):
 if len(rows)!=512 or len({r['sample_id'] for r in rows})!=512:raise ValueError('Original512 unique calls')
 parts={k:0. for k in ['prepare_s','H2D_sync_s','model_front_s','first_block_to_score_s','CPU_return_s','internal_gap_s']}
 for i,r in enumerate(rows):
  ticks=[float(r[k]) for k in ['preprocess_start','prepare_end','model_call_start','first_block_start','gpu_output','call_return']]
  if not all(math.isfinite(x) for x in ticks) or not all(a<=b for a,b in zip(ticks,ticks[1:])):raise ValueError('Actual ordered boundaries required')
  for k,v in zip(list(parts)[:5],[b-a for a,b in zip(ticks,ticks[1:])]):parts[k]+=v
  if i:
   gap=ticks[0]-float(rows[i-1]['call_return'])
   if gap<0:raise ValueError('No overlapping B1 calls')
   parts['internal_gap_s']+=gap
 elapsed=float(rows[-1]['call_return'])-float(rows[0]['preprocess_start'])
 if not math.isclose(sum(parts.values()),elapsed,abs_tol=1e-6):raise ValueError('Wall partition must close')
 return dict(continuous_wall_s=elapsed,time16_ms=elapsed/32*1000,ms_per_video=elapsed/512*1000,videos=512,groups16=32,components=parts,
  timer_start='before first raw video preparation',timer_end='last CPU score returned',includes_all511_internal_gaps=True,weight_loading_excluded=True,energy_measured=False)

def sources():
 from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as core
 paths=[Path(__file__),HERE/'funhpc_lightgen_20261101.py',HERE/'serialized_mps_guard_20261029.py']
 return {**core.source_map(),**{p.relative_to(REPO).as_posix():sha(p) for p in paths}}

def verify_sources():
 commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'];bound=sources()
 for n,h in bound.items():
  if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n.replace('\\','/')],cwd=REPO,timeout=10)).hexdigest()!=h:raise ValueError('Published unchanged sources required: '+n)
 return dict(source_commit=commit,sources=bound)

def engine_context():
 from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
 engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=ROOT
 return engine

def guard(deadline):
 from serialized_mps_guard_20261029 import serialized_guard_class
 engine=engine_context()
 return serialized_guard_class(engine.MPSGuard)(SimpleNamespace(deadline=deadline))

def baseline(a):
 from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as core
 from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_dataset_once as legacy
 if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Original manifest identity')
 frozen=verify_sources();leaf=ROOT/name((a.model,a.cap,a.repeat));leaf.mkdir(exist_ok=False)
 legacy.select_profile('rentalb');legacy.frozen.A100_UUID=GPU
 engine_context();g=guard(a.deadline);records=[];evidence=[];finish=None
 def status(state,**kw):write(leaf/'status.json',dict(status=state,pid=os.getpid(),unix=time.time(),samples=len(records),**kw))
 try:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(a.cap):raise ValueError('Explicit MPS cap before CUDA')
  import torch,numpy as np,cv2
  g.check();torch.set_num_threads(8);cv2.setNumThreads(1);status('loading')
  _,reference=core.reference(a.model)
  pt=TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017'/a.model/'train/best_checkpoint.pt'
  if sha(pt)!=reference['checkpoint_sha256']:raise ValueError('Original head identity')
  args=SimpleNamespace(model=a.model,model_path=Path('/data/models')/a.model,checkpoint=pt,manifest=MANIFEST,cap=a.cap,repeat=a.repeat,batch=1,gpu_uuid=GPU,deadline=a.deadline)
  identity=dict(**frozen,model=a.model,cap=a.cap,repeat=a.repeat,batch=1,frames_per_video=4,gpu_uuid=GPU,explicit_warmup_forwards=0,manifest_sha256=MANIFEST_SHA,energy_measured=False)
  with torch.inference_mode():
   rows,prepare,transfer,forward,first_start,finish=legacy.runtime(args,g,identity)
   selected=rows[:512]
   if core.ordered_id_sha(selected)!=IDS_SHA:raise ValueError('Original ordered512 required')
   props=torch.cuda.get_device_properties(0)
   if props.multi_processor_count!=(54 if a.cap==50 else 108) or props.total_memory<70*1024**3:raise ValueError('Actual80GB MPS54/108 required')
   if identity['assets']!=reference['assets'] or identity['floating_parameter_dtypes']!=reference['floating_parameter_dtypes'] or identity['quality_readout_dtypes']!=reference['quality_readout_dtypes']:raise ValueError('Original assets and math precision')
   before=core.file_identities(selected)
   identity.update(cpu_model=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),torch_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads(),visible_SM=props.multi_processor_count,torch_version=torch.__version__,torch_cuda=torch.version.cuda)
   write(leaf/'identity.json',identity)
   core.collect(selected,prepare,transfer,forward,first_start,torch.cuda.synchronize,g.fast,records=records,progress=lambda n:status('evaluating',completed=n),input_observer=lambda r,c:evidence.append(dict(sample_id=r['sample_id'],**core.cpu_input_evidence(c))))
  cost=timing(records)
  if core.file_identities(selected)!=before:raise ValueError('Input metadata changed during measurement')
  inputs=core.file_identities(selected,hash_content=True)
  expected={x['sample_id']:x['sha256'] for x in read('/data/lgvq/data_transfer_manifest.json')['files']}
  if any(x['sha256']!=expected[k] for k,x in inputs.items()):raise ValueError('Original input bytes changed')
  finish();finish=None
  from LightGenV2.tasks.t06_video_quality_assessment.quality_token_common import metrics
  performance=metrics(np.array([r['target_mos'] for r in records]),np.array([r['prediction'] for r in records]))
  g.check();write(leaf/'input_audit.json',dict(samples=evidence,input_files=inputs))
  report=dict(status='complete',**identity,cost=cost,performance512=performance,effective_input_tokens512=sum(x.get('effective_input_tokens',0) for x in evidence))
  write(leaf/'report.json',report);status('complete')
  print(json.dumps(dict(point=leaf.name,status='complete',time16_ms=cost['time16_ms'],continuous_wall_s=cost['continuous_wall_s'],performance512=performance)),flush=True)
 except BaseException as e:status('failed',error=repr(e));raise
 finally:
  if finish:finish()
  with (leaf/'calls.csv').open('x',newline='') as f:
   if records:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  g.close(leaf/'process_audit.json')

def suite(a):
 import fcntl
 from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
 frozen=verify_sources();ROOT.mkdir(exist_ok=False);lock=(TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a');active=False;child=None
 state=dict(status='admission',pid=os.getpid(),deadline=a.deadline,plan=[name(p) for p in plan()],completed=[],**frozen)
 def save():write(ROOT/'suite_status.json',state)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save();engine=engine_context()
  for _ in range(3):
   if engine.base.guard_records(GPU):raise RuntimeError('Foreign GPU process; no retry')
   validate_idle_memory(GPU);time.sleep(1)
  Path(PIPE).parent.mkdir(parents=True,exist_ok=True);engine.daemon(True);active=True
  for m,c,r in plan():
   if time.time()>=a.deadline-120:raise TimeoutError('Fixed overall deadline; no automatic extension')
   py='/data/venvs/'+('kimi' if m=='LG2' else m)+'/bin/python'
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(c),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
   if m=='LG2':env['PYTHONPATH']='/data/qwen_deps:'+str(REPO)
   cmd=[py,'-u',str(Path(__file__)),'--phase','lg' if m=='LG2' else 'baseline','--model',m,'--cap',str(c),'--repeat',str(r),'--deadline',str(a.deadline)]
   with (ROOT/(name((m,c,r))+'.log')).open('x') as f:
    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(status='running',phase=name((m,c,r)),child_pid=child.pid,phase_started=time.time());save()
    try:rc=child.wait(timeout=max(1,a.deadline-time.time()))
    except BaseException:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise
   if rc:raise RuntimeError('Point stopped, preserve evidence; no blind retry: '+state['phase'])
   state['completed'].append(state['phase']);save()
  state.update(status='complete',finished=time.time())
 except BaseException as e:state.update(status='stopped',error=repr(e));raise
 finally:
  if active:engine_context().daemon(False);state['mps_released']=True
  save();lock.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','baseline','lg'],required=True);p.add_argument('--model',choices=['deepseek','kimi','LG2']);p.add_argument('--cap',type=int,choices=[50,100]);p.add_argument('--repeat',type=int,choices=[1,2,3]);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
 if not 0<a.deadline-time.time()<=14400:raise ValueError('Finite maximum4h, no auto-renew')
 if a.phase=='suite':suite(a)
 elif a.phase=='baseline':baseline(a)
 else:
  from funhpc_lightgen_20261101 import worker
  worker(a)
if __name__=='__main__':main()
