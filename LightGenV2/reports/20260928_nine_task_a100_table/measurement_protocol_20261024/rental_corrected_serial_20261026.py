"""New authorized A10080 rental: original DS/Kimi B1, current Qwen boundary.

One fresh process per model, 512 original TEST videos cycled four times. Every
video is prepared and transferred serially; remove each image preparation/H2D
span while keeping actual CPU tokenizer intervals and all other internal waits.
No Graph, prefetch, training, warmup, altered native forward or power mutation.
"""
import argparse,csv,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
ROOT=TASK/'runs/simulation/a10080_corrected_serial_20261026_01'
GPU='GPU-ecac2c49-306a-9ec5-1572-8fcd2d232165'
PIPE='/data/tmp/a10080_corrected_serial_20261026_01'
N=2048
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
def sources():
 from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
 return dict(engine.sources(),**{Path(__file__).relative_to(REPO).as_posix():sha(__file__)})
def control(command):
 return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,timeout=10,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=PIPE))
def gpu_processes():
 raw=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
 return [r for r in csv.reader(raw.splitlines()) if len(r)==3 and r[1].strip()==GPU]
def union_ms(intervals):
 total=0.;end=-float('inf')
 for r in sorted(intervals,key=lambda r:r['start']):
  assert r['start']<=r['end'];total+=max(0.,r['end']-max(end,r['start']));end=max(end,r['end'])
 return total*1000
def account(rows):
 assert len(rows)==N
 removed=sum((r['H2D_end']-r['CPU_start'])*1000-union_ms(r['tokenizer_intervals']) for r in rows)
 full=(rows[-1]['GPU_ready_host']-rows[0]['CPU_start'])*1000
 assert full>removed>0
 return full-removed,removed,full
class TokenizerAudit:
 def __init__(self,inner,records):self.inner=inner;self.records=records
 def timed(self,name,fn,*a,**kw):
  start=time.perf_counter()
  try:return fn(*a,**kw)
  finally:self.records.append(dict(method=name,start=start,end=time.perf_counter()))
 def __call__(self,*a,**kw):return self.timed('__call__',self.inner,*a,**kw)
 def __getattr__(self,name):
  value=getattr(self.inner,name)
  if name in ('encode','encode_plus','batch_encode_plus','tokenize','apply_chat_template') and callable(value):return lambda *a,**kw:self.timed(name,value,*a,**kw)
  return value
def bind_engine():
 from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
 engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=ROOT
 return engine
def worker(args):
 import numpy as np
 import torch
 from LightGenV2.tasks.t06_video_quality_assessment import rental_precision_recheck as native
 from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_batch_followup as prepare_native
 from LightGenV2.tasks.t06_video_quality_assessment import rental_vlm_temporal as old
 assert os.environ['CUDA_VISIBLE_DEVICES']==GPU and os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']=='50'
 output=args.output;assert not output.exists();output.mkdir()
 bound=sources();engine=bind_engine()
 class Guard(engine.MPSGuard):
  def check(self):
   assert sources()==bound,'Source changed';super().check()
 guard=Guard(args);records=[];tokenizer=[];predictions=[]
 identity=dict(model=args.model,cap=50,batch=1,video_calls=N,unique_videos=512,cycles=4,explicit_warmup_forwards=0,
  gpu_uuid=GPU,command=sys.argv,pid=os.getpid(),sources=bound,published_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],
  actual_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
 write(output/'identity.json',identity)
 def status(value,**kw):write(output/'status.json',dict(status=value,pid=os.getpid(),unix=time.time(),samples=len(records),total=N,**kw))
 try:
  guard.check();torch.set_num_threads(8);status('loading_original_native_model')
  options=SimpleNamespace(model=args.model,precision='native',dual_fp32=False,gpu_uuid=GPU)
  model,processor,prompt,first,transfer,compute=native.runtime(options,SimpleNamespace(bind=lambda:None),identity)
  guard.bind();props=torch.cuda.get_device_properties(0)
  assert props.multi_processor_count==54 and props.total_memory>=70*1024**3 and 'A100' in props.name
  limit=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True));assert limit==400
  identity.update(visible_multiprocessors=props.multi_processor_count,gpu_name=props.name,power_limit_w=limit,
   original_math_unchanged=True,full_native_vocabulary_projection_and_FP32_quality_head=True,
   no_graph=True,no_prefetch=True,CPU_preparation_single_video=True)
  write(output/'identity.json',identity)
  all_rows=[r for r in old.rows() if r['split']=='test'];assert len(all_rows)==558
  selected=all_rows[:512];processor.tokenizer=TokenizerAudit(processor.tokenizer,tokenizer)
  with torch.inference_mode():
   for index in range(N):
    guard.fast();row=selected[index%512];a=time.perf_counter();lo=len(tokenizer)
    cpu,positions=prepare_native.prepare_batch(args.model,[row],processor,prompt);b=time.perf_counter()
    inputs=transfer(cpu);torch.cuda.synchronize();c=time.perf_counter()
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();h=time.perf_counter()
    mos,logits=compute(inputs);end.record();end.synchronize();ready=time.perf_counter()
    prediction=mos.detach().float().cpu().tolist();returned=time.perf_counter()
    assert len(prediction)==1 and math.isfinite(prediction[0]);predictions.append(prediction[0])
    intervals=tokenizer[lo:];assert all(a<=t['start']<=t['end']<=b for t in intervals)
    evidence=dict(index=index,source_index=index%512,cycle=index//512,sample_id=row['sample_id'],CPU_start=a,CPU_end=b,H2D_end=c,
     GPU_host_start=h,GPU_ready_host=ready,CPU_result_return=returned,GPU_model_cuda_ms=begin.elapsed_time(end),
     tokenizer_intervals=intervals,tokenizer_CPU_ms=union_ms(intervals),prediction=prediction[0],frame_positions=positions[0],
     input_shapes={k:list(v.shape) for k,v in cpu.items()},input_ids=cpu['input_ids'].tolist(),attention_mask=cpu['attention_mask'].tolist())
    assert a<=b<=c<=h<=ready<=returned
    if records:assert a>=records[-1]['CPU_result_return']
    records.append(evidence);del cpu,inputs,mos,logits
    if (index+1)%32==0:status('evaluating_original_strict_B1');print(f'{args.model} {index+1}/{N}',flush=True)
  corrected,removed,full=account(records)
  array=np.asarray(predictions,dtype=np.float32);assert array.shape==(N,) and np.isfinite(array).all()
  assert all(np.array_equal(array[:512],array[k:k+512]) for k in range(512,N,512))
  np.savez_compressed(output/'outputs.npz',prediction=array)
  result=dict(status='complete',model=args.model,video_calls=N,unique_videos=512,cycles=4,cap=50,rounds=1,
   corrected_GPU_ready_ms=corrected,removed_image_preparation_and_H2D_ms=removed,full_before_subtraction_ms=full,
   corrected_CPU_result_ms=corrected+(records[-1]['CPU_result_return']-records[-1]['GPU_ready_host'])*1000,
   GPU_model_cuda_sum_ms=sum(r['GPU_model_cuda_ms'] for r in records),tokenizer_CPU_ms=sum(r['tokenizer_CPU_ms'] for r in records),
   time16_ms=corrected*16/N,equivalent_frame_ms=corrected/(4*N),records=records,
   accounting_matches_current_Qwen=True,every_video_image_preparation_and_original_H2D_excluded=True,
   other_dispatch_sync_return_waits_retained=True,no_fast_window_selection=True,
   default_original_forward=True,no_graph=True,no_prefetch=True,CPU_preparation_single_video=True)
  write(output/'timing.json',result)
  write(output/'input_files.json',[dict(sample_id=r['sample_id'],video_path=str(r['video_path']),bytes=Path(r['video_path']).stat().st_size,sha256=sha(r['video_path'])) for r in selected])
  assert old.verify_assets(Path('/data/models')/args.model)==identity['assets'],'Official assets changed'
  assert sources()==bound;guard.check();status('complete')
 except BaseException as error:status('failed',error=repr(error));write(output/'partial_records.json',records);raise
 finally:guard.close(output/'process_audit.json');write(output/'identity.json',identity)
def suite(args):
 import fcntl
 assert not ROOT.exists();ROOT.mkdir(parents=True);child=None;active=False
 state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,plan=['kimi','deepseek'],completed=[],N=N,cap=50,rounds_per_model=1,gpu_uuid=GPU,sources=sources())
 def save():write(ROOT/'suite_status.json',state)
 save()
 try:
  with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for _ in range(12):
    assert not gpu_processes(),'Foreign GPU process';time.sleep(.25)
   assert control('get_server_list').returncode!=0
   Path(PIPE).mkdir(parents=True,exist_ok=False);(ROOT/'mps_logs').mkdir()
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_LOG_DIRECTORY=str(ROOT/'mps_logs'),CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
   subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True);active=True
   for model in state['plan']:
    assert time.time()<args.deadline-180,'Preserve backup time';assert sources()==state['sources']
    leaf=ROOT/(model+'50_repeat01');command=['/data/venvs/'+model+'/bin/python','-u',str(Path(__file__)),'--phase','worker','--model',model,'--output',str(leaf),'--deadline',str(args.deadline)]
    write(ROOT/(model+'.command.json'),command)
    with (ROOT/(model+'.log')).open('x') as log:
     child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     state.update(status='running',model=model,child=child.pid);save()
     while child.poll() is None:
      if time.time()>args.deadline-120:raise TimeoutError('Reserve final backup time')
      time.sleep(1)
    assert child.returncode==0,'Preserve failed point; do not blind restart'
    assert json.loads((leaf/'status.json').read_text())['status']=='complete'
    state['completed'].append(model);save()
   state.update(status='complete',finished=time.time())
 except BaseException as error:
  state.update(status='stopped',error=repr(error))
  if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
  raise
 finally:
  if active:
   result=control('quit');state.update(mps_release_returncode=result.returncode,mps_release_stdout=result.stdout,mps_release_stderr=result.stderr)
  state['mps_released']=not active or state.get('mps_release_returncode')==0;save()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True);p.add_argument('--model',choices=['kimi','deepseek']);p.add_argument('--output',type=Path);p.add_argument('--deadline',type=float,required=True)
 a=p.parse_args();assert REPO==Path('/data/2026OpticsMoE') and time.time()<a.deadline<=time.time()+3600
 if a.phase=='suite':suite(a)
 else:a.output=a.output.resolve();assert a.output.parent==ROOT;worker(a)
