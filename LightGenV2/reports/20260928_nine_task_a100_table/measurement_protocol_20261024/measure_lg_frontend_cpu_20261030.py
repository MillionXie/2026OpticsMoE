"""One CPU-only original512 preprocessing profile; no GPU/model forward.

Time active CPU work, not OS dispatch, queues or disk waiting. Decode uses one
FFmpeg thread so caller-thread CPU timing does not lose decoder worker work.
Existing resize/PIL packing is measured but NOT added twice to old LG timing.
"""
import ast,csv,hashlib,inspect,json,os,subprocess,sys,textwrap,time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
SIM=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
ROOT=SIM/'lg_frontend_cpu_1886_20261030_01'
OLD=SIM/'lightgen_rental_20261029_01/lg_repeat01'
PREP_SHA='157cf5b717afe420c38e25c88ffea753c5b4d1c1fef7bf0add97e56764ba0b4f'
MODEL=Path('/root/data/lgvq_20261029/qwen_demo_assets_20261030_01')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=True,indent=2,allow_nan=False));tmp.replace(p)
def initialize():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,cv2
 from transformers import AutoProcessor
 from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
 global PROCESSOR,DECODE,NATIVE,METER,CV2
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 PROCESSOR=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
 NATIVE=native;CV2=cv2;METER={}
 def timed(name,fn,*a,**kw):
  c=time.thread_time();w=time.perf_counter()
  try:return fn(*a,**kw)
  finally:
   METER[name+'_cpu_s']=METER.get(name+'_cpu_s',0)+time.thread_time()-c
   METER[name+'_wall_s']=METER.get(name+'_wall_s',0)+time.perf_counter()-w
 class Capture:
  def __init__(self,path):
   self.cap=timed('backend',cv2.VideoCapture,path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
   if int(self.cap.get(cv2.CAP_PROP_N_THREADS))!=1:raise ValueError('Actual single decoder thread required')
  def get(self,*a):return timed('backend',self.cap.get,*a)
  def set(self,*a):return timed('backend',self.cap.set,*a)
  def read(self,*a):return timed('backend',self.cap.read,*a)
  def release(self):return timed('backend',self.cap.release)
 class ObservedCV:
  def __getattr__(self,key):return getattr(cv2,key)
  VideoCapture=Capture
  @staticmethod
  def cvtColor(*a,**kw):return timed('RGB',cv2.cvtColor,*a,**kw)
  @staticmethod
  def resize(*a,**kw):return timed('already_counted_resize',cv2.resize,*a,**kw)
 class ImageProxy:
  def __getattr__(self,key):return getattr(native.Image,key)
  @staticmethod
  def fromarray(*a,**kw):return timed('already_counted_PIL',native.Image.fromarray,*a,**kw)
 source=textwrap.dedent(inspect.getsource(native.decode_frames));tree=ast.parse(source)
 class Observe(ast.NodeTransformer):
  def visit_Name(self,n):
   if n.id=='cv2':return ast.copy_location(ast.Name(id='_observed_cv2',ctx=n.ctx),n)
   return n
 tree=ast.fix_missing_locations(Observe().visit(tree));namespace=dict(native.decode_frames.__globals__,_observed_cv2=ObservedCV(),Image=ImageProxy())
 exec(compile(tree,'<same_decode_math_single_decoder_thread>','exec'),namespace)
 DECODE=namespace['decode_frames']
def sample(item):
 global METER
 import torch
 METER={};c=time.thread_time();p=time.process_time();w=time.perf_counter()
 images=DECODE(Path(item['video_path']),4)
 decode_cpu=time.thread_time()-c;decode_process=time.process_time()-p;decode_wall=time.perf_counter()-w
 # Exact native FP32 patch content validates decoded pixels. This validation's
 # processor work is outside the new charge: old resize/normalize/patch remains.
 pixels=PROCESSOR(text=['x']*4,images=images,return_tensors='pt')['pixel_values'].contiguous().numpy()
 if hashlib.sha256(pixels.tobytes()).hexdigest()!=item['original_flat_sha256']:raise ValueError('Original decoded pixel/patch SHA differs')
 c=time.thread_time();p=time.process_time();w=time.perf_counter()
 quality=NATIVE.quality_tokens_from_images(images,video_count=1,frame_count=4,output_grid=7)
 quality_cpu=time.thread_time()-c;quality_process=time.process_time()-p;quality_wall=time.perf_counter()-w
 if torch.cuda.is_initialized():raise RuntimeError('CPU profile initialized CUDA')
 excluded=METER.get('already_counted_resize_cpu_s',0)+METER.get('already_counted_PIL_cpu_s',0)
 rgb=METER.get('RGB_cpu_s',0);backend=METER.get('backend_cpu_s',0)
 other=decode_cpu-excluded-rgb-backend
 if other<0 or tuple(quality.shape)!=(1,4,49,14):raise ValueError('Native CPU partition or geometry differs')
 return dict(sample_id=item['sample_id'],worker_pid=os.getpid(),decode_backend_cpu_s=backend,RGB_cpu_s=rgb,
  crop_metadata_cpu_s=other,quality_cpu_s=quality_cpu,added_cpu_s=decode_cpu-excluded+quality_cpu,
  already_counted_resize_PIL_cpu_s=excluded,decode_cpu_s=decode_cpu,decode_process_cpu_s=decode_process,
  decode_wall_s=decode_wall,quality_process_cpu_s=quality_process,quality_wall_s=quality_wall,
  raw_video_sha256=sha(item['video_path']),original_flat_SHA_equal=True,quality_sha256=hashlib.sha256(quality.numpy().tobytes()).hexdigest(),
  cuda_initialized=False,decoder_threads=1,torch_threads=torch.get_num_threads(),measurements=METER)
def main():
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing
 import importlib.metadata
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('GPU must be explicitly hidden')
 if ROOT.exists():raise FileExistsError('Preserve this CPU profile; do not rerun')
 if sha(OLD/'input_preparation.json')!=PREP_SHA:raise ValueError('Trusted original512 preparation required')
 rows=json.loads((OLD/'input_preparation.json').read_text())['rows']
 if len(rows)!=512 or len({r['sample_id'] for r in rows})!=512:raise ValueError('Original512 required')
 commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'];paths=[Path(__file__),REPO/'experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/cache_qwen_front.py']
 sources={p.relative_to(REPO).as_posix():sha(p) for p in paths}
 for name,h in sources.items():
  if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)).hexdigest()!=h:raise ValueError('Published source required')
 ROOT.mkdir();result=[];start=time.time()
 try:
  write(ROOT/'status.json',dict(status='CPU_preprocessing',pid=os.getpid(),completed=0,started=start))
  with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
   futures=[pool.submit(sample,r) for r in rows]
   for item,f in zip(rows,futures):
    value=f.result(timeout=max(1,900-(time.time()-start)))
    if value['sample_id']!=item['sample_id'] or value['raw_video_sha256']!=item['sha256']:raise ValueError('Original input bytes/order changed')
    result.append(value)
    if len(result)%32==0:write(ROOT/'status.json',dict(status='CPU_preprocessing',pid=os.getpid(),completed=len(result),started=start))
  write(ROOT/'cpu_preprocessing01.json',dict(status='CPU_ONLY_NATIVE512_PREPROCESSING_COMPLETE',published_commit=commit,sources=sources,
   input_preparation_sha256=PREP_SHA,rows=result,profiling_processes=8,decoder_threads_per_process=1,
   python=sys.version,packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','opencv-python']},
   GPU_or_neural_forward_executed=False,added_charge='active decode/RGB/crop and fixed quality14 preparation CPU only',
   excluded='already-counted resize/PIL, processor normalization/patch verification, disk/off-CPU waits, OS dispatch, IPC, queue gaps',
   raw_wall_and_process_CPU_diagnostics_saved=True,finished=time.time()))
  write(ROOT/'status.json',dict(status='complete',pid=os.getpid(),completed=512,finished=time.time()))
 except BaseException as e:
  write(ROOT/'cpu_preprocessing_partial01.json',dict(rows=result));write(ROOT/'status.json',dict(status='failed',pid=os.getpid(),completed=len(result),error=repr(e)));raise
if __name__=='__main__':main()
