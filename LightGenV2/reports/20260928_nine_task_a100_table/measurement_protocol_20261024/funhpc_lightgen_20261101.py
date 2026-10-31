"""Same original LG2 calculation, new slow-host storage and diagnostic output.

Keep the original FFT execution and all numeric audits. A numeric failure is
retained as diagnostic_only, never silently promoted to qualified compute.
"""
from pathlib import Path
from types import SimpleNamespace
import json,os,time,sys
import run_funhpc_lgvq_20261101 as queue


def worker(a):
 import run_lgvq_compute_scope_20261029 as runtime
 import run_lgvq_resize_norm_20261029 as wrapper
 import lgvq_compute_scope_20261029 as scope
 import lgvq_resize_norm_20261029 as prep
 import lgvq_live_frontend as live
 from LightGenV2.tasks.t06_video_quality_assessment import strict_tf_window_20261020 as windows
 from run_rental_user_zoh_20261030 import interval_energy
 import cv2
 cv2.setNumThreads(1)
 scope.prepare_cpu_worker=prep.prepare_cpu_worker;scope.prepare_video=prep.prepare_video
 scope.patch_video=prep.patch_video;scope.summarize_compute=prep.summarize_compute
 runtime.GPU=queue.GPU;runtime.PYTHON=sys.executable
 def write(path,value):
  if path.name=='compute_identity.json':value=dict(value,gpu_uuid=queue.GPU,power_limit_W=400,profile='FUNHPC_LG2_NATIVE_RESIZE_PATCH_DIAGNOSTIC')
  wrapper.write(path,value)
 runtime.write=write
 # Original dataset/parameter binding contents remain unchanged; setup supplies
 # checked DATA aliases, not a changed processor/model implementation.
 windows.interval_energy=interval_energy
 leaf=queue.ROOT/queue.name(('LG2',100,a.repeat));leaf.mkdir(exist_ok=False)
 sources={**runtime.sources(),**queue.sources(),Path(__file__).relative_to(queue.REPO).as_posix():queue.sha(__file__),Path(wrapper.__file__).relative_to(queue.REPO).as_posix():queue.sha(wrapper.__file__),Path(prep.__file__).relative_to(queue.REPO).as_posix():queue.sha(prep.__file__)}
 runtime.verify(os.environ['MEASUREMENT_PUBLISHED_COMMIT'],sources)
 queue.write(leaf/'source_identity.json',dict(source_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],sources=sources))
 g=queue.guard(a.deadline);start=time.perf_counter()
 try:
  g.check()
  try:
   report=runtime.execute(SimpleNamespace(output=leaf,pipe=queue.PIPE,deadline=a.deadline),g)
   queue.write(leaf/'compute_report.json',report)
  except ValueError as e:
   if str(e)!='Original exact same-A100 per-video output gate failed':raise
   # Original implementation writes all raw outputs/trace BEFORE the numeric
   # admission check. Derive only a distinctly labelled diagnostic on those.
   raw=queue.read(leaf/'raw_operator_records.json');numeric=queue.read(leaf/'numeric_audit.json')
   report=prep.summarize_compute(scope.traced_kernels(queue.read(leaf/'raw_kernel_trace.json')),raw['patch'],raw['tokenizer'],32)
   report.update(status='diagnostic_only_numeric_gate_failed',all_predictions_bitwise_equal=False,numeric_audit=numeric,
    actual_FFT_software_wall_s=raw['formal_host_end']-raw['formal_host_start'])
   queue.write(leaf/'diagnostic_report.json',report)
  raw=queue.read(leaf/'raw_operator_records.json')
  from recalculate_lg_front1886_20261030 import calculate
  schedule=calculate(report,raw,1886)
  value=dict(status='complete' if report.get('all_predictions_bitwise_equal') else 'diagnostic_only_numeric_gate_failed',
   videos=512,groups16=32,explicit_warm_fields=3,numeric_audit=queue.read(leaf/'numeric_audit.json'),
   actual_FFT_software_wall_s=report['actual_FFT_software_wall_s'],actual_FFT_software_ms16=report['actual_FFT_software_wall_s']/32*1000,
   SLM1886_modelled_ms16=schedule['ms16'],stage_critical_contribution_ms16=schedule['stage_critical_contribution_ms16'],
   fresh_worker_including_loading_preparation_s=time.perf_counter()-start,
   same_original_resize_normalization_patch_scope=True,CPU_dispatch_and_wait_excluded_only_from_model=True,
   FFT_not_charged_as_physical_optics=True,physical_photonic_hardware_measured=False,
   actual_software_wall_and_modelled_schedule_not_interchangeable=True)
  queue.write(leaf/'timing_summary.json',value);queue.write(leaf/'status.json',dict(status=value['status'],pid=os.getpid(),unix=time.time(),completed=512))
  print(json.dumps(value,ensure_ascii=True),flush=True)
 except BaseException as e:
  queue.write(leaf/'status.json',dict(status='failed',pid=os.getpid(),unix=time.time(),error=repr(e)));raise
 finally:g.close(leaf/'process_audit.json')
