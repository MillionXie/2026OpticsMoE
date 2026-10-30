"""Recalculate saved LG DAG with explicit new CPU preparation and1886Hz SLM.

Immutable old raw/qualification. No model, Torch, GPU or physical optics run.
Existing CPU worker assignments/two-field pipeline stay fixed. New operations
use those SAME16 CPU resources, not extra online cores or a wall-time/16 trick.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as prep
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
SIM=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
OUT=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/LG_frontend1886_20261030_01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def optical_s(hz):
 if not math.isfinite(hz) or hz<=0:raise ValueError('Positive finite SLM rate required')
 return 1/hz+.000300+.0000307
def nodes_for(report,raw,hz,front=None):
 phases={(r['field'],r['phase']):r['charged_compute_s'] for r in report['raw_phases']}
 fields=report['fields'];nodes=[]
 for field in range(fields):
  release=[f'f{field-2}.decoder'] if field>=2 else []
  patches=[r for r in raw['patch'] if r['field']==field]
  if len(patches)!=16:raise ValueError('Exactly16 videos per coherent field')
  inputs=[]
  for r in patches:
   base=f'f{field}.patch{r["slot"]}';resource='cpu_patch_'+str(r['worker_thread']);parents=list(release)
   stages=[]
   if front is not None:
    added=front[r['sample_id']]
    stages.append(('front_decode_RGB_crop',added['decode_backend_cpu_s']+added['RGB_cpu_s']+added['crop_metadata_cpu_s']))
   stages += [('resize',r['resize_cpu_s']),('normalization',r['normalization_cpu_s']),('patch',r['patch_cpu_s'])]
   if front is not None:stages.append(('quality_pixel_preparation',added['quality_cpu_s']))
   for phase,duration in stages:
    if duration<0 or not math.isfinite(duration):raise ValueError('Finite nonnegative CPU work required')
    name=base if phase=='patch' else base+'.'+phase
    nodes.append(dict(id=name,field=field,sequence=0,phase=phase,resource=resource,parents=parents,duration_s=duration));parents=[name]
   inputs+=parents
  previous=f'f{field}.tokenizer'
  nodes.append(dict(id=previous,field=field,sequence=0,phase='tokenizer',resource='cpu_tokenizer',parents=release,duration_s=raw['tokenizer'][field]['thread_cpu_s']))
  stages=['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual']
  for i in range(1,7):
   stages += [f'optics{i}',f'oeo{i}']
   if i==3:stages+=['merger','adapter_language']
  stages+=['decoder']
  for sequence,phase in enumerate(stages,1):
   optical=phase.startswith('optics');name=f'f{field}.{phase}';parents=[previous]
   if phase=='emb_conv3d':parents+=inputs
   nodes.append(dict(id=name,field=field,sequence=sequence,phase=phase,resource='optical' if optical else 'GPU',
    parents=parents,duration_s=optical_s(hz) if optical else phases[(field,phase)]));previous=name
 return nodes
def calculate(report,raw,hz,front=None):
 nodes=nodes_for(report,raw,hz,front);schedule=scope.compute_schedule(nodes);by={n['id']:n for n in schedule['nodes']};parts=defaultdict(float)
 for name in schedule['additive_path']:parts[by[name]['phase']]+=by[name]['duration_s']/report['fields']*1000
 ms=schedule['total_s']/report['fields']*1000
 if not math.isclose(sum(parts.values()),ms,abs_tol=1e-7):raise ValueError('Critical path must close')
 return dict(ms16=ms,SLM_hz=hz,optical_layer_ms=optical_s(hz)*1000,stage_critical_contribution_ms16=dict(parts),
  cpu_online_workers=len({r['worker_thread'] for r in raw['patch']}),total_fields=report['fields'],
  includes_added_preparation=front is not None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-records',type=Path,required=True);a=p.parse_args()
 cpu=read(a.cpu_records)
 if cpu['status']!='CPU_ONLY_NATIVE512_PREPROCESSING_COMPLETE' or len(cpu['rows'])!=512 or cpu['GPU_or_neural_forward_executed']:raise ValueError('Completed CPU-only512 profile required')
 front={r['sample_id']:r for r in cpu['rows']}
 if len(front)!=512 or any(r['cuda_initialized'] or not r['original_flat_SHA_equal'] or r['decoder_threads']!=1 for r in front.values()):raise ValueError('Original pixels and CPU-only evidence required')
 frozen=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/LGVQ_LG_first_parallel_20261030_01/lgvq_reorganized_sources01.json'
 if sha(frozen)!='9873a8a7c021a977dc67682b1dcb618fbbaabf7f4944eb4ed2e137bcc1decc1d':raise ValueError('Original accepted calculation pin')
 old=read(frozen);leaf=SIM/'lightgen_rental_20261029_01/lg_repeat01';raw=read(leaf/'raw_operator_records.json')
 if sha(leaf/'raw_operator_records.json')!='f3d1bde0fbcbdbc184b611a5e163bae201049446555c33bc35cc73068491233e':raise ValueError('Original CPU trace differs')
 trace=leaf/'raw_kernel_trace.json'
 if sha(trace)!='60104da5c2fd98bbd0fd767d186dddfeb5b742d009c1b24091d1d464c1c5390e':raise ValueError('Original GPU trace differs')
 kernels=scope.traced_kernels(read(trace));report=prep.summarize_compute(kernels,raw['patch'],raw['tokenizer'],32);del kernels
 base=calculate(report,raw,1400)
 if not math.isclose(base['ms16'],old['LG_rental']['time16_s']*1000,abs_tol=1e-7):raise ValueError('New builder must reproduce old29.333 exactly')
 rental=dict(previous_scope_1400=base,previous_scope_1886=calculate(report,raw,1886),
             expanded_preparation_1400=calculate(report,raw,1400,front),expanded_preparation_1886=calculate(report,raw,1886,front))
 labs=[]
 pins={r['path']:r['sha256'] for r in old['source_files']}
 for i in (1,2,3):
  root=SIM/'lgvq_compute_scope_20261029_05'/f'lg_repeat{i:02d}';r=read(root/'compute_report.json');w=read(root/'raw_operator_records.json')
  for file in [root/'compute_report.json',root/'raw_operator_records.json']:
   if sha(file)!=pins[file.relative_to(REPO).as_posix()]:raise ValueError('Accepted laboratory source changed')
  c=calculate(r,w,1400)
  if not math.isclose(c['ms16'],r['time16_s']*1000,abs_tol=1e-7):raise ValueError('Lab original scope must close')
  labs.append(calculate(r,w,1886))
 value=dict(status='RECALCULATED_EXPANDED_CPU_FRONTEND_AND1886_SLM_NOT_PHYSICAL_E2E',rental=rental,
  lab_previous_scope_1886_pooled_ms16=sum(x['ms16'] for x in labs)/3,lab_previous_scope_1886_rounds=labs,
  lab_expanded_preparation=None,lab_missing_reason='No active-CPU decode/quality profile on laboratory CPU; do not substitute rental durations',
  CPU_profile_sha256=sha(a.cpu_records),old_frozen_calculation_sha256=sha(frozen),CPU_profile_source_commit=cpu['published_commit'],
  exposure_us=300,readout_us=30.7,SLM_hz_user_specified_not_hardware_verified=1886,
  added=['active file-open/seek/decode CPU','RGB conversion/65percent crop','native fixed quality14 input preparation'],
  double_counted_resize_or_normalization=False,OS_dispatch_and_queues_excluded=True,disk_off_CPU_wait_excluded=True,
  same_original16_online_preprocessing_resources=True,two_field_modelled_pipeline=True,
  lab_and_rental_CPU_measurements_not_mixed=True,neural_or_GPU_rerun=False,
  physical_photonic_time_measured=False,rental_original_numerical_gate_passed=False,torch_imported='torch' in sys.modules)
 if value['torch_imported']:raise ValueError('Recalculation must not import Torch')
 OUT.mkdir(exist_ok=True)
 with (OUT/'LG_time_recalculation01.json').open('x') as f:json.dump(value,f,ensure_ascii=True,indent=2,allow_nan=False)
 print(json.dumps(value,ensure_ascii=True))
if __name__=='__main__':main()
