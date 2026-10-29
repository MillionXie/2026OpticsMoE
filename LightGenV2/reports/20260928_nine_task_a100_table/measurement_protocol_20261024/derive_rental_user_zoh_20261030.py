"""Separate CPU derivation from immutable512-call failed strict run, no GPU."""
import argparse,csv,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
SIM=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
RAW=SIM/'rental_energy_remeasure_20261030_03/deepseek_cap50_repeat01'
OUT=HERE/'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/DeepSeek50_repeat01_userZOH_20261030_01'
STORAGE_SHA='defa12da6d650ef3079311939f34ec108d54ec94a5607d1ec9cf999ab88b0339'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,j):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(j,f,ensure_ascii=True,indent=2,allow_nan=False)
def rows():
 p=SIM/'lgvq_full_deepseek_kimi_rental_20261017/provisioning/original_manifest.csv'
 if sha(p)!='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc':raise ValueError('Original manifest differs')
 with p.open(newline='',encoding='utf-8-sig') as f:a=[r for r in csv.DictReader(f) if r['split']=='test']
 if len(a)!=558:raise ValueError('Original558 metadata required')
 return a[:512]
def verify_inputs(pins):
 for name,digest in pins.items():
  if Path(name).name!=name or sha(RAW/name)!=digest:raise ValueError('Trusted raw source differs')
 identity=read(RAW/'identity.json');state=read(RAW/'status.json')
 if state['status']!='failed' or state['samples']!=512 or (RAW/'report.json').exists():raise ValueError('Preserve exact failed strict run')
 storage_path=Path('/root/data/lgvq_20261029/input_local_overlay_receipt01.json')
 if sha(storage_path)!=STORAGE_SHA:raise ValueError('Original input transfer receipt differs')
 storage=read(storage_path);by={r['sample_id']:r for r in storage['records']};records={};start=time.time()
 for r in rows():
  p=Path(r['video_path']);s=p.stat();v=dict(path=str(p),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
  if v!=identity['input_files_metadata_before'][r['sample_id']]:raise ValueError('Original pre-run metadata no longer matches')
  v['sha256']=sha(p);orig=by[r['sample_id']]
  if v['sha256']!=orig['sha256'] or v['size_bytes']!=orig['bytes'] or v['path']!=orig['original_path']:raise ValueError('Original input bytes differ')
  records[r['sample_id']]=v
 value=dict(status='passed_late_independent_input_verification',start_unix=start,end_unix=time.time(),input_files=records,
   verified_after_original_failed_run=True,not_claimed_as_original_T1_postcheck=True,original_storage_receipt_sha256=STORAGE_SHA,
   raw_pins=pins,source_commit=os.environ['POSTPROCESS_PUBLISHED_COMMIT'],model_loaded=False,torch_imported='torch' in sys.modules)
 if value['torch_imported']:raise RuntimeError('CPU-only')
 save(OUT/'late_input_verification01.json',value);return dict(status=value['status'],path=str(OUT/'late_input_verification01.json'),sha256=sha(OUT/'late_input_verification01.json'))
def derive():
 import run_rental_user_zoh_20261030 as zoh
 from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
 from scipy.stats import spearmanr,pearsonr
 receipt=read(SIM/'rental_energy_remeasure_20261030_03/whole_root_backup_receipt_complete01.json')
 for r in receipt['files']:
  if sha(SIM/'rental_energy_remeasure_20261030_03'/r['name'])!=r['sha256']:raise ValueError('Sealed raw bytes differ')
 late=read(OUT/'late_input_verification01.json');identity=read(RAW/'identity.json');expected=rows()
 if late['status']!='passed_late_independent_input_verification':raise ValueError('Late byte/metadata validation required')
 zoh.entry.native.configure();ref_path,reference=zoh.entry.base.reference('deepseek')
 for key in ['assets','checkpoint_sha256','floating_parameter_dtypes','quality_readout_dtypes']:
  if identity[key]!=reference[key]:raise ValueError('Original frozen FULL model/head/precision differs')
 if identity['checkpoint_sha256']!='93e6f0de60192b0edf4e13de4af29697eb1f3ad6144567807ac0c2d3e2e395a0' or identity['batch']!=1 or identity['explicit_warmup_forwards']!=0 or identity['visible_multiprocessors']!=54:raise ValueError('Original native execution contract differs')
 for name,digest in late['raw_pins'].items():
  if sha(RAW/name)!=digest:raise ValueError('Input verifier raw pin differs')
 with (RAW/'calls.csv').open(newline='') as f:calls=list(csv.DictReader(f))
 with (RAW/'telemetry.csv').open(newline='') as f:raw=list(csv.DictReader(f))
 evidence=read(RAW/'input_audit_partial.json')['samples']
 ids=[r['sample_id'] for r in expected]
 if len(calls)!=512 or [r['sample_id'] for r in calls]!=ids or [r['sample_id'] for r in evidence]!=ids or list(late['input_files'])!=ids:raise ValueError('All512 ordered native calls required')
 with ref_path.with_name('calls.csv').open(newline='',encoding='utf-8-sig') as f:reference_calls=list(csv.DictReader(f))[:200]
 if len(reference_calls)!=200 or any(a['sample_id']!=b['sample_id'] or json.loads(a['frame_positions'])!=json.loads(b['frame_positions']) for a,b in zip(calls,reference_calls)):raise ValueError('Original frame selection differs')
 previous=-math.inf
 for c,r in zip(calls,expected):
  a,b,d,e,f=(float(c[k]) for k in ['preprocess_start','model_call_start','first_block_start','gpu_output','call_return'])
  if not previous<=a<=b<=d<e<=f or float(c['target_mos'])!=float(r['temporal']):raise ValueError('Original timing/targets differ')
  previous=f
 for name,digest in identity['sources'].items():
  blob=subprocess.check_output(['git','show',identity['git_commit']+':'+name.replace('\\','/')],cwd=REPO,timeout=15)
  if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original Git source differs')
 samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],float(r['watts']),None,None) for r in raw]
 lo=float(calls[0]['preprocess_start']);hi=float(calls[-1]['call_return']);power=zoh.interval_energy([(lo,hi)],samples,338.2);board=zoh.interval_energy([(lo,hi)],samples,0.)
 targets=[float(c['target_mos']) for c in calls];pred=[float(c['prediction']) for c in calls]
 cost=dict(continuous_wall_s=hi-lo,gpu_energy_j=board['energy_j'],estimated_system_energy_j=power['energy_j'],host_proxy_w=338.2,power_integration=power,videos=512,frames=2048)
 result=dict(status='complete_separate_CPU_derivation',qualification='user_authorized_ZOH_estimate_not_strict250ms',strict250ms_qualified=False,
  model='deepseek',cap=50,repeat=1,batch=1,measured_videos=512,explicit_warmup_forwards=0,gpu_uuid=identity['gpu_uuid'],power_limit_w=400,visible_multiprocessors=54,
  original_failed_status_unchanged=True,raw_source=RAW.relative_to(REPO).as_posix(),raw_identity_sha256=sha(RAW/'identity.json'),
  late_input_verification_sha256=sha(OUT/'late_input_verification01.json'),input_verification_was_late=True,
  cost=cost,effective_input_tokens512=sum(r.get('effective_input_tokens',0) for r in evidence),performance512=dict(srcc=float(spearmanr(targets,pred).statistic),plcc=float(pearsonr(targets,pred).statistic)),
  original_identity=identity,model_loaded=False,torch_imported='torch' in sys.modules,derivation_commit=os.environ['POSTPROCESS_PUBLISHED_COMMIT'])
 if result['torch_imported']:raise RuntimeError('CPU-only derivation')
 save(OUT/'derived_report01.json',result);return dict(status=result['status'],sha256=sha(OUT/'derived_report01.json'),path=str(OUT/'derived_report01.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['verify-inputs','derive'],required=True);p.add_argument('--pins-json');a=p.parse_args()
 print(json.dumps(verify_inputs(json.loads(a.pins_json)) if a.phase=='verify-inputs' else derive(),ensure_ascii=True))
