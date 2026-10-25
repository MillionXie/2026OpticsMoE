"""Local data-only audit of new rental serial scope and LG capture candidates."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[4];TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment';RUN=TASK/'runs/simulation'
HEADS={'deepseek':'93e6f0de60192b0edf4e13de4af29697eb1f3ad6144567807ac0c2d3e2e395a0','kimi':'e6bf55b3b7ff318d37c973b67c4a6ce1062e4760ac91c2bdc748628988022586'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-5),(a,b)
def union_ms(intervals):
 result=0.;end=-math.inf
 for t in sorted(intervals,key=lambda x:x['start']):result+=max(0,t['end']-max(end,t['start']));end=max(end,t['end'])
 return result*1000
def receipt(root):
 r=read(root/'backup_receipt_complete01.json')
 for f in r['verified_files']:
  p=root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
 return len(r['verified_files'])
def blobs(mapping,commits):
 matched={}
 for path,digest in mapping.items():
  rel=path.replace('\\','/')
  for prefix in ('/DATA/DATA1/guest3/2026OpticsMoE/','/data/2026OpticsMoE/'):
   if rel.startswith(prefix):rel=rel[len(prefix):];break
  for commit in dict.fromkeys(commits):
   b=subprocess.run(['git','show',commit+':'+rel],cwd=REPO,capture_output=True)
   if b.returncode==0 and hashlib.sha256(b.stdout).hexdigest()==digest:matched[rel]=commit;break
  else:raise ValueError('Unpublished or changed source: '+rel)
 return matched
def rental(model,audit_name='local_independent_audit01.json'):
 root=RUN/'a10080_corrected_serial_20261026_01'/(model+'50_repeat01');t=read(root/'timing.json');identity=read(root/'identity.json')
 assert read(root/'status.json')['status']=='complete' and t['status']=='complete'
 assert t['video_calls']==2048 and t['unique_videos']==512 and t['rounds']==1 and t['cap']==50
 assert identity['checkpoint_sha256']==HEADS[model] and identity['visible_multiprocessors']==54 and identity['power_limit_w']==400
 assert identity['gpu_uuid']=='GPU-ecac2c49-306a-9ec5-1572-8fcd2d232165'
 assert identity['precision']=='native' and t['default_original_forward'] and t['no_graph'] and t['no_prefetch'] and t['CPU_preparation_single_video']
 expected={'vision_class':'Block','text_class':'DeepseekV2Attention'} if model=='deepseek' else {'vision':'flash_attention_2','text_class':'DeepseekV3FlashAttention2'}
 assert identity['runtime']['actual_attention']==expected
 rows=t['records'];assert len(rows)==2048
 for i,r in enumerate(rows):
  assert (r['index'],r['source_index'],r['cycle'])==(i,i%512,i//512)
  assert r['CPU_start']<=r['CPU_end']<=r['H2D_end']<=r['GPU_host_start']<=r['GPU_ready_host']<=r['CPU_result_return']
  if i:assert r['CPU_start']>=rows[i-1]['CPU_result_return']
  assert len(r['input_ids'])==len(r['attention_mask'])==1 and bool(r['attention_mask'][0][-1])
  assert all(r['CPU_start']<=v['start']<=v['end']<=r['CPU_end'] for v in r['tokenizer_intervals'])
  close(union_ms(r['tokenizer_intervals']),r['tokenizer_CPU_ms'])
 full=(rows[-1]['GPU_ready_host']-rows[0]['CPU_start'])*1000
 removed=sum((r['H2D_end']-r['CPU_start'])*1000-union_ms(r['tokenizer_intervals']) for r in rows)
 corrected=full-removed;close(full,t['full_before_subtraction_ms']);close(removed,t['removed_image_preparation_and_H2D_ms']);close(corrected,t['corrected_GPU_ready_ms'])
 close(corrected/128,t['time16_ms']);close(corrected/8192,t['equivalent_frame_ms'])
 close(sum(r['GPU_model_cuda_ms'] for r in rows),t['GPU_model_cuda_sum_ms'])
 with np.load(root/'outputs.npz',allow_pickle=False) as f:arr=f['prediction']
 assert arr.shape==(2048,) and np.array_equal(arr,np.array([r['prediction'] for r in rows],dtype=np.float32))
 assert all(np.array_equal(arr[:512],arr[i:i+512]) for i in range(512,2048,512))
 guards=read(root/'process_audit.json');assert guards
 for g in guards:
  assert len(g['clients'])<=1
  allowed=set(g['servers'])|set(g['clients']);pids={int(r[0].strip()) for r in g['processes']};assert pids<=allowed
 source_commits=blobs(identity['sources'],[identity['published_commit'],identity['actual_HEAD']])
 qfiles=read(RUN/'a100_corrected_boundaries_20261026_02/qwen01/input_files.json');actual=read(root/'input_files.json')
 assert len(actual)==512
 assert [(x['sample_id'],x['bytes'],x['sha256']) for x in actual]==[(x['sample_id'],x['bytes'],x['sha256']) for x in qfiles]
 assert [r['sample_id'] for r in rows]==[x['sample_id'] for _ in range(4) for x in actual]
 historical=RUN/'lgvq_mps_e2e_a100_20261024/rental_80gb_20261025_01'/(model+'_cap50_repeat01')/'calls.csv'
 historical_delta=None
 if historical.exists():
  with historical.open(encoding='utf-8-sig',newline='') as f:prior=list(csv.DictReader(f))
  assert [r['sample_id'] for r in rows[:200]]==[r['sample_id'] for r in prior]
  historical_delta=float(np.max(np.abs(arr[:200]-np.array([float(r['prediction']) for r in prior],dtype=np.float32))))
 result=dict(status='passed',model=model,platform='A100-SXM4-80GB / 400W / MPS50',video_calls=2048,unique_videos=512,time16_ms=corrected/128,equivalent_frame_ms=corrected/8192,
  corrected_total_ms=corrected,full_before_subtraction_ms=full,removed_image_preparation_and_H2D_ms=removed,tokenizer_CPU_ms=t['tokenizer_CPU_ms'],GPU_model_cuda_sum_ms=t['GPU_model_cuda_sum_ms'],
  source_path=str((root/'timing.json').resolve()),source_sha256=sha(root/'timing.json'),receipt_verified_files=receipt(root),source_commits=source_commits,
  four_cycles_predictions_bitwise_equal=True,first200_vs_historical_native_max_abs=historical_delta,same_512_video_bytes_as_Qwen_and_LG=True,guard_rows=len(guards))
 write=root/audit_name;assert not write.exists();write.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result
def lg(root,audit_name='local_independent_audit01.json'):
 t=read(root/'lg01/timing.json');assert t['status']=='complete' and read(root/'suite_status.json')['status']=='complete'
 rows=t['records'];assert len(rows)==128 and t['video_calls']==2048
 kinds=('optical_simulation','detector_ROI','electronic_residual');totals={k:0. for k in kinds}
 assert len(t['capture']['segments'])==33
 capture={k:sum(x['capture_host_ms'] for x in t['capture']['segments'] if x['kind']==k) for k in kinds}
 for i,r in enumerate(rows):
  assert r['field']==i and len(r['optical_simulation_spans'])==16
  for k,n in zip(kinds,(6,6,4)):
   spans=[x['cuda_ms'] for x in r['optical_simulation_spans'] if x['kind']==k];assert len(spans)==n and min(spans)>=0;totals[k]+=sum(spans)
 h2d=sum((r['H2D_end']-r['H2D_start'])*1000 for r in rows);book=sum((r['bookkeeping_end']-r['bookkeeping_start'])*1000 for r in rows[:-1])
 charged=t['full_software_host_ms']-h2d-book-sum(totals.values())-sum(capture.values());fixed=128*6*(1000/1400+.300+.0307)
 close(charged,t['charged_actual_electronic_ms']);close(fixed,t['fixed_optical_ms']);close((charged+fixed)/128,t['time16_ms'])
 identity=read(root/'lg01/identity.json');projection=read(root/'lg01/projection_identity.json')
 matched=blobs(dict(identity['sources'],**projection['sources']),[identity['published_commit'],identity['source_HEAD']])
 with np.load(root/'lg01/outputs.npz',allow_pickle=False) as f:arr=f['prediction'].copy()
 reference=t['source_reference_for_validation_only'];p=REPO/Path(reference['path']).relative_to('/DATA/DATA1/guest3/2026OpticsMoE');assert sha(p)==reference['sha256']
 with np.load(p,allow_pickle=False) as f:assert np.array_equal(arr,f['prediction'])
 for g in read(root/'lg01/process_audit.json'):assert not g['foreign_gpu_processes'] and not g['foreign_mps_clients']
 low=read(root/'lg01/lowlevel_capture_identity.json');assert low['status']=='complete' and low['capture_audits'][0]['preparation']['calls']==1
 matched.update(blobs(low['sources'],[identity['published_commit'],identity['source_HEAD']]))
 score_path=root/'lg01/score_only_identity.json'
 if score_path.exists():
  score=read(score_path);assert score['status']=='complete' and score['original_2048_prediction_reference_bitwise_passed']
  assert score['auxiliary_outputs_omitted'] and not score['complete_auxiliary_dictionary_unchanged']
  matched.update(blobs(score['sources'],[identity['published_commit'],identity['source_HEAD']]))
 result=dict(status='passed',model='LG_scoreonly' if score_path.exists() else 'LG_lowlevel',platform='A100-PCIE-40GB / 250W / MPS100',video_calls=2048,time16_ms=(charged+fixed)/128,equivalent_frame_ms=(charged+fixed)/8192,
  charged_actual_electronic_ms=charged,fixed_optical_ms=fixed,full_software_host_ms=t['full_software_host_ms'],excluded_GPU_ms_by_kind=totals,excluded_capture_CPU_ms_by_kind=capture,
  electronic_capture_ms=sum(x['capture_host_ms'] for x in t['capture']['segments'] if x['kind']=='electronics'),capture_total_ms=t['capture']['total_ms'],
  source_path=str((root/'lg01/timing.json').resolve()),source_sha256=sha(root/'lg01/timing.json'),receipt_verified_files=receipt(root),source_commits=matched,
  all_predictions_bitwise_equal_original=True)
 dest=root/audit_name;assert not dest.exists();dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['kimi','deepseek','LG_lowlevel','LG_scoreonly'],required=True);p.add_argument('--audit-name',default='local_independent_audit01.json');a=p.parse_args()
 assert Path(a.audit_name).name==a.audit_name and a.audit_name.startswith('local_independent_audit') and a.audit_name.endswith('.json')
 result=lg(RUN/('a100_scoreonly_20261026_01' if a.model=='LG_scoreonly' else 'a100_lowlevel_capture_20261026_01'),a.audit_name) if a.model.startswith('LG_') else rental(a.model,a.audit_name)
 print(json.dumps(result,ensure_ascii=False))
