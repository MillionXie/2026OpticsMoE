"""Immutable pooled timings after all original local data audits pass."""
import hashlib
import json
import math
from pathlib import Path
import sys

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def collect(lg,qwen,destination,*,extra_metadata=None):
    lg=Path(lg);qwen=Path(qwen);state=read(lg/'suite_status.json')
    names=[f'lg_repeat{i:02d}' for i in (1,2,3)]
    if state['status']!='complete' or state['completed']!=names or not state['mps_released']:
        raise ValueError('Complete released fresh LG triple required')
    baseline=read(qwen/'local_independent_pooled_audit01.json')
    if baseline['status']!='passed' or baseline['videos']!=1536:raise ValueError('Audited matched Qwen triple required')
    rounds=[]
    for name in names:
        p=lg/name;audit=read(p/'local_independent_compute_audit01.json');report=read(p/'compute_report.json')
        if audit['status']!='passed_COMPUTE_SCOPE_SCHEDULE_ESTIMATE' or audit['report_sha256']!=sha(p/'compute_report.json'):
            raise ValueError('Exact independently audited original LG round required')
        rounds.append(dict(time16_s=report['time16_s'],total_s=report['schedule']['total_s'],groups=report['fields'],
            equivalent_s_per_group=report['equivalent_s_per_group'],report_sha256=sha(p/'compute_report.json'),
            audit_sha256=sha(p/'local_independent_compute_audit01.json'),all_scores_bitwise_equal=report['all_predictions_bitwise_equal']))
    groups=sum(r['groups'] for r in rounds);seconds=sum(r['total_s'] for r in rounds);lg16=seconds/groups
    qwen_tokens=sum(read(qwen/f'qwen_cap50_repeat{i:02d}'/'report.json')['effective_input_tokens512'] for i in (1,2,3))
    equivalent={}
    for r in rounds:
        for key,value in r['equivalent_s_per_group'].items():equivalent[key]=equivalent.get(key,0)+value*r['groups']/groups
    if not math.isclose(sum(equivalent.values()),lg16,abs_tol=1e-9):raise ValueError('Pooled equivalent time does not close')
    result=dict(status='PASSED_NEW_COMPUTE_SCOPE_TIMING_TRIPLE',model='LightGen-II',unique_videos_per_round=512,
        groups_per_round=32,videos_per_group=16,frames_per_group=64,extra_complete_warm_fields_per_round=3,
        LG_time16_s=lg16,LG_frame_s=lg16/64,LG_frames_per_s=64/lg16,Qwen_time16_s=baseline['time16_s'],
        Qwen_frame_s=baseline['time16_s']/64,Qwen_frames_per_s=64/baseline['time16_s'],
        speed_ratio=baseline['time16_s']/lg16,passed_100x_new_declared_scope=lg16<baseline['LG_100x_threshold_s'],
        threshold_100x_s=baseline['LG_100x_threshold_s'],Qwen_effective_tokens=qwen_tokens,
        LG_tokens_per_s=qwen_tokens/seconds,Qwen_tokens_per_s=qwen_tokens/baseline['total_continuous_s'],
        tokens_per_s_ratio=baseline['total_continuous_s']/seconds,
        LG_tokens_basis='same ordered Qwen input-token counts; no change to other model token conventions',
        equivalent_s_per_group=equivalent,rounds=rounds,Qwen_audit_sha256=sha(qwen/'local_independent_pooled_audit01.json'),
        LG_time_basis='measured CPU computation and CUDA kernels with fixed optical cycles, modelled two-field resource schedule',
        Qwen_time_basis='measured cold B1 MPS50 continuous file preparation to CPU score, zero inference warmups',
        physical_photonic_hardware_time_measured=False,matched_end_to_end_physical_speedup_claim=False,
        energy_qualified=False,energy_note='No FFT-context NVML substitution for physical hybrid deployment GPU energy')
    if extra_metadata:result.update(extra_metadata)
    with Path(destination).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=True,allow_nan=False)
    return result

if __name__=='__main__':
    r=collect(*sys.argv[1:]);print(json.dumps({k:v for k,v in r.items() if k not in ('rounds','equivalent_s_per_group')},ensure_ascii=True))
