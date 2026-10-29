"""CPU-only replay of SHA-preserved new rental roots, never model execution."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
SIM=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
EXPECTED=[f'{m}_cap{c}_repeat{r:02d}' for m in ('kimi','deepseek') for c in (50,100) for r in (1,2,3)]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=True,indent=2,allow_nan=False)

def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6):raise ValueError(f'CPU replay differs: {a} != {b}')

def preserved(root):
    receipt=read(root/'whole_root_backup_receipt_complete01.json')
    if receipt['status']!='complete_original_root_local_and_lab_SHA_verified':raise ValueError('Complete dual-site original-byte receipt required')
    for r in receipt['files']:
        name=Path(r['name'])
        if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe evidence member')
        p=root/name
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Preserved original evidence differs')
    return sha(root/'whole_root_backup_receipt_complete01.json')

def baselines():
    import run_rental_energy_remeasure_20261029 as entry
    root=SIM/'rental_energy_remeasure_20261030_03';out=root/'local_independent_pooled_audit01.json'
    if out.exists():raise FileExistsError('Do not repeat accepted independent audit')
    receipt_sha=preserved(root);s=read(root/'suite_status.json')
    prior=SIM/'rental_energy_remeasure_20261029_01';prior_sha=preserved(prior);old=read(prior/'suite_status.json')
    second=SIM/'rental_energy_remeasure_20261029_02';second_sha=preserved(second);mid=read(second/'suite_status.json')
    if old['status']!='stopped' or old['completed']!=EXPECTED[:5]:raise ValueError('Exact five-qualified-point prefix required')
    if mid['status']!='stopped' or mid['completed']!=EXPECTED[5:6] or not mid.get('mps_released'):raise ValueError('Qualified sixth point preserved across safe deadline transition')
    if s['status']!='complete' or s['completed']!=EXPECTED[6:] or s.get('mps_released') is not True:raise ValueError('All six remaining original points must close')
    entry.configure()
    # Transport-only location of the unchanged exact-SHA receipt, not a gate change.
    entry.timing.STORAGE_RECEIPT=root/'evidence/input_local_overlay_receipt01.json'
    manifest=entry.base.TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/provisioning/original_manifest.csv'
    audits=[]
    for name in EXPECTED:
        leaf=(prior if name in EXPECTED[:5] else second if name==EXPECTED[5] else root)/name
        report=read(leaf/'report.json')
        source='LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/rental_power_observer_20261029.py'
        blob=subprocess.check_output(['git','show',report['git_commit']+':'+source],cwd=REPO,timeout=15)
        if hashlib.sha256(blob).hexdigest()!=report['sources'][source]:raise ValueError('Original observer source pin differs')
        versions=[ast.literal_eval(n.value) for n in ast.parse(blob.decode()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VERSION' for t in n.targets)]
        if len(versions)!=1:raise ValueError('Exact declared observer version required')
        current=entry.power.VERSION
        try:
            entry.power.VERSION=versions[0]  # Replay each original declared version, not today's source label.
            a=entry.audit_point(leaf,manifest)
        finally:entry.power.VERSION=current
        if a['status']!='passed' or not a['energy_qualified'] or a['measured_videos']!=512:raise ValueError('New time+energy audit failed')
        audits.append(a)
    if 'torch' in sys.modules:raise RuntimeError('Baseline audit must not import Torch')
    value=dict(status='passed',round_audits=audits,original_root_receipt_sha256=receipt_sha,prior_root_receipt_sha256=prior_sha,second_root_receipt_sha256=second_sha,
               qualification='five original plus one qualified point plus six completion points; failed267-video attempt and unmeasured loading transition excluded and preserved',
               actual_board_power_and_energy=True,host_power_is_fixed338_2W_proxy=True,
               full_window_replayed=True,original250ms_gate_retained=True,model_loaded=False,torch_imported=False)
    save(out,value);return dict(status='passed',path=str(out),sha256=sha(out),rounds=len(audits))

def lightgen():
    import numpy as np
    import lgvq_compute_scope_20261029 as scope
    import lgvq_resize_norm_20261029 as prep
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    root=SIM/'lightgen_rental_20261029_01';out=root/'local_independent_cpu_audit01.json'
    if out.exists():raise FileExistsError('Do not repeat accepted independent audit')
    receipt_sha=preserved(root);s=read(root/'suite_status.json');leaf=root/'lg_repeat01'
    if s['status']=='stopped':return lightgen_diagnostic(root,receipt_sha)
    if s['status']!='complete' or s['completed']!=['lg_repeat01'] or not s.get('mps_released'):raise ValueError('Single complete LightGen round required')
    r=read(leaf/'compute_report.json');raw=read(leaf/'raw_operator_records.json');ident=read(leaf/'compute_identity.json')
    for name,key in [('raw_kernel_trace.json','raw_trace_sha256'),('raw_operator_records.json','raw_operator_records_sha256'),('outputs.npz','outputs_sha256')]:
        if sha(leaf/name)!=r[key]:raise ValueError('LightGen original report hash differs')
    frozen=read(root/'source_receipt.json')
    for name,digest in frozen['sources'].items():
        blob=subprocess.check_output(['git','show',frozen['published_commit']+':'+name],cwd=REPO,timeout=15)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published LightGen source differs')
    reference=SIM/'a100_lgvq_graph_pipeline_20261026_03/lg_repeat01/graph/graph_outputs.npz'
    if sha(reference)!=ident['reference_sha256'] or ident['unique_videos']!=512 or ident['explicit_extra_warm_fields']!=3:raise ValueError('Original reference population differs')
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction'][:32].copy()
    with np.load(leaf/'outputs.npz',allow_pickle=False) as f:actual=f['prediction'];warm=f['warm_prediction']
    if actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes()!=expected.tobytes() or warm.shape!=(3,16) or warm.tobytes()!=expected[:3].tobytes():raise ValueError('All512 and warm outputs must match exactly')
    with (leaf/'telemetry.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if any(x['utilization_percent'] or x['memory_used_mib'] for x in rows):raise ValueError('No invented telemetry diagnostics')
    samples=[Sample(float(x['host_time_unix']),float(x['host_monotonic_s']),x['phase'],float(x['watts']),None,None) for x in rows]
    energy=interval_energy([(raw['formal_monotonic_start'],raw['formal_monotonic_end'])],samples,host_w=0.)
    for key in ['energy_j','elapsed_s','max_power_sample_age_s']:close(energy[key],r['diagnostic_FFT_board_energy'][key])
    close(raw['formal_host_end']-raw['formal_host_start'],r['actual_FFT_software_wall_s'])
    kernels=scope.traced_kernels(read(leaf/'raw_kernel_trace.json'))
    if kernels!=read(leaf/'kernel_phase_records.json'):raise ValueError('Original kernel phase replay differs')
    schedule=prep.summarize_compute(kernels,raw['patch'],raw['tokenizer'],32)
    close(schedule['time16_s'],r['time16_s'])
    if schedule['schedule']!=r['schedule']:raise ValueError('Original resource-schedule replay differs')
    if 'torch' in sys.modules:raise RuntimeError('LightGen audit must not import Torch')
    value=dict(status='passed',report_sha256=sha(leaf/'compute_report.json'),original_root_receipt_sha256=receipt_sha,
               all512_and_three_warm_fields_bitwise_equal=True,raw_kernel_phase_and_schedule_replayed=True,
               actual_FFT_wall_replayed=True,board_energy_replayed=energy,original250ms_gate_retained=True,
               physical_photonic_hardware_measured=False,model_loaded=False,torch_imported=False)
    save(out,value);return dict(status='passed',path=str(out),sha256=sha(out),scores=512)

def lightgen_diagnostic(root,receipt_sha):
    import numpy as np
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    leaf=root/'lg_repeat01';out=root/'local_independent_cpu_audit01.json';numeric=read(leaf/'numeric_audit.json')
    if read(leaf/'status.json')['status']!='failed' or numeric['all_formal_scores_bitwise_equal']:raise ValueError('Exact failed numerical gate evidence required')
    identity=read(leaf/'compute_identity.json');raw=read(leaf/'raw_operator_records.json')
    ref=SIM/'a100_lgvq_graph_pipeline_20261026_03/lg_repeat01/graph/graph_outputs.npz'
    if sha(ref)!=identity['reference_sha256']:raise ValueError('Original reference differs')
    with np.load(ref,allow_pickle=False) as f:expected=f['prediction'][:32].copy()
    with np.load(leaf/'outputs.npz',allow_pickle=False) as f:actual=f['prediction'];warm=f['warm_prediction']
    if actual.shape!=(32,16) or actual.dtype!=expected.dtype or warm.shape!=(3,16):raise ValueError('Actual complete512 output population required')
    equal=int(np.count_nonzero(actual.view('uint32')==expected.view('uint32')));difference=float(np.max(np.abs(actual-expected)))
    if equal!=numeric['equal_scores'] or difference!=numeric['maximum_absolute_difference']:raise ValueError('Saved numeric failure replay differs')
    fields=[x for x in raw['host_records'] if not x['warm']]
    if [x['field'] for x in fields]!=list(range(32)):raise ValueError('All formal fields required')
    elapsed=raw['formal_host_end']-raw['formal_host_start']
    if elapsed<=0:raise ValueError('Actual positive software window required')
    frozen=read(root/'source_receipt.json')
    for name,digest in frozen['sources'].items():
        b=subprocess.check_output(['git','show',frozen['published_commit']+':'+name],cwd=REPO,timeout=15)
        if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Original published source differs')
    with (leaf/'telemetry.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    samples=[Sample(float(x['host_time_unix']),float(x['host_monotonic_s']),x['phase'],float(x['watts']),None,None) for x in rows]
    energy=interval_energy([(raw['formal_monotonic_start'],raw['formal_monotonic_end'])],samples,host_w=0.)
    value=dict(status='diagnostic_only_numeric_gate_failed',source=leaf.relative_to(REPO).as_posix(),numeric_audit_sha256=sha(leaf/'numeric_audit.json'),
               original_root_receipt_sha256=receipt_sha,actual_FFT_software_window_s=elapsed,actual_FFT_s_per_16_videos=elapsed/32,
               measured_videos=512,formal_fields=32,extra_warm_fields=3,equal_scores=equal,maximum_absolute_difference=difference,
               gpu_board_energy_j=energy['energy_j'],gpu_board_mean_w=energy['energy_j']/energy['elapsed_s'],board_energy_replayed=energy,
               physical_photonic_hardware_measured=False,numeric_gate_passed=False,formal_qualified_result=False,
               model_loaded=False,torch_imported='torch' in sys.modules,raw_kernel_trace_sha256=sha(leaf/'raw_kernel_trace.json'),
               kernel_schedule_replayed=False,full_fresh_process_monotonic_wall_not_saved=True)
    if value['torch_imported']:raise RuntimeError('CPU-only diagnostic must not import Torch')
    save(out,value);return dict(status=value['status'],path=str(out),sha256=sha(out),software_window_s=elapsed,equal_scores=equal)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('baselines','lightgen'),required=True);a=p.parse_args()
    print(json.dumps((baselines if a.phase=='baselines' else lightgen)(),ensure_ascii=True))
