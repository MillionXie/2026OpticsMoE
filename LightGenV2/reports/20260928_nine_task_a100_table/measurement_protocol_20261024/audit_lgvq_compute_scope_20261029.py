"""Saved-data replay of kernel ownership, exact scores and resource schedule."""
import hashlib
import json
import math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def audit(leaf,destination):
    import numpy as np
    import lgvq_compute_scope_20261029 as scope
    import lgvq_async_worker_20261026 as original
    leaf=Path(leaf);status=read(leaf/'status.json');identity=read(leaf/'compute_identity.json')
    numeric=read(leaf/'numeric_audit.json');report=read(leaf/'compute_report.json')
    if status['status']!='complete' or not numeric['all_formal_scores_bitwise_equal'] or not numeric['warm_scores_bitwise_equal']:
        raise ValueError('True complete exact-output point required')
    reference=REPO/original.REFERENCE_RELATIVE
    if sha(reference)!=original.REFERENCE_SHA256:raise ValueError('Original reference SHA differs')
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction'][:32]
    with np.load(leaf/'outputs.npz',allow_pickle=False) as f:
        if f['prediction'].tobytes()!=expected.tobytes() or f['warm_prediction'].tobytes()!=expected[:3].tobytes():
            raise ValueError('Independent bitwise score replay failed')
    if identity['explicit_extra_warm_fields']!=3 or identity['cap']!=100 or identity['visible_SM']!=108:
        raise ValueError('Warmup/resource contract differs')
    for name,key in [('raw_kernel_trace.json','raw_trace_sha256'),('raw_operator_records.json','raw_operator_records_sha256'),('outputs.npz','outputs_sha256')]:
        if sha(leaf/name)!=report[key]:raise ValueError('Original file hash mismatch')
    raw=read(leaf/'raw_operator_records.json');kernels=scope.traced_kernels(read(leaf/'raw_kernel_trace.json'))
    if kernels!=read(leaf/'kernel_phase_records.json'):raise ValueError('Raw kernel attribution replay differs')
    if len([r for r in raw['host_records'] if r['warm']])!=3 or len(raw['patch'])!=512 or len(raw['tokenizer'])!=32:
        raise ValueError('Workload/cold-first-formal-field completeness differs')
    replay=scope.summarize_compute(kernels,raw['patch'],raw['tokenizer'],32)
    if replay['schedule']!=report['schedule'] or replay['equivalent_s_per_group']!=report['equivalent_s_per_group']:
        raise ValueError('Measured-kernel resource schedule replay differs')
    # Independently verify causal and resource feasibility and additive closure.
    nodes={n['id']:n for n in report['schedule']['nodes']};resources={}
    for n in report['schedule']['nodes']:
        if n['end_s']<n['start_s'] or not math.isclose(n['end_s']-n['start_s'],n['duration_s'],abs_tol=1e-10):
            raise ValueError('Negative or inconsistent computation')
        if any(nodes[p]['end_s']>n['start_s']+1e-10 for p in n['parents']):raise ValueError('Dependency violated')
        intervals=resources.setdefault(n['resource'],[])
        if any(min(b,n['end_s'])-max(a,n['start_s'])>1e-10 for a,b in intervals):raise ValueError('Shared resource overlap')
        intervals.append((n['start_s'],n['end_s']))
    total=max(n['end_s'] for n in nodes.values())
    if not math.isclose(total/32,report['time16_s'],abs_tol=1e-10) or not math.isclose(
            sum(report['equivalent_s_per_group'].values()),report['time16_s'],abs_tol=1e-10):
        raise ValueError('Equivalent group time does not close')
    for field in range(32):
        optical=[n for n in nodes.values() if n['field']==field and n['phase'].startswith('optics')]
        if len(optical)!=6 or any(n['duration_s']!=scope.OPTICAL_S for n in optical):raise ValueError('Six fixed optical cycles differ')
    audits=read(leaf/'process_audit.json')
    if not audits or any(a['foreign_gpu_processes'] or a['foreign_mps_clients'] for a in audits):
        raise ValueError('Exclusive guard evidence missing or foreign occupancy')
    if report['physical_hardware_measured'] or report['energy_qualified']:
        raise ValueError('Cannot qualify counterfactual as physical latency or measured deployment energy')
    result=dict(status='passed_COMPUTE_SCOPE_SCHEDULE_ESTIMATE',time16_s=report['time16_s'],
        measured_kernel_count=len(kernels),groups=32,unique_videos=512,warm_fields=3,
        physical_hardware_measured=False,energy_qualified=False,
        all_scores_bitwise_equal=True,causal_resources_and_additive_equivalent_time_verified=True,
        report_sha256=sha(leaf/'compute_report.json'),trace_sha256=sha(leaf/'raw_kernel_trace.json'))
    with Path(destination).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=True)
    return result

if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2]),ensure_ascii=True))
