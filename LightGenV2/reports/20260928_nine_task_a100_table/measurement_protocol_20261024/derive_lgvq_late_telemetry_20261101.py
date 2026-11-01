"""CPU-only admission of the completed v1r1 before a late telemetry fault.

Original failed status/raw are immutable. Only new exclusive derived files are
written. No skipped power rows inside the formal window, fabricated timestamps,
Torch import, model forward or GPU rerun is allowed.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import lgvq_front_scopes_20261101 as policy
import lgvq_compute_scope_20261029 as scope
import lgvq_async_worker_20261026 as old
import git_blob_identity_20261029 as identity

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
LEAF=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_compute_scope_20261029_15/v1_repeat01'


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def write(name,value):
    with (LEAF/name).open('x') as f:json.dump(value,f,ensure_ascii=True,indent=2,allow_nan=False)


def main():
    if (LEAF/'CPU_derivation_receipt01.json').exists():raise ValueError('Already accepted; do not repeat')
    before={p.name:sha(p) for p in LEAF.iterdir() if p.is_file()}
    failed=read(LEAF/'status.json')
    assert failed['status']=='failed' and 'Non-numeric telemetry' in failed['error']
    raw=read(LEAF/'raw_operator_records.json');compute=read(LEAF/'compute_identity.json')
    frozen=read(LEAF.parent/'source_receipt.json');identity.verify(frozen['published_commit'],frozen['sources'],REPO)
    assert compute['variant']==1 and len(raw['host_records'])==35 and len(raw['patch'])==512
    assert read(LEAF/'dynamic_CPU_validation.json')['status']=='all_512_native_CPU_bytes_verified'
    assert all(r['raw_patch_bytes_equal'] for r in raw['patch'])
    assert sum(r['warm'] for r in raw['host_records'])==3
    reference=REPO/old.REFERENCE_RELATIVE;assert sha(reference)==old.REFERENCE_SHA256
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction'][:32].copy()
    with np.load(LEAF/'outputs.npz',allow_pickle=False) as f:
        prediction=f['prediction'];warm=f['warm_prediction']
        assert prediction.shape==expected.shape and prediction.dtype==expected.dtype and prediction.tobytes()==expected.tobytes()
        assert warm.shape==(3,16) and warm.dtype==expected.dtype and warm.tobytes()==expected[:3].tobytes()
    begin,end=raw['formal_monotonic_start'],raw['formal_monotonic_end']
    errors=[json.loads(line) for line in (LEAF/'telemetry_stdout.jsonl').read_text().splitlines() if json.loads(line).get('valid_numeric_row') is False]
    assert errors and all(r['host_monotonic_s']>end for r in errors)
    with (LEAF/'telemetry.csv').open(newline='') as f:samples=list(csv.DictReader(f))
    stamps=[float(x['host_monotonic_s']) for x in samples];watts=[float(x['watts']) for x in samples]
    assert all(math.isfinite(x) for x in stamps+watts) and all(b>a for a,b in zip(stamps,stamps[1:]))
    assert all(0<w<=1000 for w in watts) and stamps[0]<=begin<end<=stamps[-1]
    spans=[(max(a,begin),min(b,end),w,a) for a,b,w in zip(stamps,stamps[1:],watts) if min(b,end)>max(a,begin)]
    elapsed=sum(b-a for a,b,w,t in spans);assert math.isclose(elapsed,end-begin,abs_tol=1e-9)
    joules=sum((b-a)*w for a,b,w,t in spans);age=max(b-t for a,b,w,t in spans)
    gap=sum(b-a for a,b,w,t in spans if b-t>.25)
    audit=read(LEAF/'process_audit.json');assert audit and all(not x['foreign_gpu_processes'] and not x['foreign_mps_clients'] for x in audit)
    kernels=scope.traced_kernels(read(LEAF/'raw_kernel_trace.json'))
    policy.VARIANT=1
    report=policy.summarize_compute(kernels,raw['patch'],raw['tokenizer'],32)
    report.update(status='complete_separate_CPU_derivation_late_telemetry_outside_formal_window',
                  all_predictions_bitwise_equal=True,actual_FFT_software_wall_s=raw['formal_host_end']-raw['formal_host_start'],
                  diagnostic_FFT_board_energy=dict(energy_j=joules,elapsed_s=elapsed,host_proxy_w=0.,
                      max_power_sample_age_s=age,strict250ms_qualified=age<=.25,
                      gap_over250ms_count=sum(b-t>.25 for a,b,w,t in spans),
                      gap_over250ms_support_s=gap,gap_over250ms_support_fraction=gap/elapsed,
                      no_synthetic_or_retimed_samples=True,late_observer_error_outside_formal_window=True),
                  raw_trace_sha256=before['raw_kernel_trace.json'],raw_operator_records_sha256=before['raw_operator_records.json'],
                  outputs_sha256=before['outputs.npz'],physical_hardware_measured=False)
    assert 'torch' not in sys.modules
    numeric=dict(all_formal_scores_bitwise_equal=True,warm_scores_bitwise_equal=True,equal_scores=512,
                 maximum_absolute_difference=0.,reference_sha256=old.REFERENCE_SHA256)
    write('derived_numeric_audit01.json',numeric)
    write('derived_kernel_phase_records01.json',kernels)
    write('derived_compute_report01.json',report)
    assert all(sha(LEAF/name)==digest for name,digest in before.items())
    result=dict(status='complete_separate_CPU_derivation',source_runtime_commit=frozen['published_commit'],
                original_failed_status_preserved=True,original_files_before_SHA256=before,
                formal_window=[begin,end],late_invalid_power_rows=errors,
                earliest_error_after_formal_end_s=min(r['host_monotonic_s'] for r in errors)-end,
                no_invalid_power_inside_formal_window=True,formal_power_bracketed_by_actual_rows=True,
                numeric=numeric,formal_scores=512,warm_scores=48,
                report_sha256=sha(LEAF/'derived_compute_report01.json'),
                kernel_records_sha256=sha(LEAF/'derived_kernel_phase_records01.json'),
                numeric_audit_sha256=sha(LEAF/'derived_numeric_audit01.json'),
                no_Torch_or_GPU=True,time16_s=report['time16_s'],unix=time.time())
    write('CPU_derivation_receipt01.json',result)
    print(json.dumps(dict(status=result['status'],time16_s=result['time16_s'],error_after_formal_s=result['earliest_error_after_formal_end_s'],formal_energy_J=joules)))


if __name__=='__main__':main()
