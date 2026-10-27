"""Independent data-only warmed audit; cannot qualify original cold table."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import audit_lgvq_async_point_20261026 as base
from audit_lgvq_a100_patch_parallel_20261027 import validate as validate_patch
import lgvq_a100_warmup_20261027 as warm


def audit(folder,out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise FileExistsError('Preserve warmed audit')
    read=lambda n:json.loads((folder/n).read_text(encoding='utf8'))
    identity=read('identity.json');sources=identity['sources']
    required={'run_lgvq_a100_warmup_20261027.py','lgvq_a100_warmup_20261027.py','run_lgvq_async_20261026.py'}
    if not required<={Path(n).name for n in sources}:raise ValueError('Explicit warmed source closure required')
    for name,digest in sources.items():
        blob=subprocess.check_output(['git','show',identity['published_commit']+':'+name],cwd=base.REPO,timeout=10)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published warmed source differs')
    patch=read('a100_patch_parallel_identity.json')
    if patch['python_forward_calls']!=6 or patch['native_branch_calls']!=12 or len(patch['input_patch_counts'])!=6:
        raise ValueError('One warm plus original three eager/two real captures required')
    # Validate unchanged geometry independently of the explicitly checked extra call.
    validate_patch(dict(patch,python_forward_calls=5,native_branch_calls=10,input_patch_counts=patch['input_patch_counts'][1:]),2)
    if len(set(patch['input_patch_counts']))!=1:raise ValueError('Complete warm field geometry differs')
    raw=read('async_raw_timing.json');extra=read('extra_warmup_identity.json')
    base.compare(extra,raw['extra_warmup'],'warm identity')
    info=warm.accounting(raw)
    with np.load(folder/'extra_warmup_outputs.npz',allow_pickle=False) as f:actual=f['prediction']
    reference=base.REPO/base.REFERENCE_RELATIVE
    if base.sha(reference)!=base.REFERENCE_SHA256:raise ValueError('Original reference changed')
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction'][0:1]
    if actual.shape!=(1,16) or actual.dtype!=np.float32 or actual.tobytes()!=expected.tobytes():
        raise ValueError('All 16 warm scores must be original bitwise reference')
    passes=[e for e in extra['six_pass_events'] if e['kind']=='optical_simulation']
    from lgvq_phase_ledger_20261026 import PASSES
    if tuple(e['label'] for e in passes)!=PASSES or any(e['cuda_end_ms']<e['cuda_start_ms'] for e in extra['six_pass_events']):
        raise ValueError('Exactly six unchanged complete warm propagations required')
    if extra['tokenizer']['field']!=-1 or extra['tokenizer']['input_ids']!=raw['records'][0]['tokenizer']['input_ids']:
        raise ValueError('Actual extra prompt tokenization required')
    if read('async_identity.json').get('timing_profile')!=warm.PROFILE:raise ValueError('Warm profile not labelled before run')
    previous=base.adapt;base.adapt=warm.adapt
    try:replayed=base.audit(folder,folder/'local_warmed_data_replay01.json')
    finally:base.adapt=previous
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    with (folder/'telemetry.csv').open(encoding='utf-8-sig',newline='') as f:
        samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],float(r['watts']),
                    float(r['utilization_percent']),float(r['memory_used_mib'])) for r in csv.DictReader(f)]
    a,b=raw['perf_monotonic_alignment_bracket'];offset=(a+b)/2-raw['host_start']
    energy=interval_energy([(extra['host_start']+offset,raw['host_end']+offset)],samples,host_w=0.)
    if energy['max_power_sample_age_s']>.25:raise ValueError('Inclusive warm power support exceeds original age gate')
    if not math.isclose(info['warmup_software_ms']+info['warm_to_main_gap_ms']+info['measured_main_software_ms'],info['inclusive_software_ms'],abs_tol=1e-7):
        raise ValueError('Warm-inclusive software wall does not close')
    result=dict(status='passed',timing_profile=warm.PROFILE,original_cold_scope_qualified=False,
        qualified_for_final_hybrid_table=False,all_frozen_source_blobs_verified=len(sources),
        warm_16_and_main_2048_scores_bitwise_equal=True,all_main_DAG_nodes_reproduced=True,
        warmup_accounting=info,inclusive_actual_FFT_board_energy=energy,
        base_warmed_data_audit_sha256=base.sha(folder/'local_warmed_data_replay01.json'),
        raw_sha256=replayed['raw_sha256'],warmup_identity_sha256=base.sha(folder/'extra_warmup_identity.json'),
        warmup_outputs_sha256=base.sha(folder/'extra_warmup_outputs.npz'),auditor_sha256=base.sha(__file__),
        physical_optical_hardware_measured=False,prewarm_optical_projection_not_inferred_from_software_wall=True)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('leaf',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();r=audit(args.leaf,args.out)
    print(json.dumps({k:v for k,v in r.items() if k not in ('raw_sha256',)},ensure_ascii=True))
