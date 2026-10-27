"""Independent CPU replay of the new explicit shared-launch observation schema."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import audit_lgvq_async_point_20261026 as base
from audit_lgvq_a100_patch_parallel_20261027 import validate as validate_patch
from lgvq_a100_batch_raw_adapter_20261027 import adapt
import lgvq_async_worker_20261026 as native


def audit(folder,out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise FileExistsError('Preserve existing batch audit')
    read=lambda n:json.loads((folder/n).read_text(encoding='utf8'))
    identity=read('identity.json');sources=identity['sources']
    required={'run_lgvq_async_20261026.py','run_lgvq_a100_patch_parallel_20261027.py',
        'lgvq_a100_patch_parallel_20261027.py','run_lgvq_a100_batch_replay_20261027.py',
        'lgvq_a100_batch_replay_20261027.py','lgvq_a100_batch_raw_adapter_20261027.py'}
    if not required<={Path(n).name for n in sources}:raise ValueError('Original and composed source closure required')
    for name,digest in sources.items():
        blob=subprocess.check_output(['git','show',identity['published_commit']+':'+name],cwd=base.REPO,timeout=10)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published source differs: '+name)
    validate_patch(read('a100_patch_parallel_identity.json'),2)
    receipt=read('a100_body_batch_identity.json')
    if receipt!=dict(original_worker_source_sha256=hashlib.sha256(inspect.getsource(native.worker).encode()).hexdigest(),
        only_two_imports_redirected=True,first_three_eager_and_two_real_captures_unchanged=True):
        raise ValueError('Only exact two worker imports may change')
    previous=base.adapt;base.adapt=adapt
    try:result=base.audit(folder,folder/'local_independent_async_audit01.json')
    finally:base.adapt=previous
    raw=read('async_raw_timing.json')
    builds=[b['slot'] for r in raw['records'] for b in r.get('body_batch_builds',[])]
    if builds!=[0,1] or sum('body_batch_submission' in r for r in raw['records'])!=123:
        raise ValueError('Exactly two compositions and 123 measured launches required')
    summary=dict(status='passed',all_frozen_source_blobs_verified=len(sources),composed_slots=2,
        actual_steady_body_host_launches=123,unchanged_ordered_child_executions=123*37,
        all_original_stage_timestamps_retained=True,all_composition_and_shared_submission_cost_charged=True,
        every_original_DAG_node_start_reproduced=True,no_ideal_overlap_or_queue_subtraction=True,
        base_audit_sha256=base.sha(folder/'local_independent_async_audit01.json'),
        auditor_sha256=base.sha(__file__),raw_sha256=result['raw_sha256'],physical_optical_hardware_measured=False)
    with out.open('x',encoding='utf8') as stream:json.dump(summary,stream,ensure_ascii=True,indent=2)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('leaf',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();r=audit(args.leaf,args.out)
    print(json.dumps({k:v for k,v in r.items() if k!='raw_sha256'},ensure_ascii=True))
