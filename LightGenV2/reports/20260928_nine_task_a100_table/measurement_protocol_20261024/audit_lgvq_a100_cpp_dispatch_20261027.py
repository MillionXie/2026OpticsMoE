"""Supplement to unchanged original raw-clock/numerics/power/DAG auditor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import audit_lgvq_async_point_20261026 as base
from audit_lgvq_a100_patch_parallel_20261027 import validate as validate_patch


def audit(folder,out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise FileExistsError('Preserve existing native audit')
    read=lambda n:json.loads((folder/n).read_text(encoding='utf8'))
    identity=read('identity.json');sources=identity['sources']
    required={'run_lgvq_async_20261026.py','lgvq_a100_cpp_dispatch_20261027.cpp',
        'lgvq_a100_cpp_dispatch_20261027.py','run_lgvq_a100_cpp_dispatch_20261027.py'}
    if not required<={Path(n).name for n in sources}:raise ValueError('Original/native closure incomplete')
    for name,digest in sources.items():
        blob=subprocess.check_output(['git','show',identity['published_commit']+':'+name],cwd=base.REPO,timeout=10)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published source differs: '+name)
    validate_patch(read('a100_patch_parallel_identity.json'),2)
    info=read('a100_cpp_dispatch_identity.json');build=info['build']
    expected=dict(profile='original_37_replays_in_one_CPP_call',slot_calls=[62,61],original_graph_replays=4551,
        original_timing_event_records=9102,no_removed_or_nested_graph_replays=True,
        all_original_host_costs_inside_continuous_wall=True,per_segment_CPU_timestamps_are_measured=True,
        no_clock_shift_or_ideal_overlap_subtraction=True)
    if any(info.get(k)!=v for k,v in expected.items()):raise ValueError('Original replay/event counts changed')
    a,b=build['clock_bracket']
    cpp=next(n for n in sources if Path(n).name=='lgvq_a100_cpp_dispatch_20261027.cpp')
    builder=next(n for n in sources if Path(n).name=='build_lgvq_cpp_dispatch_20261027.py')
    if (not a<=build['cpp_clock_observation']<=b or build['same_clock_verified'] is not True
            or build['compiler_returncode']!=0 or build['CUDA_context_created_during_build'] is not False
            or build['source_sha256']!=sources[cpp] or build['builder_sha256']!=sources[builder]
            or len(build['binary_sha256'])!=64):
        raise ValueError('Native build or actual common-clock evidence missing')
    result=base.audit(folder,folder/'local_independent_async_audit01.json')
    summary=dict(status='passed',all_frozen_source_blobs_verified=len(sources),native_dispatch_identity=info,
        base_audit_sha256=base.sha(folder/'local_independent_async_audit01.json'),auditor_sha256=base.sha(__file__),
        original_raw_adapter_used_without_changes=True,raw_sha256=result['raw_sha256'],physical_optical_hardware_measured=False)
    with out.open('x',encoding='utf8') as stream:json.dump(summary,stream,ensure_ascii=True,indent=2)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('leaf',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();r=audit(args.leaf,args.out)
    print(json.dumps(dict(status=r['status'],source_blobs=r['all_frozen_source_blobs_verified']),ensure_ascii=True))
