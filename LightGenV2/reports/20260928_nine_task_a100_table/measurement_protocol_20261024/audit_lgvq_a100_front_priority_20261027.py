"""Priority supplement; same original numeric/power/raw-DAG audit unchanged."""
import argparse
import json
from pathlib import Path
from audit_lgvq_a100_cpp_dispatch_20261027 import audit as cpp_audit
from audit_lgvq_async_point_20261026 import sha


def audit(folder,out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise FileExistsError('Preserve prior priority audit')
    read=lambda n:json.loads((folder/n).read_text(encoding='utf8'))
    info=read('a100_front_priority_identity.json')
    if info!=dict(frontend_priority=-1,body_priority=0,original_streams_and_dependencies_retained=True,
                  no_ideal_overlap_or_queue_subtraction=True):raise ValueError('Actual frontend/body priorities differ')
    if read('a100_patch_parallel_identity.json').get('auxiliary_stream_priorities')!=[-1]:
        raise ValueError('Actual auxiliary patch stream priority differs')
    result=cpp_audit(folder,folder/'local_independent_cpp_dispatch_audit01.json')
    summary=dict(status='passed',priority_identity=info,source_blobs=result['all_frozen_source_blobs_verified'],
        full_cpp_audit_sha256=sha(folder/'local_independent_cpp_dispatch_audit01.json'),auditor_sha256=sha(__file__),
        original_raw_adapter_and_all_numeric_power_timing_gates_retained=True)
    with out.open('x',encoding='utf8') as stream:json.dump(summary,stream,ensure_ascii=True,indent=2)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('leaf',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();r=audit(args.leaf,args.out);print(json.dumps(r,ensure_ascii=True))
