"""CPU-only supplementary audit; cannot qualify a failed original A100 leaf."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from audit_lgvq_async_point_20261026 import REPO,sha,verify_receipt


def validate(info,partitions=2):
    if partitions not in (2,4):raise ValueError('Explicit audited partition count required')
    expected=dict(profile=f'native_Conv3d_{"two" if partitions==2 else "four"}_stream_complete_patch_batches',
        class_source_sha256='b80abab0ebe915256e6c0630c70472670e4e6aa99d6a8d1f4c0cfc62cd7977e0',
        kernel=[2,16,16],stride=[2,16,16],weight_shape=[1024,3,2,16,16],
        weight_stride=[1536,512,256,16,1],weight_dtype='torch.bfloat16',bias_dtype='torch.bfloat16',
        original_math_geometry_weight_bias_precision_retained=True,
        original_patch_forward_unchanged=True,original_cold_fields_and_captures_retained=True,
        explicit_extra_warmup_forwards=0,all_branch_and_join_costs_inside_prefix=True,
        fixed_equal_partitions=partitions,python_forward_calls=5,native_branch_calls=5*partitions,
        fork_after_input=True,join_before_concat=True,output_order='original_left_then_right',
        backend_policy_unchanged=True,hardware_scope='ORIGINAL_A100_250W_108SM',
        original_A100_reference_retained=True,all_original_math_and_timing_gates_retained=True)
    if any(info.get(k)!=v for k,v in expected.items()):
        raise ValueError('Original native batch-parallel scope or actual invocation counts differ')
    counts=info.get('input_patch_counts',[])
    if len(counts)!=5 or len(set(counts))!=1 or any(type(n)!=int or n<partitions or n%partitions for n in counts):
        raise ValueError('Exactly three eager/two captured complete-patch forwards required')


def audit(folder,out,partitions=2):
    folder=Path(folder);out=Path(out)
    if out.exists():raise FileExistsError('Preserve existing parallel audit')
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8-sig'))
    base=read('local_independent_async_audit01.json')
    if (base['status']!='passed' or base['all_2048_scores_bitwise_equal_sealed_reference'] is not True
            or base['four_cycles_bitwise_equal'] is not True or base['every_original_DAG_node_start_reproduced'] is not True):
        raise ValueError('Original independent numeric/power/DAG audit must pass first')
    verify_receipt(folder,read('backup_receipt_complete01.json'))
    for name,digest in base['raw_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('Original audited raw changed')
    info=read('a100_patch_parallel_identity.json');validate(info,partitions)
    identity=read('identity.json');sources=identity['sources']
    required={'run_lgvq_async_20261026.py','run_lgvq_a100_patch_parallel_20261027.py','lgvq_a100_patch_parallel_20261027.py'}
    if partitions==4:required.add('run_lgvq_a100_patch_fourway_20261027.py')
    if not required<={Path(n).name for n in sources}:raise ValueError('Original/new closure incomplete')
    for name,digest in sources.items():
        data=subprocess.check_output(['git','show',identity['published_commit']+':'+name],cwd=REPO,timeout=10)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Published source blob differs')
    result=dict(status='passed',all_frozen_source_blobs_verified=len(sources),parallel_identity=info,
        base_audit_sha256=sha(folder/'local_independent_async_audit01.json'),
        parallel_identity_sha256=sha(folder/'a100_patch_parallel_identity.json'),auditor_sha256=sha(__file__),
        original_full_A100_numeric_power_timing_DAG_audit_retained=True,physical_optical_hardware_measured=False,
        GPU_overlap_not_inferred_from_stream_count=True)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('leaf',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--partitions',type=int,choices=[2,4],default=2)
    a=p.parse_args();r=audit(a.leaf,a.out,a.partitions)
    print(json.dumps({'status':r['status'],'source_blobs':r['all_frozen_source_blobs_verified']},ensure_ascii=True))
