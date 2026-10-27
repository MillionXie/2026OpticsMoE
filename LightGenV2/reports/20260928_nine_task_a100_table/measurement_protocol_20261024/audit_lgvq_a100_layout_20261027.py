"""CPU-only supplement to the original complete A100 async audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from audit_lgvq_async_point_20261026 import REPO,sha,verify_receipt


def validate_layout(info):
    if (info['hardware_scope']!='ORIGINAL_A100_250W_108SM'
            or info['actual_original_kernel']!=[2,16,16] or info['actual_original_stride']!=[2,16,16]
            or info['weight_shape']!=[1024,3,2,16,16]
            or info['weight_dtype']!='torch.bfloat16' or info['bias_dtype']!='torch.bfloat16'
            or info['class_source_sha256']!='b80abab0ebe915256e6c0630c70472670e4e6aa99d6a8d1f4c0cfc62cd7977e0'
            or info['original_convolution_and_bias_retained'] is not True
            or info['original_A100_reference_retained'] is not True
            or info['all_original_math_and_timing_gates_retained'] is not True
            or info['input_layout_conversion_inside_actual_forward'] is not True
            or info['weight_values_or_precision_changed'] is not False
            or info['GEMM_or_geometry_replacement'] is not False or info['extra_forward_calls']!=0):
        raise ValueError('Original A100 biased BF16 convolution/layout-only scope required')


def audit(folder,out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise FileExistsError('Preserve existing layout audit')
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8-sig'))
    base=read('local_independent_async_audit01.json')
    if (base['status']!='passed' or base['all_2048_scores_bitwise_equal_sealed_reference'] is not True
            or base['four_cycles_bitwise_equal'] is not True or base['every_original_DAG_node_start_reproduced'] is not True):
        raise ValueError('Original independent A100 audit must pass first')
    verify_receipt(folder,read('backup_receipt_complete01.json'))
    for name,digest in base['raw_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('Original audited raw changed')
    info=read('a100_layout_identity.json');validate_layout(info)
    identity=read('identity.json');sources=identity['sources']
    required={'run_lgvq_async_20261026.py','run_lgvq_a100_layout_20261027.py','lgvq_development_conv_layout_20261027.py'}
    if not required<={Path(n).name for n in sources}:raise ValueError('Original and layout source closure incomplete')
    for name,digest in sources.items():
        data=subprocess.check_output(['git','show',identity['published_commit']+':'+name],cwd=REPO,timeout=10)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Published original/layout source blob differs')
    result=dict(status='passed',all_frozen_source_blobs_verified=len(sources),layout_identity=info,
        base_audit_sha256=sha(folder/'local_independent_async_audit01.json'),
        layout_identity_sha256=sha(folder/'a100_layout_identity.json'),auditor_sha256=sha(__file__),
        original_full_A100_numeric_power_timing_DAG_audit_retained=True,physical_optical_hardware_measured=False)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('leaf',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();r=audit(a.leaf,a.out)
    print(json.dumps({'status':r['status'],'source_blobs':r['all_frozen_source_blobs_verified']},ensure_ascii=True))
