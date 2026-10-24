from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import audit_temporal_matrix as m


def test_new_rental_fixed_runs_never_old_dataset_once_or_failed_anchor():
    for model in ('deepseek','kimi'):
        for cap in (100,50):
            assert [p.name for p in m.paths(model,cap)]==[f'{model}_cap{cap}_repeat{i:02d}' for i in (1,2,3)]
            assert all(p.parent.name=='rental_80gb_20261025_01' for p in m.paths(model,cap))
    assert m.ANCHOR_RUN=='rental_qwen_anchor_20261025_02'


def test_two_physical_platforms_are_never_merged_or_relabelled():
    a=m.platform('qwen');b=m.platform('deepseek');c=m.platform('rental_qwen_anchor')
    assert a['power_limit_w']==250 and b['power_limit_w']==400
    assert a['gpu_uuid']!=b['gpu_uuid'] and c==b
    assert 'PCIE-40GB' in a['gpu_name'] and 'SXM4-80GB' in b['gpu_name']


def test_existing_shared40GB_qwen_triple_is_unchanged():
    assert [p.parent.name for p in m.paths('qwen',100)]==[
        'qwen100_repaired01','qwen100_remaining01','qwen100_remaining01']
