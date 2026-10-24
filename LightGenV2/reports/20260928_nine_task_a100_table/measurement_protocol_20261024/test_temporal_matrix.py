"""CPU-only exact run mapping and pooled (not fastest-trial) denominator checks."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('temporal_matrix',Path(__file__).with_name('audit_temporal_matrix.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_canonical_paths_no_rejected_or_duplicate_trials():
    allpaths=[str(p) for model in ('qwen','glm','clip','yolo') for cap in (100,50) for p in m.paths(model,cap)]
    assert len(allpaths)==len(set(allpaths))==24
    assert not any('qwen100_shared' in p for p in allpaths)
    assert str(m.paths('glm',100)[2]).replace('\\','/').endswith('lab_remaining02/glm100_r3/glm_cap100_repeat03')
    assert str(m.paths('qwen',100)[0]).replace('\\','/').endswith('qwen100_repaired01/qwen_cap100_repeat01')


def test_three_repeat_aggregate_uses_all600_and_time_weighted_power():
    audits=[dict(status='passed',first200_ids_sha256='same',report_sha256=str(i),
        cost=dict(continuous_wall_s=s,estimated_system_energy_j=s*400,gpu_energy_j=s*p,
                  ms_per_video=s*5,estimated_joules_per_video=s*2)) for i,(s,p) in enumerate(((2,30),(4,40),(6,50)))]
    reports=[dict(effective_input_tokens200=100,performance200=dict(srcc=.7),original_full558_reference=dict(srcc=.8))]*3
    a=m.aggregate(reports,audits)
    assert a['sample_ms']==20 and a['equivalent_frame_ms']==5
    assert a['sample_j']==8 and a['common_input_tokens_total']==300
    assert a['time_weighted_board_w']==pytest.approx(43.3333333333333)
    with pytest.raises(ValueError):m.aggregate(reports[:2],audits[:2])
