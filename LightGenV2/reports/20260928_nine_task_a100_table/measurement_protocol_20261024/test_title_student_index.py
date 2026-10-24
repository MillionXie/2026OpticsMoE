"""Only canonical enrollment and aggregation; no model/CUDA or real data reads."""
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import build_measured_index as builder


def test_fixed_rounds_never_choose_failed_foreign_or_power_gap_attempts():
    paths=builder.title_student_paths(Path('root'))
    assert [p.as_posix() for p in paths]==[
        'root/a100_title_student_input_inclusive_20261024_02/ours_repeat01',
        'root/a100_title_student_input_inclusive_20261025_04/ours_repeat02',
        'root/a100_title_student_input_inclusive_20261025_04/ours_repeat03']


def fake_audit(path,data):
    repeat=int(path.name[-2:])
    return dict(status='passed',repeat=repeat,calls=200,common_input_tokens200=9576,
        projected_seconds200=repeat,phase_estimated_j200=10*repeat,
        measured_unique100_hit1=.85,optical_ms_total=626.82,
        charged_electronic_ms_total=1000*repeat-626.82,software_wall_s=repeat+1,
        bridge_zero_reason='Pure title, no original visual path')


def test_aggregate_keeps_all_rounds_and_actual_work_not_fastest(monkeypatch):
    monkeypatch.setattr(builder,'audit_abo_title_student',fake_audit)
    result=builder.title_student_result(builder.REPO/'synthetic_run_root',Path('unused'))
    assert result['calls']==600 and result['sample_ms']==10 and result['sample_j']==.1
    assert result['common_input_tokens_total']==28728
    assert result['common_tokens_per_s']==28728/6
    assert result['measured_unique100_hit1']==[.85]*3
    assert result['full100_historical_reference_hit1']==.86
    assert result['not_a_device_measured_e2e'] is True
    assert result['optical_passes_per_query']==3 and result['online_visual_or_bridge_calls']==0
    assert [r['repeat'] for r in result['raw_references']]==[1,2,3]


def test_reject_mismatched_round_work(monkeypatch):
    def wrong(path,data):return dict(fake_audit(path,data),common_input_tokens200=999)
    monkeypatch.setattr(builder,'audit_abo_title_student',wrong)
    with pytest.raises(ValueError):builder.title_student_result(builder.REPO/'synthetic_run_root',Path('unused'))


def test_saved_shape_MAC_keeps_exact_cost_trials_and_scope(monkeypatch):
    value=dict(calls=600,no_GPU_or_neural_forwards=True,no_new_time_or_power=True,
        complete_scalar_OP_claim=False,online_vision_or_merger_calls=0,
        offline_gallery_inference_included=False,physical_optical_dense_equivalent_OP=None,
        actual_NN_MAC600=300,charged_serial_NN_MAC600=240,parallel_residual_NN_MAC600=57,
        native_position_MAC600_not_in_selected_timing_spans=3,
        per_call_first_repeat=[dict(actual_MAC=100)],
        cost_references=[dict(seconds200=2,joules200=5,summary_sha256='original',run='original_run')])
    monkeypatch.setattr(builder,'read',lambda p:value)
    monkeypatch.setattr(builder,'sha',lambda p:'a'*64)
    row=dict(ours=dict(sample_ms=2/.6,sample_j=5/600,
        raw_references=[dict(report_sha256='original',raw_run_path='original_run')]))
    builder.attach_abo_title_student_existing_shape_MAC(row)
    assert row['ours']['derived_NN_dense_MAC_subtotal']['OP600']==600
    assert row['ours']['derived_NN_dense_MAC_subtotal']['charged_serial_OP600']==480
    value['no_GPU_or_neural_forwards']=False
    with pytest.raises(ValueError):builder.attach_abo_title_student_existing_shape_MAC(row)
    value['no_GPU_or_neural_forwards']=True;value['cost_references'][0]['run']='other_run'
    with pytest.raises(ValueError):builder.attach_abo_title_student_existing_shape_MAC(row)
