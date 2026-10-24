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
