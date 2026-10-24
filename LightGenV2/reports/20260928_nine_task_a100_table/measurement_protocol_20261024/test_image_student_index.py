from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import build_measured_index as builder


def fake(path,data):
    i=int(path.name[-2:])
    return dict(status='passed',repeat=i,calls=200,common_input_tokens200=13728,
        original_bank_sha256='original bank',projected_seconds200=i,phase_estimated_j200=10*i,
        first200_diagnostic_hit1=.67,optical_ms_total=1253.64,charged_electronic_ms_total=i*1000,
        software_wall_s=i+1)


def test_only_complete_original_triple_not_interrupted_runs_or_accuracy_replacement(monkeypatch):
    monkeypatch.setattr(builder,'audit_abo_image_student',fake)
    root=builder.REPO/'unit_fixture_not_real_data'
    result=builder.image_student_result(root,Path('unused'))
    assert all('a100_image_student_input_inclusive_20261025_04' in str(p) for p in builder.image_student_paths(root))
    assert result['calls']==600 and result['sample_ms']==10 and result['sample_j']==.1
    assert result['common_input_tokens_total']==41184
    assert result['full2400_historical_reference_hit1']==.8058333333333333
    assert result['measured_first200_diagnostics_only']==[.67]*3
    assert result['optical_passes_per_query']==6 and result['actual_native_merger_calls']==600


@pytest.mark.parametrize('field,value',[('calls',199),('common_input_tokens200',0),('repeat',2)])
def test_reject_wrong_round_work(monkeypatch,field,value):
    monkeypatch.setattr(builder,'audit_abo_image_student',lambda path,data:dict(fake(path,data),**{field:value}))
    with pytest.raises(ValueError):builder.image_student_result(builder.REPO/'unit_fixture_not_real_data',Path('unused'))
