"""Immutable enrollment and honest pending common work; no neural/GPU calls."""
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import build_measured_index as builder


def fake_audit(path):
    repeat=int(path.name[-2:])
    return dict(status='passed',repeat=repeat,calls=200,common_Qwen_input_tokens200=None,
        projected_seconds200=repeat,phase_estimated_j200=10*repeat,
        measured_first200_metrics=dict(changed_cell_accuracy=.855),optical_ms_total=1253.64,
        charged_electronic_ms_total=1000*repeat-1253.64,
        diagnostic_software_wall_s=repeat+10,diagnostic_software_board_j200=repeat+100)


def test_fixed_real_rounds_not_cpu_contract_or_historical_full_test():
    assert [p.as_posix() for p in builder.openmoji_e45_paths(Path('root'))]==[
        f'root/a100_e45_input_inclusive_20261025_01/ours_repeat{i:02d}' for i in (1,2,3)]


def test_all_rounds_phase_cost_and_no_fabricated_common_tokens(monkeypatch):
    monkeypatch.setattr(builder,'audit_openmoji_e45',fake_audit)
    r=builder.openmoji_e45_result(builder.REPO/'synthetic_root')
    assert r['calls']==600 and r['sample_ms']==10 and r['sample_j']==.1
    assert r['common_input_tokens_total'] is None and r['common_tokens_per_s'] is None
    assert r['full1000_historical_simulation_changed_cell_accuracy']==.8765
    assert r['full1000_historical_laboratory_changed_cell_accuracy']==.8655
    assert [v['changed_cell_accuracy'] for v in r['measured_first200_metrics']]==[.855]*3
    assert [v['repeat'] for v in r['raw_references']]==[1,2,3]
    assert r['optical_passes_per_query']==6 and r['not_a_device_measured_e2e'] is True
    assert r['diagnostic_actual_software_seconds_total']==36


@pytest.mark.parametrize('extra',[dict(calls=199),dict(common_Qwen_input_tokens200=1234),dict(repeat=1)])
def test_reject_incomplete_or_invented_work(monkeypatch,extra):
    monkeypatch.setattr(builder,'audit_openmoji_e45',lambda p:dict(fake_audit(p),**extra))
    with pytest.raises(ValueError):builder.openmoji_e45_result(builder.REPO/'synthetic_root')
