"""Immutable enrollment and honest pending common work; no neural/GPU calls."""
from pathlib import Path
import csv
import io
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


def test_new_native_pair_derives_common_work_without_editing_ours_raw(monkeypatch):
    monkeypatch.setattr(builder,'audit_openmoji_e45',fake_audit)
    monkeypatch.setattr(builder,'audit_openmoji_qwen',lambda p:dict(status='passed',repeat=int(p.name[-2:]),calls=200,
        common_Qwen_input_tokens200=12807,continuous_wall_s=10,system_energy_j=4000,board_energy_j=500,
        formal_full1000_reference_metrics={'changed_cell_accuracy':.812},measured_first200_diagnostics_only={'changed_cell_accuracy':.8}))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['index','sample_id','task','instruction','instruction_utf8_sha256','valid_input_tokens'])
    writer.writeheader()
    for i in range(200):writer.writerow(dict(index=i,sample_id=f'test{i}',task='add',instruction='original',instruction_utf8_sha256='abc',valid_input_tokens=71 if i==0 else 64))
    data=stream.getvalue()
    monkeypatch.setattr(Path,'open',lambda *a,**k:io.StringIO(data))
    own=builder.openmoji_e45_result(builder.REPO/'synthetic_root')
    native,common=builder.openmoji_pair(builder.REPO/'synthetic_root',own)
    assert native['calls']==600 and common['total']==38421
    assert own['common_input_tokens_total']==38421
    assert common['ours_per_s']/common['qwen_per_s']==common['time_ratio']==5
    assert own['raw_references'][0]['common_Qwen_input_tokens200'] is None
    assert native['formal_full1000_reference_metrics']['changed_cell_accuracy']==.812
    assert native['measured_first200_diagnostics_only'][0]['changed_cell_accuracy']==.8
