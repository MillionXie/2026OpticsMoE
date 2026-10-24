from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run_t12_baseline_contract import predecessor_ready,predecessor_file_ready,BODY_SHA,frozen_sources


def proof():return dict(status='passed',no_gpu=True,body_sha256=BODY_SHA,
    checks=[dict(test_index=i,bitwise_equal=True) for i in (200,2303)])


def test_pending_is_not_passed_and_stopped_cannot_be_bypassed(tmp_path):
    assert not predecessor_file_ready(tmp_path)
    assert not predecessor_ready(dict(status='waiting_previous_queue'))
    with pytest.raises(RuntimeError):predecessor_ready(dict(status='stopped'))


def test_actual_original_cpu_proof_sha_inputs_required():
    state=dict(status='complete',contract_sha256='pinned')
    assert predecessor_ready(state,proof(),'pinned')
    with pytest.raises(ValueError):predecessor_ready(state,proof(),'different')
    wrong=proof();wrong['checks'][0]['bitwise_equal']=False
    with pytest.raises(ValueError):predecessor_ready(state,wrong,'pinned')
    wrong=proof();wrong['body_sha256']='other'
    with pytest.raises(ValueError):predecessor_ready(state,wrong,'pinned')


def test_frozen_original_baseline_sources_include_asset_identity():
    values=frozen_sources()
    assert len(values)==10 and all(len(pin)==64 for pin in values.values())
    assert any(p.endswith('t12_baseline_live.py') for p in values)
    assert any(p.endswith('T12_HISTORICAL_SEVEN_MODEL_IDENTITY_20261004.json') for p in values)
