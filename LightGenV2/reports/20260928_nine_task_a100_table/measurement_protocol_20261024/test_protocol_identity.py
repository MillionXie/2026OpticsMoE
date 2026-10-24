"""CPU-only row identity correction; no model/data/CUDA is imported."""
import hashlib
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from protocol_identity import effective_protocol


def test_changes_only_selected_identity_not_existing_timing_contract():
    path=HERE/'protocol.json';raw=path.read_bytes();base=json.loads(raw)
    result=effective_protocol()
    assert path.read_bytes()==raw
    assert result['protocol_id']==base['protocol_id']
    assert {k:v for k,v in result.items() if k not in ('rows','identity_amendments')} == {
        k:v for k,v in base.items() if k!='rows'}
    assert [r for r in result['rows'] if r['id']!='07']==[r for r in base['rows'] if r['id']!='07']
    row=next(r for r in result['rows'] if r['id']=='07')
    assert row['ours_target']==.8765 and row['ours_lab_target']==.8655
    assert row['qwen_target']==.812 and row['ours_epoch']==45
    assert result['identity_amendments'][0]['no_new_timing_result']


def test_rejects_unbound_or_mutated_base(tmp_path):
    path=tmp_path/'base.json';path.write_bytes((HERE/'protocol.json').read_bytes()+b' ')
    with pytest.raises(ValueError):effective_protocol(path)


@pytest.mark.parametrize('key,value', [('ours_target',.888),('ours_target',.8655),
    ('ours_epoch',70),('layout_version','grid_v2'),('qwen_pt_sha256','0'*64)])
def test_rejects_other_weight_layout_or_simulation_value(tmp_path,key,value):
    a=json.loads((HERE/'protocol_amendment_openmoji_e45.json').read_bytes())
    a['effective_row'][key]=value
    path=tmp_path/'amendment.json';path.write_text(json.dumps(a),encoding='utf8')
    with pytest.raises(ValueError):effective_protocol(amendment_path=path)


def test_keeps_original_old_grid_identity_as_audit_history():
    a=json.loads((HERE/'protocol_amendment_openmoji_e45.json').read_bytes())
    base=json.loads((HERE/'protocol.json').read_bytes())
    original=next(r for r in base['rows'] if r['id']=='07')
    assert original['ours_pt_sha256']==a['superseded_identity']['ours_pt_sha256']
    assert original['ours_target']==.888
    assert hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()==a['base_protocol_sha256']
