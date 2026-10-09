import json
from pathlib import Path

def test_electronic_followup_matches_optical_round6_training_budget():
    configs=Path(__file__).resolve().parents[1]/'configs'
    optical=json.loads((configs/'regularization_round6.json').read_text())['candidates']
    electronic=json.loads((configs/'electronic_matched_round6.json').read_text())['candidates']
    assert len(optical)==len(electronic)==3
    for a,b in zip(optical,electronic):
        assert b['architecture']=='electronic'
        assert b['init_run']=='crc9_moe_regularized_r2_s17_20261009_electronic'
        for key,value in a.items():
            if key not in ('name','architecture','init_run'):assert b[key]==value
    assert sum(c['epochs'] for c in electronic)==90
