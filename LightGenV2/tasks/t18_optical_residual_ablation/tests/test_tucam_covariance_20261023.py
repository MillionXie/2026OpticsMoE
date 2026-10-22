"""Static scope/identity tests; numeric fold parity is run on laboratory CPU."""
import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]


def test_followup_is_one_finite_original_parameter_trajectory():
    code=(HERE/'lab_caltech20_tucam_covariance_20261023.py').read_text();ast.parse(code)
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())
    assert cfg['continuation_epochs']==24 and cfg['continuation_seconds']==600
    assert .4001<cfg['TRAIN_rule_alpha_floor']<=.95
    assert "geometry['electronic']['ratio']>geometry['fused']['ratio']" in code
    assert 'TEST_used_to_fit=False' in code and 'new_inference_layers=0' in code
    assert 'camera.' not in code and 'torch.nn.Linear' in code.split('def selftest():')[1]
    assert "parent=dependency/'best.pt'" in code and 'time.monotonic()-start<7200' in code


def test_full_data_and_prefix_identity_precede_fit():
    code=(HERE/'lab_caltech20_tucam_covariance_20261023.py').read_text()
    assert "identities==m.read(dependency/'protocol.json')['prefix_SHA256']" in code
    assert "dm['sha256']==cfg['data_SHA256']" in code
    assert "m.sha(diagnostic)==follow['diagnostic_SHA256']" in code
    assert 'loop.validate_state(m,model,m.state_cpu(captured),allowed(model))' in code
    assert 'chosen,result=finish(' in code and 'selected_E0=all(torch.equal' in code
