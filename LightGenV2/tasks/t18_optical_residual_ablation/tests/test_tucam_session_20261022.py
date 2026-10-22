"""Static no-hardware contract checks; dynamic logical mapping runs on lab CPU."""
import ast,json
from pathlib import Path
TASK=Path(__file__).resolve().parents[1]

def test_session_has_no_model_fitting_or_TEST_choice():
    code=(TASK/'lab_caltech20_tucam_session_20261022.py').read_text();tree=ast.parse(code)
    assert {'probe','calibrate','logical','camera_for','selftest'}<={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert 'optimizer' not in code and '.backward(' not in code and 'model_for(' not in code
    assert 'raw.astype(np.float32)' in code and 'value*255/full_scale' in code

def test_admission_contract_is_explicit():
    p=json.loads((TASK/'lab_caltech20_TUCam_session_profile_20261022.json').read_text())
    assert p['calibration_TEST_used'] is False and p['camera_auto_exposure'] is False
    assert p['automatic_retry'] is False and p['phase_LUT_changed'] is False
    assert p['sensor_expected_shape']==[2048,2048] and p['logical_shape']==[478,478]
    assert p['exposure_probe_us']==sorted(set(p['exposure_probe_us'])) and max(p['exposure_probe_us'])==80000
