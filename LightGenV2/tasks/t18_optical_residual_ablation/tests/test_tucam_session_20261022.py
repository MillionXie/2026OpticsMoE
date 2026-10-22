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

def test_continuation_identity_and_bounded_original_head():
    code=(TASK/'lab_caltech20_tucam_L6_20261022.py').read_text();ast.parse(code)
    assert 'reuse_complete_g2(' in code and "for stage in ('4','5')" in code
    assert 'hardware_manifest=str(a.hardware)' in code and 'heterogeneous_cameras_explicit=True' in code
    assert "{'linear.weight','linear.bias'}" in code and 'planned_frames=4022' in code
    assert 'camera.reset' not in code and '.cuda(' not in code and 'sensor_full_scale' in code


def test_same_phase_TRAIN_exposure_does_not_refit_geometry_or_use_TEST():
    code=(TASK/'lab_caltech20_tucam_exposure_20261023.py').read_text();ast.parse(code)
    assert 'TRAIN_each_class=2,TEST_used=False' in code and 'range(40)' in code
    assert "rows[0]['canonical_SHA256']!=receipt['canonical_SHA256']" in code
    assert "m.sha(a.out/'phase.bmp')==receipt['phase_SHA256']" in code
    assert 'native_background_corrected_exposure_ratios' in code
    assert 'field_inverse_unchanged=True,optical_LUTs_unchanged=True' in code
    assert '.backward(' not in code and 'automatic_retry=False' in code
    resume=(TASK/'lab_caltech20_tucam_L6_20261022.py').read_text()
    assert "proof['SHA256']==hw['same_phase_TRAIN_exposure_parent_SHA256']" in resume
    assert resume.count('model.measured={}')>=2
