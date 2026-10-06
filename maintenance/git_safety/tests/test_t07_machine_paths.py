import json
from maintenance.git_safety.check_t07_machine_paths import inspect


def fixture(tmp_path):
    config = tmp_path / 'machine.json'
    config.write_text(json.dumps({'camera': {'sdk_root': 'camera', 'exposure_us': 150},
        'amplitude_slm': {'sdk_path': 'vendor', 'binary_folder': 'binary'}, 'settle_delay_ms': 200}))
    for name in ('camera/demo/base_dll/bin/CEasyCapS.dll', 'camera/cti/x86_64/cxplink_gentl.cti',
                 'phase/Blink_C_wrapper.dll', 'phase/linear.lut'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'fixture-not-a-real-sdk')
    for name in ('vendor', 'binary'):
        (tmp_path / name).mkdir()
    return config, tmp_path / 'phase', tmp_path / 'phase/linear.lut'


def test_paths_follow_config_not_current_directory(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    result = inspect(*args)
    assert result['passed']
    assert result['paths']['amplitude_sdk']['path'] == str((tmp_path / 'vendor').resolve())
    assert not result['devices_opened']


def test_missing_relative_vendor_and_explicit_override(tmp_path):
    args = fixture(tmp_path)
    (tmp_path / 'vendor').rmdir()
    assert not inspect(*args)['passed']
    alternate = tmp_path / 'actual-vendor'
    alternate.mkdir()
    original = args[0].read_bytes()
    assert inspect(*args, amplitude_sdk=alternate)['passed']
    assert args[0].read_bytes() == original


def test_missing_camera_transport_fails(tmp_path):
    args = fixture(tmp_path)
    (tmp_path / 'camera/cti/x86_64/cxplink_gentl.cti').unlink()
    assert not inspect(*args)['passed']


def test_current_hash_and_capture_overrides_do_not_change_config(tmp_path):
    args = fixture(tmp_path)
    original = args[0].read_bytes()
    result = inspect(*args)
    assert len(result['paths']['phase_lut']['sha256']) == 64
    assert result['original_config_settings']['exposure_us'] == 150
    assert result['effective_capture_settings']['exposure_us'] == 400
    assert args[0].read_bytes() == original
