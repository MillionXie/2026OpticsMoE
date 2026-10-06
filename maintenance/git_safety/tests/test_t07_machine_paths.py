import json
from maintenance.git_safety.check_t07_machine_paths import inspect


def fixture(tmp_path):
    config = tmp_path / 'machine.json'
    config.write_text(json.dumps({'camera': {'sdk_root': str(tmp_path / 'camera'), 'exposure_us': 150},
        'amplitude_slm': {'sdk_path': 'vendor', 'binary_folder': 'binary'}, 'settle_delay_ms': 200}))
    for name in ('vendor/holoeye/slmdisplaysdk/__init__.py', 'binary/holoeye_slmdisplaysdk.dll',
                 'camera/demo/base_dll/bin/CEasyCapS.dll', 'camera/cti/x86_64/cxplink_gentl.cti',
                 'phase/Blink_C_wrapper.dll', 'phase/linear.lut'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'fixture-not-a-real-sdk')
    for name in ('vendor', 'binary'):
        (tmp_path / name).mkdir(exist_ok=True)
    return config, tmp_path / 'phase', tmp_path / 'phase/linear.lut'


def test_configured_amplitude_paths_follow_config_not_current_directory(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    result = inspect(*args)
    assert result['passed']
    assert result['paths']['amplitude_sdk']['path'] == str((tmp_path / 'vendor').resolve())
    assert not result['devices_opened']


def test_relative_camera_follows_real_camera_cwd_not_config_parent(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    config = json.loads(args[0].read_text())
    config['camera']['sdk_root'] = 'camera'
    args[0].write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path.parent)
    result = inspect(*args)
    assert not result['passed']
    assert result['paths']['camera_wrapper']['path'] == str(
        (tmp_path.parent / 'camera/demo/base_dll/bin/CEasyCapS.dll').resolve())
    monkeypatch.chdir(tmp_path)
    assert inspect(*args)['passed']


def test_relative_explicit_amplitude_override_follows_bench_cwd(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    elsewhere = tmp_path / 'working'
    wrapper = elsewhere / 'vendor/slmdisplaysdk.py'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes(b'explicit-cwd-wrapper')
    monkeypatch.chdir(elsewhere)
    result = inspect(*args, amplitude_sdk='vendor')
    assert result['passed']
    assert result['paths']['amplitude_sdk']['path'] == str(wrapper.parent.resolve())


def test_missing_relative_vendor_and_explicit_override(tmp_path):
    args = fixture(tmp_path)
    original_wrapper = tmp_path / 'vendor/holoeye/slmdisplaysdk/__init__.py'
    original_wrapper.unlink()
    assert not inspect(*args)['passed']
    alternate = tmp_path / 'actual-vendor'
    alternate.mkdir()
    (alternate / 'slmdisplaysdk.py').write_bytes(b'fixture-not-a-real-sdk')
    original = args[0].read_bytes()
    assert inspect(*args, amplitude_sdk=alternate)['passed']
    assert args[0].read_bytes() == original


def test_empty_vendor_or_missing_native_runtime_fails(tmp_path):
    args = fixture(tmp_path)
    (tmp_path / 'vendor/holoeye/slmdisplaysdk/__init__.py').unlink()
    result = inspect(*args)
    assert not result['passed']
    assert result['paths']['amplitude_sdk']['present']
    assert not result['paths']['amplitude_python_wrapper']['present']
    (tmp_path / 'vendor/slmdisplaysdk.py').write_bytes(b'fixture')
    (tmp_path / 'binary/holoeye_slmdisplaysdk.dll').unlink()
    assert not inspect(*args)['passed']


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
