"""Pure SHS contract checks. No camera, SLM, DLL or acquisition is opened."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from LightGenV2.hardware_common.shs import capture, phase_hdmi, sdk, slm_camera
from experiments.hardware_sdk.devices import HoloeyeSLM


def test_mono8_keeps_offset_padding_and_zero():
    image, bits = sdk.decode_mono(bytes([55, 0, 2, 99, 3, 4, 88]), 2, 2,
                                  0x01080001, xpadding=1, image_offset=1)
    assert bits == 8 and image.tolist() == [[0, 2], [3, 4]]


def test_unpacked_mono10_is_not_rescaled():
    raw = np.array([0, 1023, 3, 12], dtype='<u2').tobytes()
    image, bits = sdk.decode_mono(raw, 2, 2, 0x01100003)
    assert bits == 10 and image.tolist() == [[0, 1023], [3, 12]]


@pytest.mark.parametrize('raw,width,height,fmt', [
    (b'', 2, 2, 0x01080001), (b'1234', 2, 2, 0x010C0004),
    (np.array([1024], dtype='<u2').tobytes(), 1, 1, 0x01100003),
    (b'1234', 0, 2, 0x01080001),
])
def test_invalid_camera_buffers_are_rejected(raw, width, height, fmt):
    with pytest.raises(sdk.SDKError):
        sdk.decode_mono(raw, width, height, fmt)


def test_fresh_drains_sdk_buffers_before_returning_one():
    class FakeCamera:
        config = {'buffer_count': 4}
        frames = 0
        def grab(self):
            self.frames += 1
            return self.frames
    camera = FakeCamera()
    assert sdk.Camera.fresh(camera) == 7 and camera.frames == 7


def test_restore_exposure_before_returning_to_high_frame_rate():
    class FakeCamera:
        def __init__(self):
            self.values = {'Gain': 'Gain_X4', 'ExposureTime': '400', 'AcquisitionFrameRate': '2000'}
            self.operations = []
        def stop(self): self.operations.append(('stop', None))
        def get(self, name): return self.values[name]
        def set(self, name, value):
            self.operations.append((name, str(value)))
            self.values[name] = str(value)
    camera = FakeCamera()
    before = {k: {'value': v} for k, v in {
        'Gain': 'Gain_X1', 'ExposureTime': '444.2', 'AcquisitionFrameRate': '2250'}.items()}
    assert capture.restore_settings(camera, before, list(before)) == []
    assert all(camera.get(k) == v['value'] for k, v in before.items())
    assert camera.operations.index(('AcquisitionFrameRate', '100')) < camera.operations.index(('ExposureTime', '444.2'))


def test_slm_driver_resolves_machine_paths_against_explicit_config_base(tmp_path):
    config = {'amplitude_slm': {'connected': True, 'driver': 'holoeye',
                               'sdk_path': 'vendor/holoeye_python', 'sdk_api_version': 3}}
    driver = slm_camera.slm_driver(config, tmp_path)
    assert isinstance(driver, HoloeyeSLM)
    assert driver.sdk_path == (tmp_path / 'vendor/holoeye_python').resolve()
    assert driver.sdk_api_version == 3 and driver._slm is None


def test_disconnected_slm_is_not_implicitly_enabled(tmp_path):
    with pytest.raises(RuntimeError, match='not connected'):
        slm_camera.slm_driver({'amplitude_slm': {'connected': False}}, tmp_path)


def test_controller_requires_explicit_machine_config_base():
    with pytest.raises(TypeError):
        slm_camera.Controller({})


def test_hidden_phase_launch_is_rejected():
    with pytest.raises(RuntimeError, match='SW_HIDE'):
        phase_hdmi.validate_launch_visibility(1, 0)
    assert phase_hdmi.validate_launch_visibility(0, 0)['dwFlags'] == 0


def test_phase_native_bytes_and_sha_are_not_resized_or_inverted(tmp_path):
    p = tmp_path / 'phase.bmp'
    image = np.zeros((1200, 1920), dtype=np.uint8)
    image[0, 0], image[-1, -1] = 1, 254
    Image.fromarray(image).save(p)
    assert np.array_equal(phase_hdmi.load_native(p, phase_hdmi.sha(p)), image)
    with pytest.raises(ValueError, match='SHA'):
        phase_hdmi.load_native(p, '0' * 64)


def test_phase_sdk_constant_zero_ack_exception_is_hash_pinned():
    phase = phase_hdmi.PhaseHDMI('unused', 'unused')
    phase.pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    phase.dll = SimpleNamespace(Write_image=lambda pointer, bits: 0)
    phase.wrapper_sha = '0d3cc283165bb62ed60a4c8b1c1a256af9441e6342654511fd1f80dbe46ce225'
    assert phase.repeat() == (0, True)
    phase.wrapper_sha = '0' * 64
    with pytest.raises(RuntimeError, match='Write_image failed'):
        phase.repeat()


def test_invalid_phase_format_fails_without_loading_sdk():
    with pytest.raises(ValueError, match='format'):
        phase_hdmi.PhaseHDMI('unused', 'unused', pixel_format='invalid')
