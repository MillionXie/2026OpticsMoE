"""CPU-only DVP capture tests; all device objects are fake, no SDK is loaded."""
import ctypes as C
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from LightGenV2.hardware_common.dvp_legacy import Camera, Frame


@pytest.fixture
def bench_module(monkeypatch):
    # Use a placeholder phase class so these tests work before machine-only
    # phase dependencies exist. It is replaced by a fully fake object below.
    monkeypatch.setitem(sys.modules, 'LightGenV2.hardware_common.shs.phase_hdmi',
                        SimpleNamespace(PhaseHDMI=object))
    module = importlib.import_module('LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm.bench')
    monkeypatch.setattr(module, '_configured', False)
    return module


def config_fixture(root):
    values = {'camera_dll': 'camera.dll', 'phase_lut': 'phase.lut',
              'phase_sdk': 'phase', 'amplitude_sdk': 'amp', 'amplitude_bin': 'bin'}
    for key, name in values.items():
        if key in ('camera_dll', 'phase_lut'):
            (root / name).write_bytes(b'synthetic path only, never loaded')
        else:
            (root / name).mkdir()
    path = root / 'machine.json'
    path.write_text(json.dumps(values))
    return path


def test_machine_paths_explicit_and_relative(bench_module, tmp_path):
    values = bench_module.configure(config_fixture(tmp_path))
    assert values['camera_dll'] == str((tmp_path / 'camera.dll').resolve())
    assert bench_module._configured


def test_missing_machine_config_does_not_construct_device(bench_module, tmp_path, monkeypatch):
    monkeypatch.setattr(bench_module, 'Camera', lambda *a: pytest.fail('device construction'))
    with pytest.raises(RuntimeError, match='explicit machine'):
        bench_module.Bench(tmp_path, 10000, 240, {})
    config = tmp_path / 'machine.json'
    config.write_text('{}')
    with pytest.raises(ValueError, match='Missing explicit'):
        bench_module.configure(config)


def test_orientation_no_photometric_change(bench_module):
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for name in ('flip_v', 'identity', 'rot270'):
        actual = bench_module.orient(image, name)
        assert sorted(actual.ravel()) == sorted(image.ravel())
        assert not np.shares_memory(image, actual)


@pytest.mark.parametrize('side', [224, 478])
def test_native_input_matches_17_to_8_sampling(bench_module, tmp_path, side):
    from LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm import capture_full_stage
    gray = np.arange(side * side).astype(np.uint8).reshape(side, side)
    source, output = tmp_path / 'compact.bmp', tmp_path / 'native.bmp'
    Image.fromarray(gray).save(source)
    capture_full_stage.native_bmp(source, output)
    actual = np.asarray(Image.open(output))
    size = round(side * 17 / 8)
    positions = (np.arange(size) + .5 - size / 2) * 8
    index = np.floor(positions / 17 + side / 2).astype(int).clip(0, side - 1)
    expected = np.zeros((1080, 1920), np.uint8)
    top, left = (1080 - size) // 2, (1920 - size) // 2
    expected[top:top+size, left:left+size] = gray[np.ix_(index, index)]
    np.testing.assert_array_equal(actual, expected)
    assert capture_full_stage.ORIENTATIONS == ('flip_v', 'identity', 'identity', 'flip_v', 'rot270', 'rot270')
    assert capture_full_stage.CORNER_POINTS == [[995, 172], [4310, 172], [4303, 3440], [980, 3435]]


def test_help_does_not_configure_or_open_hardware(bench_module, monkeypatch, capsys):
    from LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm import capture_full_stage
    monkeypatch.setattr(sys, 'argv', ['capture_full_stage', '--help'])
    monkeypatch.setattr(bench_module, 'configure', lambda *args: pytest.fail('configuration on help'))
    with pytest.raises(SystemExit) as result:
        capture_full_stage.main()
    assert result.value.code == 0
    text = capsys.readouterr().out
    assert '--project' in text and '--machine-config' in text and '--stage' in text


def test_original_indexed_test_router_manifest(bench_module, tmp_path):
    from LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm import capture_full_stage
    records = [{'index': i, 'sample_id': f'synthetic-{i}', 'file': f'image_{i:04d}.png',
                'sha256': 'original-amplitude-hash', 'checkpoint_sha256': capture_full_stage.EXPECTED}
               for i in reversed(range(2400))]
    path = tmp_path / 'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8-sig')
    actual = capture_full_stage.read_capture_manifest(path, 'vision_router', 'full_test')
    assert list(actual) == [f'image_{i:04d}' for i in range(2400)]
    assert actual['image_0000']['sample_id'] == 'synthetic-0'
    assert actual['image_0000']['sha256'] == 'original-amplitude-hash'
    assert 'key' not in records[0]


@pytest.mark.parametrize('problem', ['duplicate_index', 'filename', 'other_stage', 'train', 'mixed'])
def test_reject_invalid_indexed_capture_manifest(bench_module, tmp_path, problem):
    from LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm import capture_full_stage
    records = [{'index': i, 'file': f'image_{i:04d}.png'} for i in range(2400)]
    stage, run = 'vision_router', 'full_test'
    if problem == 'duplicate_index': records[-1]['index'] = 0
    if problem == 'filename': records[0]['file'] = 'wrong.png'
    if problem == 'other_stage': stage = 'vision_expert'
    if problem == 'train': run = 'finetune_train800'
    if problem == 'mixed': records[0]['key'] = 'image_0000'
    path = tmp_path / 'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in records))
    with pytest.raises(ValueError):
        capture_full_stage.read_capture_manifest(path, stage, run)


def test_current_manifest_keys_preserved_and_duplicates_rejected(bench_module, tmp_path):
    from LightGenV2.tasks.t08_abo_image_text_retrieval.physical10cm import capture_full_stage
    records = [{'key': 'train_0001', 'file': 'train_0001.png', 'sha256': 'original'}]
    path = tmp_path / 'manifest.jsonl'
    path.write_text(json.dumps(records[0]))
    assert capture_full_stage.read_capture_manifest(path, 'vision_router', 'finetune_train800') == {'train_0001': records[0]}
    path.write_text('\n'.join(json.dumps(row) for row in records * 2))
    with pytest.raises(ValueError, match='Duplicate'):
        capture_full_stage.read_capture_manifest(path, 'vision_router', 'finetune_train800')


def test_bench_six_frames_order_and_gain(bench_module, tmp_path, monkeypatch):
    events = []
    class Device:
        def __init__(self, label): self.label = label
        def __enter__(self): events.append('open-' + self.label); return self
        def __exit__(self, *args): events.append('close-' + self.label)
    class Amp(Device):
        def preload_files(self, paths): events.append('preload')
        def display_file(self, path): events.append('display')
    class Phase(Device):
        def show(self, path): events.append('phase-show'); return {'synthetic': True}
    class FakeCamera(Device):
        def __init__(self): super().__init__('camera'); self.number = 0
        def settings(self, **kwargs): events.append(kwargs); return kwargs
        def capture(self):
            self.number += 1
            return np.full((478, 478), self.number, np.uint8), {'frame_id': self.number}
    monkeypatch.setattr(bench_module, 'Camera', lambda *args: FakeCamera())
    monkeypatch.setattr(bench_module, 'HoloeyeSLM', lambda *args: Amp('amp'))
    monkeypatch.setattr(bench_module, 'PhaseHDMI', lambda *args, **kwargs: Phase('phase'))
    monkeypatch.setattr(bench_module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(bench_module, 'warp', lambda image: image)
    bench_module.configure(config_fixture(tmp_path))
    amplitude, phase = tmp_path / 'amp.bmp', tmp_path / 'phase.bmp'
    amplitude.write_bytes(b'amp'); phase.write_bytes(b'phase')
    with bench_module.Bench(tmp_path, 10000, 240, {}) as bench:
        images, _ = bench.capture('vision_router', phase, [amplitude], ['sample'], 'flip_v', save=False)
    assert np.all(images == 6)
    assert bench.rows[0]['frame_ids'] == [1, 2, 3, 4, 5, 6]
    assert bench.rows[0]['no_photometric_normalization']
    assert {'exposure': 10000, 'gain': 1.0} in events
    assert events[:3] == ['open-phase', 'open-amp', 'open-camera']
    assert events[-3:] == ['close-camera', 'close-amp', 'close-phase']


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_native_camera_returns_owned_raw_buffer(dtype):
    source = np.arange(6, dtype=dtype).reshape(2, 3)
    buffer = C.create_string_buffer(source.tobytes())
    camera = Camera.__new__(Camera)
    camera.handle = C.c_uint32(1)
    def call(name, handle, frame, pointer, timeout):
        assert name == 'GetFrame'
        value = frame._obj
        value.format = 0; value.bits = 0; value.width = 3; value.height = 2
        value.bytes = source.nbytes; value.id = 42
        pointer._obj.value = C.addressof(buffer)
    camera.call = call
    actual, metadata = camera.capture()
    np.testing.assert_array_equal(actual, source)
    buffer[0] = b'\xff'
    assert actual.flat[0] == 0 and metadata['frame_id'] == 42


@pytest.mark.parametrize('fmt,byte_count', [(1, 6), (0, 7)])
def test_invalid_native_layout_rejected(fmt, byte_count):
    camera = Camera.__new__(Camera)
    camera.handle = C.c_uint32(1)
    def call(name, handle, frame, pointer, timeout):
        value = frame._obj
        value.format = fmt; value.bits = 0; value.width = 3; value.height = 2; value.bytes = byte_count
    camera.call = call
    with pytest.raises(RuntimeError, match='Unexpected DVP|Unsupported packed'):
        camera.capture()
