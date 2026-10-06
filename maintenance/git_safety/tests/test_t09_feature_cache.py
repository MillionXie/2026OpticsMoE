import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from maintenance.git_safety import check_t09_feature_cache as module
from LightGenV2.tasks.t09_multimodal_matching import vision


def fixture(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    checkpoint = tmp_path / 'frontend.pt'
    torch.manual_seed(19)
    model = vision.VisionEncoder(8, widths=(2, 3, 4)).eval()
    torch.save({'model': model.state_dict(), 'architecture': {'widths': [2, 3, 4]}}, checkpoint)
    values = {}
    for split in ('train', 'val'):
        images = np.random.default_rng(17).integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
        np.savez(data / (split + '_images.npz'), images=images)
        values[split] = module.reconstruct(checkpoint, images)
    cache = tmp_path / 'feature_cache.npz'
    np.savez(cache, **values)
    manifest = {'files': {s + '_images.npz': module.sha(data / (s + '_images.npz'))
                          for s in ('train', 'val')}}
    (data / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    meta = dict(checkpoint_sha256=module.sha(checkpoint),
                data_manifest_sha256=module.sha(data / 'manifest.json'), cache_sha256=module.sha(cache),
                features={s + '_feature_sha256': module.hashlib.sha256(values[s].tobytes()).hexdigest()
                          for s in ('train', 'val')})
    cache.with_suffix('.json').write_text(json.dumps(meta), encoding='utf8')
    return data, checkpoint, cache, Path(vision.__file__)


def test_cpu_exact_roundtrip_preserves_every_fixture_file(tmp_path):
    args = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = module.check(*args)
    assert result['passed'] and result['original_assets_unchanged']
    assert all(r['exact_bytes'] for r in result['splits'].values())
    assert result['device'] == 'cpu' and not result['test_accessed']
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_checkpoint_gate_precedes_model_execution(tmp_path):
    args = fixture(tmp_path)
    args[1].write_bytes(b'untrusted changed checkpoint')
    with patch.object(module, 'reconstruct', side_effect=AssertionError('must not execute')):
        result = module.check(*args)
    assert not result['passed'] and not result['splits']
    assert any('Bound SHA mismatch' in e for e in result['errors'])


def test_numeric_disagreement_is_not_hidden_or_recalibrated(tmp_path):
    args = fixture(tmp_path)
    with patch.object(module, 'reconstruct', return_value=np.full((3, 128), 10, dtype=np.float32)):
        result = module.check(*args)
    assert not result['passed']
    assert all(r['outside_tolerance'] > 0 for r in result['splits'].values())
    assert result['rtol'] == 1e-5 and result['atol'] == 1e-5


def test_source_difference_stops_execution(tmp_path):
    args = fixture(tmp_path)
    alternate = tmp_path / 'different_source.py'
    alternate.write_bytes(b'# different historical computation\n')
    with patch.object(module, 'reconstruct', side_effect=AssertionError('must not execute')):
        result = module.check(*args[:3], alternate)
    assert not result['passed'] and not result['frontend_source_equal_lf']
