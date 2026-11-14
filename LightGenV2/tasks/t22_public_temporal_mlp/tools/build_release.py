"""Package the normalized-RGB inference project with private verification."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch

if __package__:
    from . import _paths
    from .checkpoint_utils import file_sha256, inference_state_name, tensor_fingerprint
else:
    import _paths
    from checkpoint_utils import file_sha256, inference_state_name, tensor_fingerprint


ROOT = _paths.ROOT
PUBLIC = ROOT.parents[2] / 'LightGen2'
SOURCE_FILES = (
    '.gitignore', '.gitattributes', 'model.py', 'infer.py', 'visualize.py',
    'README.md', 'requirements.txt', 'LICENSE', 'NOTICE', 'licenses/QWEN_LICENSE.txt',
)
RESULT_FILES = (
    '01_predictions.csv', '02_metrics.json', '03_prediction_vs_mos.png',
    '04_input_frames.png', '05_optical_expert_share.npz', '05_optical_expert_share.png',
) + tuple(f'{number:02d}_{branch}_{kind}.{suffix}'
          for branch, first in (('vision', 6), ('language', 9))
          for number, kind in enumerate(('mask', 'input_field', 'output_field'), first)
          for suffix in ('npz', 'png'))
ARTIFACT_FILES = ('weights/LightGen2.pt', 'README.docx', 'data/test_rgb.npz') + tuple(
    'results/' + name for name in RESULT_FILES)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8', newline='\n')


def import_public_module(name):
    sys.path.insert(0, str(PUBLIC.parent))
    try:
        return importlib.import_module('LightGen2.' + name)
    finally:
        sys.path.pop(0)


def verify_optical_arrays(directory, predictions, *, shared_phase=False):
    reports = {}
    metrics = json.loads((directory / '02_metrics.json').read_text('utf-8'))
    start = (metrics['visualized_field'] - 1) * 16
    captured = predictions[start:start + 16]
    real_count = len(captured)
    captured = captured + predictions[:16 - real_count]
    expected_valid = (np.arange(16) < real_count).tolist()
    for branch, first in (('vision', 6), ('language', 9)):
        with np.load(directory / f'{first:02d}_{branch}_mask.npz', allow_pickle=False) as mask, \
                np.load(directory / f'{first+1:02d}_{branch}_input_field.npz', allow_pickle=False) as source, \
                np.load(directory / f'{first+2:02d}_{branch}_output_field.npz', allow_pickle=False) as target:
            for archive in (mask, source, target):
                if archive['video_names'].tolist() != [row['sample_id'] for row in captured]:
                    raise ValueError('Optical array video order differs')
                if archive['valid'].tolist() != expected_valid:
                    raise ValueError('Optical array padding flags differ')
            phase = np.zeros((518, 518), dtype=np.float32)
            if mask['phase_tiles'].shape != (16, 111, 111):
                raise ValueError('Full phase tiles are missing')
            if shared_phase and not np.array_equal(
                    mask['phase_tiles'], np.broadcast_to(mask['phase_tiles'][:1], (16, 111, 111))):
                raise ValueError('Video slots have different learned masks in ' + branch)
            for tile, (row, column) in zip(mask['phase_tiles'], mask['slot_origins']):
                phase[row+2:row+113, column+2:column+113] = tile
            np.testing.assert_array_equal(phase, mask['phase'])
            modulated = source['field'] * mask['modulation']
            np.testing.assert_allclose(modulated, source['modulated_field'], rtol=1e-6, atol=1e-7)
            replayed = np.fft.ifft2(np.fft.fft2(modulated) * target['transfer'])
            error = float(np.linalg.norm(replayed - target['field']) / np.linalg.norm(target['field']))
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError('Optical propagation replay differs in ' + branch)
            np.testing.assert_allclose(np.abs(target['field'])**2, target['intensity'], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(source['display_amplitude'], np.clip(source['amplitude'] / source['display_scale'], 0, 1), rtol=1e-6)
            np.testing.assert_allclose(target['display_intensity'], np.clip(target['intensity'] / target['display_scale'], 0, 1), rtol=1e-6)
            np.testing.assert_allclose(target['display_scale'], np.percentile(target['intensity'], 99.8), rtol=1e-6)
            reports[branch] = {'phase_tiles': 16, 'numpy_propagation_relative_error': error}
    with np.load(directory / '05_optical_expert_share.npz', allow_pickle=False) as routing:
        if routing['sample_ids'].tolist() != [row['sample_id'] for row in predictions]:
            raise ValueError('Full routing identities differ')
        for branch in ('vision', 'language'):
            selected = routing[branch + '_selected']
            if selected.shape[0] != 558 or not np.all(selected.sum(-1) == 2):
                raise ValueError('Expected exactly two selections for each actual item')
            counts = selected.reshape(-1, 4).sum(0)
            np.testing.assert_array_equal(counts, routing[branch + '_selection_count'])
            np.testing.assert_allclose(100 * counts / counts.sum(), routing[branch + '_share_percent'])
    return reports


def build_release():
    profile = json.loads((ROOT / 'configs/selected.json').read_text('utf-8-sig'))
    source_path = ROOT / profile['checkpoint']
    if file_sha256(source_path) != profile['checkpoint_sha256']:
        raise ValueError('Research checkpoint differs from the selected profile')
    original = torch.load(source_path, map_location='cpu', weights_only=True)
    public = torch.load(PUBLIC / 'weights/LightGen2.pt', map_location='cpu', weights_only=True)
    omitted = {'target_mean', 'target_std', 'level_boundaries'}
    expected = {inference_state_name(name): tensor for name, tensor in original['model'].items()
                if name not in omitted}
    if expected.keys() != public['model'].keys():
        raise ValueError('Public state names differ')
    for name, tensor in expected.items():
        if not torch.equal(tensor, public['model'][name]):
            raise ValueError('Public model tensor differs: ' + name)
    public_model = import_public_module('model')
    source_config = dict(original['config'])
    target_config = dict(public['config'])
    source_config.setdefault('shared_global_phase', False)
    target_config.setdefault('shared_global_phase', False)
    expected_architecture = (public_model.ARCHITECTURE if target_config['shared_global_phase']
                             else public_model.LEGACY_ARCHITECTURE)
    if public['schema'] != 2 or public['architecture'] != expected_architecture:
        raise ValueError('Public architecture conflicts with global phase sharing')
    for field in fields(public_model.ModelConfig):
        if target_config[field.name] != source_config[field.name]:
            raise ValueError('Public configuration differs: ' + field.name)

    manifest = ROOT / 'assets/manifest.csv'
    if file_sha256(manifest) != original['metadata']['dataset']['manifest']['sha256']:
        raise ValueError('Dataset manifest differs')
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == 'test']
    public_infer = import_public_module('infer')
    with np.load(PUBLIC / 'data/test_rgb.npz', allow_pickle=False) as inputs:
        sample_ids, targets = public_infer.read_input_metadata(inputs)
        if len(sample_ids) != 558 or sample_ids != [row['sample_id'] for row in rows]:
            raise ValueError('Normalized RGB identities differ from the selected 558-video order')
        np.testing.assert_array_equal(targets, [float(row['temporal']) for row in rows])
        if inputs['rgb_0035'].shape != (14, 4, 3, 448, 448):
            raise ValueError('Tail normalized RGB geometry differs')
    result_dir = PUBLIC / 'results'
    with (result_dir / '01_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    if [row['sample_id'] for row in predictions] != sample_ids:
        raise ValueError('Published predictions differ from evaluation identities')
    np.testing.assert_array_equal([float(row['target_MOS']) for row in predictions], targets)
    from scipy.stats import spearmanr
    score = float(spearmanr([float(row['predicted_MOS']) for row in predictions], targets).statistic)
    metrics = json.loads((result_dir / '02_metrics.json').read_text('utf-8'))
    if score != metrics['SRCC'] or metrics['videos'] != 558 or metrics['fields'] != 35:
        raise ValueError('Published summary differs from per-video scores')
    if metrics['total_runtime_seconds'] <= 0:
        raise ValueError('Complete program runtime is missing')
    shared_phase = original['config'].get('shared_global_phase', False)
    if shared_phase:
        for branch in ('frame_global', 'video_global'):
            if original['model'][branch + '.raw_global_phase'].shape != (1, 111, 111):
                raise ValueError('Expected one trainable phase tile in ' + branch)
    replay = verify_optical_arrays(result_dir, predictions, shared_phase=shared_phase)
    names = SOURCE_FILES + ARTIFACT_FILES
    actual = {path.relative_to(PUBLIC).as_posix() for path in PUBLIC.rglob('*') if path.is_file()}
    if actual != set(names):
        raise ValueError('Unexpected or missing public files: ' + repr(actual.symmetric_difference(names)))
    for name in names:
        path = PUBLIC / name
        if path.suffix in ('.py', '.md', '.json', '.txt'):
            text = path.read_text('utf-8-sig')
            if any(marker in text for marker in ('Xml12', 'OneDrive', '/DATA/', '/home/', 'xwechat_files', 'model_parameters')):
                raise ValueError('Private metadata in public file: ' + name)
    checksums = {name: file_sha256(PUBLIC / name) for name in sorted(names)}
    release_root = ROOT / 'releases'
    stage = Path(tempfile.mkdtemp(prefix='lightgen2-release-')).resolve()
    for name, digest in checksums.items():
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PUBLIC / name, destination)
        if file_sha256(destination) != digest:
            raise ValueError('Staged file differs: ' + name)
    archive = release_root / '.LightGen2.zip.building'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for name in names:
            compression = zipfile.ZIP_STORED if name.endswith('.npz') else zipfile.ZIP_DEFLATED
            zipped.write(stage / name, 'LightGen2/' + name, compress_type=compression)
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != {'LightGen2/' + name for name in names}:
            raise ValueError('Unexpected ZIP entries')
        for name, expected_hash in checksums.items():
            digest = hashlib.sha256()
            with zipped.open('LightGen2/' + name) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError('ZIP file differs: ' + name)
    mirror = release_root / 'current'
    if mirror.exists():
        if mirror.resolve().parent != release_root.resolve():
            raise ValueError('Release mirror is outside the expected directory')
        backup = Path(tempfile.mkdtemp(prefix='lightgen2-previous-release-')) / 'current'
        mirror.rename(backup)
    stage.rename(mirror)
    archive.replace(release_root / 'LightGen2.zip')
    digest = file_sha256(release_root / 'LightGen2.zip')
    (release_root / 'LightGen2.zip.sha256').write_text(digest + '  LightGen2.zip\n', encoding='utf-8')
    verification = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'canonical_directory': 'LightGen2', 'files': checksums,
                    'retained_model_tensors_bitwise_equal': True,
                    'shared_global_phase': shared_phase,
                    'inference_state_fingerprint': tensor_fingerprint(public['model']),
                    'metrics': metrics, 'optical_array_replay': replay, 'archive_sha256': digest}
    write_json(ROOT / 'runs/simulation/current/validation/public_inference/release_verification.json', verification)
    print(json.dumps({'files': len(names), 'SRCC': score, 'archive_sha256': digest,
                      'archive': str(release_root / 'LightGen2.zip')}, indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    build_release()
