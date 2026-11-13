"""Package the root LightGen2 inference project and keep verification private."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib
import json
import os
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
    from .checkpoint_utils import export_inference_checkpoint, file_sha256
else:
    import _paths
    from checkpoint_utils import export_inference_checkpoint, file_sha256

if __package__ and __package__ != 'tools':
    from ..model import load_lightgen
else:
    from model import load_lightgen


ROOT = _paths.ROOT
PUBLIC = ROOT.parents[2] / 'LightGen2'
SOURCE_FILES = (
    '.gitignore', '.gitattributes', 'model.py', 'infer.py', 'demo.py', 'visualize.py',
    'README.md', 'requirements.txt', 'LICENSE', 'NOTICE', 'licenses/QWEN_LICENSE.txt',
)
RESULT_FILES = tuple(f'{branch}_{kind}.{suffix}'
                     for branch in ('vision', 'language')
                     for kind in ('mask', 'input_field', 'output_field')
                     for suffix in ('npz', 'png')) + (
    'input_frames.png', 'optical_expert_share.npz', 'optical_expert_share.png', 'predictions.json',
)
ARTIFACT_FILES = ('weights/LightGen2.pt', 'README.docx', 'test_videos.csv') + tuple(
    'results/' + name for name in RESULT_FILES)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8', newline='\n')


def import_public_module(name):
    sys.path.insert(0, str(PUBLIC.parent))
    try:
        return importlib.import_module('LightGen2.' + name)
    finally:
        sys.path.pop(0)


def verify_files(directory, checksums):
    for name, expected in checksums.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != expected:
            raise ValueError('Release file differs: ' + name)


def verify_optical_arrays(directory):
    """Reconstruct the displayed phase tiles, modulation and propagation."""
    reports = {}
    for branch in ('vision', 'language'):
        with np.load(directory / f'{branch}_mask.npz', allow_pickle=False) as mask, \
                np.load(directory / f'{branch}_input_field.npz', allow_pickle=False) as source, \
                np.load(directory / f'{branch}_output_field.npz', allow_pickle=False) as target:
            for archive in (mask, source, target):
                if len(archive['video_names']) != 16 or len(set(archive['video_names'])) != 16:
                    raise ValueError('Optical arrays must identify 16 distinct videos')
                if not np.array_equal(archive['video_names'], mask['video_names']):
                    raise ValueError('Optical array video order differs')
            phase = np.zeros((518, 518), dtype=np.float32)
            if mask['phase_tiles'].shape != (16, 111, 111):
                raise ValueError('Full phase tiles are missing')
            for tile, (row, column) in zip(mask['phase_tiles'], mask['slot_origins']):
                phase[row + 2:row + 113, column + 2:column + 113] = tile
            if not np.array_equal(phase, mask['phase']):
                raise ValueError('Phase canvas differs from its sixteen tiles')
            modulated = source['field'] * mask['modulation']
            np.testing.assert_allclose(modulated, source['modulated_field'], rtol=1e-6, atol=1e-7)
            replayed = np.fft.ifft2(np.fft.fft2(modulated) * target['transfer'])
            relative_error = float(np.linalg.norm(replayed - target['field']) / np.linalg.norm(target['field']))
            if not np.isfinite(relative_error) or relative_error > 1e-5:
                raise ValueError('Optical propagation replay differs in ' + branch)
            np.testing.assert_allclose(np.abs(target['field']) ** 2, target['intensity'], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(source['display_amplitude'], np.clip(source['amplitude'] / source['display_scale'], 0, 1), rtol=1e-6)
            np.testing.assert_allclose(target['display_intensity'], np.clip(target['intensity'] / target['display_scale'], 0, 1), rtol=1e-6)
            np.testing.assert_allclose(target['display_scale'], np.percentile(target['intensity'], 99.8), rtol=1e-6)
            reports[branch] = {'phase_tiles': 16, 'canvas': [518, 518],
                               'numpy_propagation_relative_error': relative_error}
    with np.load(directory / 'optical_expert_share.npz', allow_pickle=False) as routing:
        for branch in ('vision', 'language'):
            selected = routing[branch + '_selected']
            if not np.all(selected.sum(-1) == 2):
                raise ValueError('Expected exactly two expert selections')
            count = selected.reshape(-1, 4).sum(0)
            np.testing.assert_array_equal(count, routing[branch + '_selection_count'])
            np.testing.assert_allclose(100 * count / count.sum(), routing[branch + '_share_percent'])
    return reports


def build_release(device='cpu'):
    profile = json.loads((ROOT / 'configs/selected.json').read_text('utf-8-sig'))
    source = ROOT / profile['checkpoint']
    if file_sha256(source) != profile['checkpoint_sha256']:
        raise ValueError('Research checkpoint differs from the selected profile')
    import csv
    manifest_path = ROOT / 'assets/manifest.csv'
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if file_sha256(manifest_path) != checkpoint['metadata']['dataset']['manifest']['sha256']:
        raise ValueError('Evaluation identities differ from the selected checkpoint')
    with manifest_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == 'test']
    if len(rows) != 558 or len({row['sample_id'] for row in rows}) != 558:
        raise ValueError('Expected the original 558-video evaluation list')
    for row in rows:
        identity = Path(row['sample_id'])
        if identity.is_absolute() or '..' in identity.parts or not row['video_path'].endswith('/' + row['sample_id']):
            raise ValueError('Evaluation video identity differs: ' + row['sample_id'])
    with (PUBLIC / 'test_videos.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('sample_id', 'video_path', 'mos'))
        writer.writeheader()
        writer.writerows({'sample_id': row['sample_id'], 'video_path': row['sample_id'],
                         'mos': row['temporal']} for row in rows)
    public_model = import_public_module('model')
    exported = export_inference_checkpoint(
        source, PUBLIC / 'weights/LightGen2.pt',
        [field.name for field in fields(public_model.ModelConfig)], public_model.ARCHITECTURE)
    if not (PUBLIC / 'README.docx').is_file():
        raise FileNotFoundError('Create and visually review the editable public README.docx first')
    environment = os.environ.copy()
    environment.update(PYTHONPATH='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
    result = subprocess.run(
        [sys.executable, 'demo.py', '--device', device, '--output', 'results'],
        cwd=PUBLIC, env=environment, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    public_infer = import_public_module('infer')
    _, paths, _ = public_infer.read_evaluation_csv(
        PUBLIC / 'test_videos.csv', PUBLIC / 'demo_videos', limit=16)
    _, pixels = next(public_infer.iter_video_fields(paths))
    torch.set_num_threads(4)
    original, _ = load_lightgen(source, device)
    inference, _ = public_model.load_lightgen2(PUBLIC / 'weights/LightGen2.pt', device)
    with torch.inference_mode():
        expected = original(pixels=pixels.to(device))['prediction']
        actual = inference(pixels=pixels.to(device))['prediction']
    if actual.requires_grad or not torch.equal(expected, actual):
        raise ValueError('Public inference differs from the selected research model')
    optical_replay = verify_optical_arrays(PUBLIC / 'results')
    demo_files = tuple('demo_videos/' + row['sample_id'] for row in rows[:16])
    names = SOURCE_FILES + ARTIFACT_FILES + demo_files
    for name in names:
        path = PUBLIC / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix in ('.py', '.md', '.json', '.txt'):
            text = path.read_text('utf-8-sig')
            markers = ('Xml12', 'OneDrive', '/DATA/', '/home/', 'xwechat_files', 'model_parameters')
            if any(marker in text for marker in markers):
                raise ValueError('Private metadata in public file: ' + name)
    checksums = {name: file_sha256(PUBLIC / name) for name in sorted(names)}
    verification = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'canonical_directory': 'LightGen2', 'files': checksums,
        'checkpoint_export': exported, 'verification_device': device,
        'optical_array_replay': optical_replay,
        'demo_videos': 16, 'source_demo_prediction_maximum_difference': float((expected - actual).abs().max()),
    }
    release_root = ROOT / 'releases'
    release_root.mkdir(exist_ok=True)
    output = release_root / 'current'
    stage = Path(tempfile.mkdtemp(prefix='lightgen2-release-')).resolve()
    for name in names:
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PUBLIC / name, destination)
    verify_files(stage, checksums)
    archive = release_root / '.LightGen2.zip.building'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for name in names:
            zipped.write(stage / name, 'LightGen2/' + name)
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != {'LightGen2/' + name for name in names}:
            raise ValueError('Unexpected files in public ZIP')
        for name, digest in checksums.items():
            if hashlib.sha256(zipped.read('LightGen2/' + name)).hexdigest() != digest:
                raise ValueError('ZIP checksum differs: ' + name)
    if output.exists():
        resolved_output = output.resolve()
        if resolved_output.parent != release_root.resolve() or resolved_output.name != 'current':
            raise ValueError('Previous mirror is outside the release directory')
        backup = Path(tempfile.mkdtemp(prefix='lightgen2-previous-release-')).resolve()
        shutil.move(str(resolved_output), str(backup / 'current'))
    shutil.move(str(stage), str(output))
    archive.replace(release_root / 'LightGen2.zip')
    digest = file_sha256(release_root / 'LightGen2.zip')
    (release_root / 'LightGen2.zip.sha256').write_text(digest + '  LightGen2.zip\n', encoding='utf-8')
    verification['archive_sha256'] = digest
    write_json(ROOT / 'runs/simulation/current/validation/public_inference/release_verification.json', verification)
    print(json.dumps({'canonical_directory': str(PUBLIC), 'mirror': str(output),
                      'archive': str(release_root / 'LightGen2.zip'), 'files': len(names),
                      'archive_sha256': digest, 'source_prediction_maximum_difference': 0.0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    build_release(args.device)


if __name__ == '__main__':
    main()
