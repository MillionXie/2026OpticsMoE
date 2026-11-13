"""Build the Google Drive inference package from the repository-root LightGen2."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import torch

if __package__:
    from . import _paths
    from .checkpoint_utils import export_inference_checkpoint, file_sha256
else:
    import _paths
    from checkpoint_utils import export_inference_checkpoint, file_sha256

if __package__ and __package__ != 'tools':
    from ..model import load_lightgen
    from ..infer import iter_video_fields
else:
    from model import load_lightgen
    from infer import iter_video_fields

ROOT = _paths.ROOT
PUBLIC = ROOT.parents[2]/'LightGen2'
RUNTIME_FILES = (
    '.gitignore', '.gitattributes', 'infer.py', 'model.py', 'README.md', '中文导读.md', 'MODEL_CARD.md',
    'requirements.txt', 'requirements-visualization.txt', 'LICENSE', 'NOTICE',
    'licenses/QWEN_LICENSE.txt', 'examples/make_example.py',
    'tools/__init__.py', 'tools/_paths.py', 'tools/plot_style.py',
    'tools/phase_plotting.py', 'tools/visualize_fields.py', 'tools/plot_global_ablation.py',
)
RESULT_FILES = (
    'global_ablation_summary.csv', 'global_ablation_seeds.csv', 'global_ablation_protocol.json',
    'global_ablation.png', 'phase_masks.png', 'optical_expert_share.png',
    'phase_controls.json', 'propagation_controls.json', 'reference_metrics.json',
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', 'utf-8', newline='\n')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_results(run,output):
    output.mkdir(exist_ok=True)
    summaries={}
    conditions={}
    directories = [run/'global_ablation']
    if (run/'global_ablation/initialization_control').exists():
        directories.append(run/'global_ablation/initialization_control')
    for directory in directories:
        if json.loads((directory/'status.json').read_text('utf-8-sig'))['state']!='completed':
            raise ValueError('Global ablation is incomplete')
        protocol = json.loads((directory/'protocol.json').read_text('utf-8-sig'))
        if protocol['checkpoint_sha256'] != file_sha256(run/'LightGen.pt'):
            raise ValueError('Global ablation belongs to another checkpoint')
        for row in read_csv(directory/'summary.csv'):
            summaries[row['kind'],row['stage']]=row
        for row in read_csv(directory/'all_conditions.csv'):
            conditions[row['kind'],row['stage'],row['seed']]=row
    write_csv(output/'global_ablation_summary.csv',list(summaries.values()))
    write_csv(output/'global_ablation_seeds.csv',list(conditions.values()))
    original=json.loads((run/'global_ablation/protocol.json').read_text('utf-8-sig'))
    write_json(output/'global_ablation_protocol.json',{
        'role':'post_training_forward_interventions_on_original_TEST_development',
        'videos':558,'retraining':False,'random_seeds':original['seeds'],
        'grouping':'original_order_with_cyclic_tail_padding_and_valid_sample_mask',
        'bypass_applied': any(row['kind'] == 'bypass' for row in conditions.values()),
        'random_phase':'uniform physical phase in [0,2*pi), with trained mappings',
        'random_raw_phase':'raw N(0,0.25^2) through the original sigmoid, with trained mappings',
        'random_global':'reinitialize the current GlobalStage parameters and use uniform physical phase; retain the final task readout',
        'random_global_default':'complete default GlobalStage initialization; retain the final task readout',
        'standard_deviation':'sample SD over ten seeds',
    })
    for name in ('phase_masks.png','optical_expert_share.png'):
        shutil.copyfile(run/'figures'/name,output/name)
    shutil.copyfile(run/'phase_effect/phase_effect.json', output/'phase_controls.json')
    if (run/'propagation_controls.json').exists():
        shutil.copyfile(run/'propagation_controls.json', output/'propagation_controls.json')
    metrics = json.loads((run/'metrics.json').read_text('utf-8'))
    control = json.loads((run/'electronic_capacity_control.json').read_text('utf-8'))
    write_json(output/'reference_metrics.json', {
        'evaluation_scope': 'original_TEST_development', 'videos': 558,
        'reference': {key: metrics[key] for key in ('n', 'srcc', 'krcc', 'plcc', 'rmse', 'mae')},
        'direct_video_srcc': 0.7992454772867189,
        'slot_cycle_mean_srcc': 0.7953460323798703,
        'slot_cycle_minimum_srcc': 0.7885972250393563,
        'electronic_control': {'role': control['role'], 'parameters': control['parameters'],
                               'head': control['head'], 'metrics': control['metrics'],
                               'initialization': control['initialization'],
                               'comparison_context': control['comparison_context']},
    })


def verify_manifest(directory):
    manifest = json.loads((directory/'MANIFEST.json').read_text('utf-8-sig'))
    for name, expected in manifest['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != expected:
            raise ValueError('Release file checksum differs: '+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'releases/current')
    parser.add_argument('--readme-pdf', type=Path, help='Matching PDF edition of the public README')
    args = parser.parse_args()
    output = args.out.resolve()
    release_root = (ROOT/'releases').resolve()
    if output.parent != release_root or output.name != 'current':
        raise ValueError('The generated mirror uses releases/current')
    if not PUBLIC.is_dir():
        raise FileNotFoundError(PUBLIC)
    if args.readme_pdf is not None and (args.readme_pdf.suffix.lower() != '.pdf' or not args.readme_pdf.is_file()):
        raise ValueError('The README reading edition must be an existing PDF file')
    release_root.mkdir(exist_ok=True)
    profile = json.loads((ROOT/'configs/selected.json').read_text('utf-8-sig'))
    source = ROOT/profile['checkpoint']
    if file_sha256(source) != profile['checkpoint_sha256']:
        raise ValueError('Selected research checkpoint checksum differs')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_lightgen2_release_model', PUBLIC/'model.py')
    public_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = public_module
    spec.loader.exec_module(public_module)
    exported = export_inference_checkpoint(
        source, PUBLIC/'weights/LightGen.pt',
        [field.name for field in fields(public_module.ModelConfig)], public_module.ARCHITECTURE)
    (PUBLIC/'configs').mkdir(exist_ok=True)
    write_json(PUBLIC/'configs/selected.json', {
        'checkpoint': 'weights/LightGen.pt',
        'checkpoint_sha256': exported['public_checkpoint_sha256'],
        'architecture': public_module.ARCHITECTURE,
        'model_parameters': profile['model_parameters'],
    })
    copy_results(source.parent, PUBLIC/'results')
    environment = os.environ.copy()
    environment.update(PYTHONPATH='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
    commands = [
        [sys.executable, 'examples/make_example.py', '--output', 'examples/demo.mp4'],
        [sys.executable, 'infer.py', '--weights', 'weights/LightGen.pt', '--videos', 'examples/demo.mp4',
         '--device', 'cpu', '--output', 'examples/expected_predictions.json'],
        [sys.executable, 'tools/plot_global_ablation.py', '--summary', 'results/global_ablation_summary.csv',
         '--output', 'results/global_ablation.png'],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=PUBLIC, env=environment, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    torch.set_num_threads(4)
    original, _ = load_lightgen(source)
    inference, _ = public_module.load_lightgen(PUBLIC/'weights/LightGen.pt')
    if sum(parameter.numel() for parameter in inference.parameters()) != profile['model_parameters']:
        raise ValueError('Inference parameter count differs')
    _, pixels = next(iter_video_fields([PUBLIC/'examples/demo.mp4']))
    with torch.inference_mode():
        expected = original(pixels=pixels)['prediction']
        actual = inference(pixels=pixels)['prediction']
    if actual.requires_grad:
        raise ValueError('The public forward must run in inference mode')
    maximum_difference = float((expected-actual).abs().max())
    if not torch.equal(expected, actual):
        raise ValueError('Inference forward differs from the selected research model')
    if args.readme_pdf is not None:
        shutil.copyfile(args.readme_pdf, PUBLIC/'README.pdf')
    names = list(RUNTIME_FILES)+['weights/LightGen.pt', 'configs/selected.json',
                                'examples/demo.mp4', 'examples/expected_predictions.json']
    names += ['results/'+name for name in RESULT_FILES]
    if args.readme_pdf is not None:
        names.append('README.pdf')
    for name in names:
        path = PUBLIC/name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix in ('.py', '.md', '.json', '.csv', '.txt'):
            text = path.read_text('utf-8-sig')
            if any(marker in text for marker in ('Xml12', 'OneDrive', '/DATA/', '/home/', 'xwechat_files')):
                raise ValueError('Local/private text in release file: '+name)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {
        'project': 'LightGen-II temporal inference', 'license': 'Apache-2.0',
        'source_commit': commit, 'canonical_directory': 'LightGen2',
        'checkpoint_export': exported,
        'files': {name: file_sha256(PUBLIC/name) for name in sorted(names)},
        'verification': {'standalone_empty_PYTHONPATH': True,
                         'retained_source_tensors_equal': True, 'inference_mode_forward': True,
                         'source_demo_prediction_maximum_difference': maximum_difference,
                         'example_MOS': float(actual[0, 0])},
    }
    write_json(PUBLIC/'MANIFEST.json', manifest)
    verify_manifest(PUBLIC)
    stage = Path(tempfile.mkdtemp(prefix='lightgen2-release-')).resolve()
    for name in names+['MANIFEST.json']:
        destination = stage/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PUBLIC/name, destination)
    verify_manifest(stage)
    archive = release_root/'.LightGen2.zip.building'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for name in names+['MANIFEST.json']:
            zipped.write(stage/name, 'LightGen2/'+name)
    with zipfile.ZipFile(archive) as zipped:
        for name, digest in manifest['files'].items():
            if hashlib.sha256(zipped.read('LightGen2/'+name)).hexdigest() != digest:
                raise ValueError('ZIP checksum differs: '+name)
    if output.exists():
        verify_manifest(output)
        backup = Path(tempfile.mkdtemp(prefix='lightgen2-previous-release-')).resolve()
        shutil.move(str(output), str(backup/'current'))
    shutil.move(str(stage), str(output))
    archive.replace(release_root/'LightGen2.zip')
    digest = file_sha256(release_root/'LightGen2.zip')
    (release_root/'LightGen2.zip.sha256').write_text(digest+'  LightGen2.zip\n', 'utf-8')
    print(json.dumps({'canonical_directory': str(PUBLIC), 'mirror': str(output),
                      'archive': str(release_root/'LightGen2.zip'), 'archive_sha256': digest,
                      'files': len(names)+1, 'weights': exported,
                      'source_prediction_maximum_difference': maximum_difference}, indent=2))


if __name__ == '__main__':
    main()
