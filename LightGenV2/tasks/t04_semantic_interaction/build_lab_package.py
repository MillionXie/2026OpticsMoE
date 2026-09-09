"""Build a relocatable, simulation-only OURS release; no full Qwen/baseline cache."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import zipfile

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_module(name):
    p = REPO.joinpath(*name.split('.'))
    if p.with_suffix('.py').is_file():
        return p.with_suffix('.py')
    if (p / '__init__.py').is_file():
        return p / '__init__.py'
    return None


def source_closure():
    names = ['LightGenV2.tasks.t04_semantic_interaction.standalone',
             'LightGenV2.tasks.t04_semantic_interaction.training',
             'LightGenV2.tasks.t04_semantic_interaction.router_repair']
    seen = set()
    excluded = {'LightGenV2.tasks.t04_semantic_interaction.qwen_shared'}
    while names:
        name = names.pop()
        file = local_module(name)
        if name in excluded or file is None or file in seen:
            continue
        seen.add(file)
        package = name.split('.') if file.name == '__init__.py' else name.split('.')[:-1]
        for i in range(1, len(package) + 1):
            names.append('.'.join(package[:i]))
        tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = '.'.join(prefix + ([node.module] if node.module else []))
                names.append(module)
                names.extend(module + '.' + alias.name for alias in node.names if alias.name != '*')
    return seen


def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def json_out(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def build(run, data, output):
    import torch
    from safetensors.torch import save_file
    run, data, output = run.resolve(), data.resolve(), output.resolve()
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Use a new release directory; never overwrite a delivered ZIP')
    config = json.loads((run / 'resolved_config.json').read_text())
    payload = torch.load(run / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    expected = 't04_embedding_only_optical_alpha0.4001_0.9500_e2_scaleccd_positionlinear_v1_routerfill_sharedhead_v2'
    if payload['architecture'] != expected or payload['epoch'] != 40:
        raise ValueError('This release is exclusively OURS standard-head epoch40')
    output.mkdir(parents=True)
    sources = source_closure()
    for src in sources:
        copy(src, output / src.relative_to(REPO))
    # Constructors reference historical configuration paths dynamically. Include
    # configs only from the import-closure packages, not their runs/data/models.
    package_dirs = set()
    for src in sources:
        rel = src.relative_to(REPO).parts
        if rel[0] == 'experiments' and len(rel) > 2:
            package_dirs.add(REPO / rel[0] / rel[1])
        elif rel[:2] == ('LightGenV2', 'tasks') and len(rel) > 3:
            package_dirs.add(REPO.joinpath(*rel[:3]))
    for package in package_dirs:
        for src in (package / 'configs').rglob('*.yaml'):
            copy(src, output / src.relative_to(REPO))
    copy(TASK / 'standalone.py', output / 'reproduce.py')
    copy(run / 'best_checkpoint.pt', output / 'weights/best_checkpoint.pt')
    phase = {name: (2 * torch.pi * torch.sigmoid(t.float())) for name, t in payload['model'].items()
             if 'raw_phase' in name or 'raw_router_phase' in name}
    torch.save({'units': 'radians', 'layout': 'matrix[y,x]', 'epoch': 40,
                'source_checkpoint_sha256': sha(run / 'best_checkpoint.pt'), 'phases': phase}, output / 'weights/phases.pt')
    (output / 'frontend').mkdir()
    state = payload['model']
    save_file({'model.visual.patch_embed.proj.weight': state['vision_stem.proj.weight'].contiguous(),
               'model.visual.patch_embed.proj.bias': state['vision_stem.proj.bias'].contiguous(),
               'model.visual.pos_embed.weight': state['vision_stem.position'].contiguous()},
              str(output / 'frontend/model.safetensors'))
    config['shared_readout_variant'] = 'standard'
    # All external paths are rewritten by reproduce.py; avoid accidental use.
    for key in ['config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint', 'prompt_cache_path',
                'optical_base_config', 'legacy_warmstart_checkpoint']:
        config[key] = None
    json_out(output / 'settings.json', config)
    for filename in ['train.jsonl', 'test.jsonl', 'dataset_summary.json', 'token_embeddings_v1.pt', 'token_embeddings_v1.json']:
        copy(data / filename, output / 'data' / filename)
    for split in ['train', 'test']:
        records = [json.loads(line) for line in (data / (split + '.jsonl')).read_text().splitlines()]
        for record in records:
            sample_dir = Path(record['relative_dir'])
            for name in ['source.png', 'target.png', 'edit_mask.png']:
                copy(data / sample_dir / name, output / 'data' / sample_dir / name)
    asset_dir = Path(json.loads((run / 'resolved_config.json').read_text())['asset_dir'])
    for src in asset_dir.iterdir():
        if src.suffix in ['.png', '.json'] or src.name.lower().startswith(('license', 'readme')):
            copy(src, output / 'assets' / src.name)
    for name in ['selected_checkpoint_test_evaluation.json', 'same_checkpoint_remove_optical.json',
                 'student_architecture.json', 'training_report.json', 'run_manifest.json', 'environment.json',
                 'resolved_config.json', 'test_predictions.jsonl']:
        copy(run / name, output / 'reference' / name)
    for src in (run / 'best_visualization').rglob('*'):
        if src.is_file():
            copy(src, output / 'reference/best_visualization' / src.relative_to(run / 'best_visualization'))
    for name in ['phase_training_audit.json', 'training_history.csv']:
        copy(run / 'metrics' / name, output / 'reference/metrics' / name)
    for src in (TASK / 'release_template').iterdir():
        copy(src, output / src.name)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    files = [{'path': f.relative_to(output).as_posix(), 'bytes': f.stat().st_size, 'sha256': sha(f)}
             for f in sorted(output.rglob('*')) if f.is_file()]
    json_out(output / 'MANIFEST.json', {'schema_version': 1, 'kind': 'ours_simulation_standalone',
             'package_git_commit': commit, 'training_git_commit': json.loads((run / 'run_manifest.json').read_text())['git_commit'],
             'architecture': expected, 'source_checkpoint_sha256': sha(run / 'best_checkpoint.pt'),
             'source_modules': len(sources), 'no_full_qwen_no_baseline_no_hardware': True, 'files': files})
    zip_path = output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for f in sorted(output.rglob('*')):
            if f.is_file():
                archive.write(f, Path(output.name) / f.relative_to(output))
    value = {'zip': str(zip_path), 'bytes': zip_path.stat().st_size, 'sha256': sha(zip_path),
             'files': len(files) + 1, 'package_git_commit': commit}
    json_out(zip_path.with_suffix('.manifest.json'), value)
    zip_path.with_suffix('.zip.sha256').write_text(value['sha256'] + '  ' + zip_path.name + '\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.data_dir, args.output_dir), indent=2))
