"""Build an explicit internal-review package; no datasets or full-Qwen weights.

Run from the reviewed Git checkout. Source is curated by a small allow-list;
assets are derived artifacts, independently hashed. Never overwrite output.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(args):
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        raise FileExistsError('Use a new bundle output; nothing is overwritten')
    args.output.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for name in ('lightgen_abo', 'configs', 'docs', 'tests', 'tools'):
        shutil.copytree(root / name, args.output / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('README.md', 'COMMAND.md', 'pyproject.toml'):
        shutil.copy2(root / name, args.output / name)
    if args.audit:
        reference = args.output / 'reference' / 'data_audit'
        reference.mkdir(parents=True)
        for name in ('audit.json', 'per_category.csv', 'test_products.csv', 'example_selection.json', 'crop_example_selection.json'):
            shutil.copy2(args.audit / name, reference / name)
        for name in ('03_error_examples.png', '04_actual_input_crop.png'):
            shutil.copy2(args.audit / name, args.output / 'docs/figures' / name)
    if args.verification:
        reference = args.output / 'reference' / 'fixed_equivalence'
        reference.mkdir(parents=True)
        for name in ('final_report.json', 'execution.json', 'equivalence.json', 'ccd_readout_audit.csv'):
            shutil.copy2(args.verification / name, reference / name)
        shutil.copy2(args.verification / 'phase_masks.png', args.output / 'docs/figures/phase_masks.png')
    if args.training:
        reference = args.output / 'reference' / 'training'
        reference.mkdir(parents=True)
        for name in ('final_report.json', 'history.json', 'execution.json'):
            shutil.copy2(args.training / name, reference / name)
    if args.smoke:
        reference = args.output / 'reference' / 'continuation_smoke'
        reference.mkdir(parents=True)
        for name in ('final_report.json', 'execution.json', 'history.json', 'selected.json', 'parameter_updates.json'):
            shutil.copy2(args.smoke / name, reference / name)
    shutil.copy2(root.parents[1] / 'AGENTS.md', args.output / 'AGENTS.md')
    assets = args.output / 'assets'
    assets.mkdir()
    shutil.copytree(args.processor, assets / 'processor')
    shutil.copy2(args.checkpoint, assets / 'best.pt')
    asset_manifest = dict(source_commit=commit, files={p.relative_to(assets).as_posix(): digest(p) for p in sorted(assets.rglob('*')) if p.is_file()})
    (assets / 'manifest.json').write_text(json.dumps(asset_manifest, indent=2), encoding='utf-8')
    manifest = dict(source_commit=commit, purpose='internal research review; simulation and fresh-optimizer continuation; not hardware SDK',
                    dataset_included=False, full_qwen_included=False,
                    files={p.relative_to(args.output).as_posix(): digest(p) for p in sorted(args.output.rglob('*')) if p.is_file()})
    (args.output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(args.output))
    archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    print(json.dumps(dict(source_commit=commit, zip=str(archive), sha256=digest(archive)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--processor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, help='Optional derived dataset analysis, not raw dataset')
    parser.add_argument('--verification', type=Path, help='Optional independent fixed-weight verification artifacts')
    parser.add_argument('--training', type=Path, help='Optional original completed training reports, not checkpoints')
    parser.add_argument('--smoke', type=Path, help='Optional independent continuation smoke reports, not checkpoints')
    build(parser.parse_args())
