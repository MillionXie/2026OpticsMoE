"""Package only the verified best optical LSP weights and committed source."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

TASK = Path('LightGenV2/tasks/t02_keypoint_detection')
RUN = TASK/'runs/simulation/refinement_20260909/staged_heatmap'
EXPECTED_SHA = '495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518'
EVIDENCE = ('best_checkpoint.pt', 'final_report.json', 'pose_protocol.json',
            'pose_protocol_split.csv', 'resolved_config.yaml', 'run_manifest.json',
            'source_anchor_test.json', 'student_architecture.json', 'training_history.json')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_paths(root):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    excluded = {'runs', 'releases', 'lab_bundles', 'hardware_sessions', 'artifacts', 'vendor_sdk', 'data', '__pycache__'}
    suffixes = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.sh', '.ps1'}
    return sorted(Path(n) for n in names if n and Path(n).suffix in suffixes
                  and not excluded.intersection(Path(n).parts)
                  and Path(n).parts[0] in {'LightGenV2', 'experiments', 'opticalmoe'})


def build(root, assets, output):
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root).strip():
        raise RuntimeError('Commit tracked changes before packaging')
    report = json.loads((assets/RUN/'final_report.json').read_text())
    if sha(assets/RUN/'best_checkpoint.pt') != EXPECTED_SHA or report['checkpoint_sha256'] != EXPECTED_SHA:
        raise RuntimeError('Not the verified best optical checkpoint')
    if report['best_epoch'] != 50 or abs(report['test']['pck_at_0.2_torso']-0.7347857142857143)>1e-12:
        raise RuntimeError('Unexpected source metrics')
    files = {p: root/p for p in source_paths(root)}
    files[Path('START_HERE.md')] = root/TASK/'HANDOFF_BEST.md'
    files.update({RUN/n: assets/RUN/n for n in EVIDENCE})
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    manifest = {'source_commit': commit, 'training_commit': json.loads((assets/RUN/'run_manifest.json').read_text())['git_commit'],
                'checkpoint_sha256': EXPECTED_SHA, 'selected_run': str(RUN),
                'external_assets': 'Reuse recipient LSP data and Qwen cache; no datasets or Qwen weights duplicated',
                'weights_included': ['best_checkpoint.pt'], 'files': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for relative, path in sorted(files.items()):
            z.write(path, relative.as_posix())
            manifest['files'].append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        z.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2, ensure_ascii=False))
    digest = sha(output)
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
    return {'zip': str(output), 'sha256': digest, 'bytes': output.stat().st_size, 'source_commit': commit}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[3], a.asset_root.resolve(), a.output.resolve()), indent=2))
