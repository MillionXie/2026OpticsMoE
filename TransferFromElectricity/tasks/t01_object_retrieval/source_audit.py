"""Compare content-addressed runtime source across recorded Git commits."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def runtime_path(path):
    p = Path(path)
    if p.suffix not in {'.py', '.yaml', '.yml'} or any(x in p.parts for x in ('runs', 'reports', 'tests')):
        return False
    if path.startswith('TransferFromElectricity/tasks/t01_object_retrieval/'):
        return not (p.name.startswith('report') or p.name in {'collect_results.py', 'source_audit.py', 'launch_rtx.py'})
    return (path.startswith('LightGenV2/tasks/t01_object_retrieval/')
            or path.startswith('experiments/qwen3_vl_embedding_2b_caltech101')
            or path.startswith('experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/')
            or path in {'TransferFromElectricity/__init__.py', 'TransferFromElectricity/tasks/__init__.py',
                        'LightGenV2/__init__.py', 'LightGenV2/tasks/__init__.py', 'experiments/__init__.py'})


def audit_sources(commits):
    rows = []
    for sha in sorted(set(commits)):
        entries = subprocess.check_output(['git', 'ls-tree', '-r', sha], cwd=ROOT, text=True)
        blobs = {}
        for line in entries.splitlines():
            metadata, path = line.split('\t', 1)
            if runtime_path(path):
                blobs[path] = metadata.split()[2]
        if not blobs:
            raise ValueError(f'No runtime source found for {sha}')
        fingerprint = hashlib.sha256(json.dumps(blobs, sort_keys=True).encode()).hexdigest()
        rows.append({'git_sha': sha, 'runtime_fingerprint': fingerprint, 'files': blobs})
    if len({r['runtime_fingerprint'] for r in rows}) != 1:
        raise ValueError('Runtime source differs; these runs cannot be matched automatically')
    return {'basis': 'Git blob content IDs for task training/configs and conservative Caltech/Grocery backend closure',
            'commits': rows, 'runtime_fingerprint': rows[0]['runtime_fingerprint']}
