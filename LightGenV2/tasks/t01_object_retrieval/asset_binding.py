"""Optional offline asset binding; historical frontend revision stays unknown."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_snapshot(snapshot: str | Path, identity_manifest: str | Path) -> dict:
    snapshot = Path(snapshot).expanduser().resolve(strict=True)
    manifest = Path(identity_manifest).expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError('Frontend snapshot must be a directory')
    raw = manifest.read_bytes()
    expected = json.loads(raw)['frozen_frontend']
    files = expected['files']
    required = {'model.safetensors', 'config.json', 'tokenizer.json',
                'tokenizer_config.json', 'preprocessor_config.json', 'chat_template.jinja'}
    if not required.issubset(files):
        raise ValueError('Frontend identity lacks required model/processor files')
    for name, row in files.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or str(relative) != name
                or '\\' in name or ':' in name):
            raise ValueError('Unsafe frontend identity path: '+name)
        path = snapshot / name
        before = path.stat()
        if not path.is_file() or before.st_size != row['bytes']:
            raise ValueError('Frontend size/type mismatch: '+name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Frontend changed during verification: '+name)
        if digest.hexdigest() != row['sha256']:
            raise ValueError('Frontend SHA mismatch: '+name)
    return {'snapshot_path': str(snapshot), 'model_id': expected['model_id'],
            'observed_cache_revision': expected.get('observed_cache_revision'),
            'identity_manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'files_verified': len(files), 'local_files_only': True,
            'historical_run_revision_proven': False}


def bind_frontend(settings, args) -> dict | None:
    snapshot = getattr(args, 'frontend_snapshot', None)
    manifest = getattr(args, 'frontend_identity', None)
    if snapshot is None and manifest is None:
        return None
    if not snapshot or not manifest or not args.run_dir:
        raise ValueError('Frontend binding requires snapshot, identity and a fresh --run-dir')
    output = Path(args.run_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Frontend binding refuses a nonempty existing run directory')
    binding = verify_snapshot(snapshot, manifest)
    if binding['model_id'] != settings.model_id:
        raise ValueError('Frontend model identity differs from the selected profile')
    settings.model_id = binding['snapshot_path']
    settings.local_files_only = True
    settings.cache_dir = None
    # A new frontend identity must not silently reuse an old teacher cache.
    # The legacy Settings exposes teacher_cache_path as a read-only property
    # derived from output_dir, not as an assignable field.
    settings.output_dir = output
    settings.lightgen_frontend_binding = binding
    return binding
