"""Accept intentionally pruned raw TIFFs only with an exact deletion audit trail.

All numeric model inputs (PNG and route JSON) retain their original SHA checks.
This does not reconstruct raw sensor pixels or change capture records.
"""
import argparse
import json
from pathlib import Path
from common import ROOT, read, sha, write

_cache = {}
OLD = """        for key,value in meta['files'].items():
            if sha(p/key)!=value: raise ValueError(f'CCD file changed: {p/key}')"""
NEW = """        from raw_cleanup import validate_record_files
        validate_record_files(root, p, record, meta)"""


def deletion_records(root):
    path = Path(root)/'raw_tiff_cleanup.jsonl'
    if not path.is_file(): return {}
    signature = (path.stat().st_mtime_ns, path.stat().st_size)
    key = str(path.resolve())
    if key not in _cache or _cache[key][0] != signature:
        entries = {}
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            row = json.loads(line)  # Fail closed on incomplete or corrupt journals.
            if row.get('action') == 'deleted': entries[(row['sample'], row['stage'])] = row
        _cache[key] = (signature, entries)
    return _cache[key][1]


def validate_record_files(root, folder, record_path, meta):
    for name, expected in meta['files'].items():
        path = Path(folder)/name
        if path.is_file():
            if sha(path) != expected: raise ValueError('CCD file changed: '+str(path))
            continue
        entry = deletion_records(root).get((meta['sample'], meta['stage']))
        allowed = (name == meta['stage']+'.tif' and entry is not None
            and entry['path'] == 'ccd/'+meta['sample']+'/'+name
            and entry['record_sha256'] == sha(record_path)
            and entry['recorded_tiff_sha256'] == expected
            and entry['png_sha256'] == meta['files'].get(meta['stage']+'.png'))
        if not allowed: raise FileNotFoundError('Missing CCD file without matching intentional-deletion record: '+str(path))


def install_reader():
    """Surgical migration for old lab adapters; retain all other local run.py code."""
    path = ROOT/'run.py'
    original = path.read_text(encoding='utf-8-sig')
    if NEW in original:
        print('Reader already supports audited TIFF removal'); return
    if original.count(OLD) != 1: raise RuntimeError('Unrecognized reader; no code modified')
    before = sha(path)
    backup = ROOT/'transfers'/('run_before_tiff_cleanup_'+before[:12]+'.py')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists(): backup.write_bytes(path.read_bytes())
    candidate = original.replace(OLD, NEW, 1)
    compile(candidate, str(path), 'exec')
    temp = path.with_name('run.raw_cleanup_install.tmp')
    if temp.exists(): raise FileExistsError(temp)
    temp.write_text(candidate, encoding='utf-8')
    temp.replace(path)
    write(ROOT/'transfers/raw_cleanup_reader_install.json', dict(before_sha256=before,
        after_sha256=sha(path), backup=str(backup), changed='Only measured() raw-file validation; capture/model/config unchanged'))
    print('Updated measured-data reader; backup:', backup)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-reader', action='store_true', required=True)
    p.parse_args()
    install_reader()
