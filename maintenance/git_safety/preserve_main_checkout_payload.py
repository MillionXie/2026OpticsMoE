"""Archive the current bytes at risk in a reviewed main checkout preflight.

Never switches branches or moves/deletes originals. Archives remain private;
old tracked connection helpers and research outputs must never enter Git anew.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args]).decode().strip()
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--archive',help='New private archive directory under .codex_tmp; default read-only')
    args=parser.parse_args()
    plan_path=(ROOT/args.plan).resolve()
    if not plan_path.is_relative_to(ROOT/'.codex_tmp'):raise ValueError('Plan must be private')
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    if git('rev-parse','HEAD')!=plan['head'] or git('rev-parse','main')!=plan['main']:
        raise RuntimeError('Preflight refs changed; re-plan')
    names=set(plan['tracked_paths_removed_by_main'])|{r['path'] for r in plan['tracked_user_edits']}|{r['path'] for r in plan['existing_new_main_paths']}
    rows=[];missing=[]
    for name in sorted(names):
        path=ROOT/name
        if not path.exists():missing.append(name);continue
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink() or not path.is_file():
            raise RuntimeError('Special payload needs review: '+name)
        rows.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
    total=sum(row['bytes'] for row in rows)
    if not args.archive:
        print(json.dumps(dict(read_only=True,files=len(rows),bytes=total,already_missing=len(missing),originals_unchanged=True)));return
    destination=(ROOT/args.archive).resolve()
    if not destination.is_relative_to(ROOT/'.codex_tmp') or destination.exists():raise ValueError('Fresh private directory required')
    if shutil.disk_usage(ROOT).free<total*2+1024**3:raise RuntimeError('Insufficient safe free space')
    destination.mkdir()
    bundle=destination/'current_files.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as saved:
        for row in rows:saved.write(ROOT/row['path'],row['path'])
    with zipfile.ZipFile(bundle) as saved:
        if saved.testzip() is not None:raise RuntimeError('CRC failed')
        for row in rows:
            if hashlib.sha256(saved.read(row['path'])).hexdigest()!=row['sha256'] or sha(ROOT/row['path'])!=row['sha256']:
                raise RuntimeError('Source changed during backup: '+row['path'])
    if git('rev-parse','HEAD')!=plan['head'] or git('rev-parse','main')!=plan['main']:
        raise RuntimeError('Refs changed during backup')
    receipt=dict(status='verified_backup_only_no_checkout',head=plan['head'],main=plan['main'],
        archive=str(bundle),archive_sha256=sha(bundle),files=rows,already_missing=missing,
        original_paths_unchanged=True,timing_files_moved_or_deleted=False,checkout_performed=False)
    (destination/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(dict(status=receipt['status'],files=len(rows),uncompressed_bytes=total,
        archive_bytes=bundle.stat().st_size,archive_sha256=receipt['archive_sha256'],already_missing=len(missing),originals_unchanged=True)))

if __name__=='__main__':main()
