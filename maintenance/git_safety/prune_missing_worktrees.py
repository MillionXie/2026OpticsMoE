"""Prune only explicitly reviewed missing registrations after private recovery backup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile


def inspect(root, expected):
    root=Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.STDOUT)
    raw_common=Path(git('rev-parse','--git-common-dir').decode().strip())
    common=(raw_common if raw_common.is_absolute() else root/raw_common).resolve()
    before=git('worktree','prune','--dry-run','--verbose','--expire','now').decode()
    ids=re.findall(r'^Removing worktrees/([^:]+): gitdir file points to non-existent location$',before,re.M)
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected) or len(before.splitlines())!=len(ids):
        raise RuntimeError('Prune preview differs from explicit missing-only list')
    rows=[]
    active=[]
    proc=Path('/proc')
    if proc.is_dir():
        for p in proc.iterdir():
            if p.name.isdigit():
                try:active.append(str((p/'cwd').resolve(strict=True)))
                except (OSError,RuntimeError):pass
    for name in ids:
        if not re.fullmatch('[A-Za-z0-9_-]+',name):raise RuntimeError('Unsafe administrative id')
        admin=(common/'worktrees'/name).resolve()
        if not admin.is_relative_to(common/'worktrees') or admin.is_symlink():
            raise RuntimeError('Administrative path escape')
        target=Path((admin/'gitdir').read_text().strip())
        if not target.is_absolute() or target.name!='.git':raise RuntimeError('Unexpected gitdir')
        target_parent=target.parent.resolve()
        if target.exists() or target.parent.exists():raise RuntimeError('Working directory exists; preserve')
        if any(p==str(target_parent) or p.startswith(str(target_parent)+os.sep) for p in active):
            raise RuntimeError('Missing directory has an active process')
        if (admin/'locked').exists() or list(admin.glob('*.lock')):
            raise RuntimeError('Locked or concurrently edited registration')
        head=(admin/'HEAD').read_text().strip()
        if not re.fullmatch('[0-9a-f]{40}',head):raise RuntimeError('Only detached commits allowed')
        git('cat-file','-e',head+'^{commit}')
        rows.append(dict(id=name,missing_path=str(target_parent),head=head))
    return git,common,before,rows


def run(root, expected, backup_dir=None, apply=False):
    git,common,preview,rows=inspect(root,expected)
    result=dict(rows=rows,preview=preview,apply=False,scientific_files_deleted=False)
    if not apply:return result
    backup=Path(backup_dir).resolve()
    if backup.exists():raise RuntimeError('Recovery directory exists; never overwrite')
    if backup.is_relative_to(common) or backup==Path(root).resolve():
        raise RuntimeError('Recovery backup must be outside Git administration')
    head=git('rev-parse','HEAD');status=git('status','--porcelain','-uno')
    raw_index=Path(git('rev-parse','--git-path','index').decode().strip())
    index=(raw_index if raw_index.is_absolute() else Path(root).resolve()/raw_index)
    index_before=index.read_bytes() if index.exists() else None
    backup.mkdir(parents=True)
    archive=backup/'missing_worktree_administration.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for row in rows:tar.add(common/'worktrees'/row['id'],arcname=row['id'],recursive=True)
    # Detached source commits remain reachable even after registration removal.
    for row in rows:
        ref='refs/archive/missing-worktrees-20261004/'+row['id']
        git('check-ref-format',ref)
        lookup=subprocess.run(['git','-C',str(root),'show-ref','--verify','--quiet',ref])
        if lookup.returncode!=1:raise RuntimeError('Recovery ref already exists or failed lookup')
        git('update-ref',ref,row['head'],'0'*40)
        row['recovery_ref']=ref
    _,_,again,rows_again=inspect(root,expected)
    if again!=preview or [(r['id'],r['head']) for r in rows_again]!=[(r['id'],r['head']) for r in rows]:
        raise RuntimeError('Registration changed; backup retained, prune not run')
    if git('rev-parse','HEAD')!=head or git('status','--porcelain','-uno')!=status:
        raise RuntimeError('Checkout changed; stop')
    output=git('worktree','prune','--verbose','--expire','now').decode()
    if any((common/'worktrees'/r['id']).exists() for r in rows):
        raise RuntimeError('Prune incomplete; inspect recovery receipt')
    if (git('rev-parse','HEAD')!=head or git('status','--porcelain','-uno')!=status
            or (index.read_bytes() if index.exists() else None)!=index_before):
        raise RuntimeError('Unexpected original checkout/index change')
    result.update(apply=True,output=output,backup=str(archive),
                  backup_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  original_checkout_unchanged=True,removed_registration_count=len(rows))
    (backup/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--expected-missing',nargs='+',required=True)
    parser.add_argument('--backup-dir')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.apply and not args.backup_dir:parser.error('--apply requires --backup-dir')
    print(json.dumps(run(args.root,args.expected_missing,args.backup_dir,args.apply),indent=2))
