"""Read-only main checkout preflight; never stash, switch, remove or rewrite files.

Classifies user edits, existing untracked/ignored files that main would adopt,
and historical tracked removals. The result is a plan, not switch permission.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
TIMING=re.compile(r'timing|latency|benchmark|profile|power|energy|speed|tops',re.I)

def git(*args):
    run=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True)
    if run.returncode:raise RuntimeError(run.stderr.decode(errors='replace'))
    return run.stdout

def tree(ref):
    result={}
    for entry in git('ls-tree','-r','-z',ref).split(b'\0'):
        if not entry:continue
        metadata,name=entry.split(b'\t',1)
        mode,kind,oid=metadata.split()
        result[name.decode()]=(mode.decode(),kind.decode(),oid)
    return result

def blob_sha(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest().encode()

def classify_bytes(path,oid):
    actual=path.read_bytes()
    if blob_sha(actual)==oid:return 'byte_identical_to_main'
    if blob_sha(actual.replace(b'\r\n',b'\n'))==oid:return 'crlf_only_difference_from_main'
    return 'different_from_main_preserve_and_review'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',help='Optional private diagnostic JSON under .codex_tmp; no source changes')
    args=parser.parse_args()
    head=git('rev-parse','HEAD').decode().strip()
    target=git('rev-parse','main').decode().strip()
    status_before=git('status','--porcelain','--untracked-files=no')
    index=Path(git('rev-parse','--path-format=absolute','--git-path','index').decode().strip())
    index_sha=hashlib.sha256(index.read_bytes()).hexdigest()
    current=tree(head);wanted=tree(target)
    dirty=[v.decode()[3:] for v in status_before.splitlines()]
    # This repository's current protected names are ASCII; refuse quoted/rename
    # records instead of misinterpreting them as literal filenames.
    if any(name.startswith('"') or ' -> ' in name for name in dirty):
        raise RuntimeError('Quoted/renamed status needs explicit review')
    user_edits=[]
    for name in dirty:
        path=ROOT/name
        outcome='absent_from_main_preserve'
        if path.is_file() and name in wanted:
            outcome=classify_bytes(path,wanted[name][2])
        user_edits.append(dict(path=name,classification=outcome,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None))
    collisions=[]
    directory_blockers=set()
    for name,(mode,kind,oid) in wanted.items():
        if name in current:continue
        path=ROOT/name
        for parent in path.parents:
            if parent==ROOT:break
            if parent.is_file():directory_blockers.add(parent.relative_to(ROOT).as_posix())
        if not path.exists():continue
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
            classification='linked_path_preserve_and_review'
        elif mode not in ('100644','100755') or kind!='blob' or not path.is_file():
            classification='special_or_directory_preserve_and_review'
        else:classification=classify_bytes(path,oid)
        collisions.append(dict(path=name,classification=classification))
    removed=sorted(set(current)-set(wanted))
    timing_removed=[name for name in removed if TIMING.search(name)]
    untracked=[p for p in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if p]
    if git('rev-parse','HEAD').decode().strip()!=head or git('rev-parse','main').decode().strip()!=target:
        raise RuntimeError('Refs changed during read-only plan')
    if git('status','--porcelain','--untracked-files=no')!=status_before or hashlib.sha256(index.read_bytes()).hexdigest()!=index_sha:
        raise RuntimeError('User files/index changed during plan')
    result=dict(read_only=True,head=head,main=target,
        tracked_user_edits=user_edits,existing_new_main_paths=collisions,
        directory_blockers=sorted(directory_blockers),tracked_paths_removed_by_main=removed,
        protected_timing_paths_removed_by_main=timing_removed,untracked_count=len(untracked),
        main_file_difference_count=len([name for name in set(current)|set(wanted) if current.get(name)!=wanted.get(name)]),
        checkout_performed=False,automatic_checkout_allowed=False,
        required_followup='Preserve every user edit and collision, keep timing removals in place; obtain the pending checkout choice before switching. Use no-overwrite-ignore protection.')
    if args.report:
        report=(ROOT/args.report).resolve()
        if not report.is_relative_to(ROOT/'.codex_tmp') or report.suffix!='.json':
            raise ValueError('Diagnostic must stay in private scratch directory')
        report.write_text(json.dumps(result,indent=2),encoding='utf8')
        counts=lambda rows:{kind:sum(r['classification']==kind for r in rows) for kind in sorted({r['classification'] for r in rows})}
        print(json.dumps(dict(read_only=True,head=head,main=target,
            tracked_user_edits=counts(user_edits),existing_new_main_paths=counts(collisions),
            directory_blockers=len(directory_blockers),tracked_removals=len(removed),
            protected_timing_removals=len(timing_removed),untracked_count=len(untracked),
            main_file_difference_count=result['main_file_difference_count'],report=str(report),checkout_performed=False),indent=2))
    else:print(json.dumps(result,indent=2))

if __name__=='__main__':main()
