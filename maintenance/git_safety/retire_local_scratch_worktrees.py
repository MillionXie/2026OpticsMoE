"""Retire two reviewed source-only scratch checkouts with recoverable archives.

Default is read-only. No force removal, no main checkout changes, no experiment
execution. Every timing-related file is additionally retained byte-for-byte in
an ignored ZIP, beyond the full committed-source Git bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]
TARGETS={'.codex_tmp/t07_robust_git':'398eeaaecfa76b05fd0b3656ca387024007a4f52',
         '.codex_tmp/t08_physical_git':'8e293d8731ba1fbb17b4a77e4d6ade3b0e785846'}
BRANCHES={'.codex_tmp/t07_robust_git':'codex/t07-robust35-20260926',
          '.codex_tmp/t08_physical_git':'codex/t08-physical-test-20260926'}
ARCHIVE=ROOT/'.codex_tmp/retired_worktrees_20261005'
TIMING=re.compile(r'timing|latency|benchmark|profile|power|energy|speed|tops',re.I)

def git(path,*args):
    result=subprocess.run(['git','-C',str(path),*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout.decode('utf-8',errors='replace')[-4000:])
    return result.stdout

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def process_guard():
    command="Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    rows=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command]).decode('utf-8-sig'))
    for row in rows:
        line=(row.get('CommandLine') or '').lower().replace('\\','/')
        if any(str((ROOT/relative).resolve()).lower().replace('\\','/') in line for relative in TARGETS):
            raise RuntimeError('A process still references the retired checkout: '+str(row['ProcessId']))

def inspect(relative,head):
    path=(ROOT/relative).resolve()
    # Explicit absolute targets, constrained to the exact scratch parent.
    if path.parent!=(ROOT/'.codex_tmp').resolve() or path.name not in ('t07_robust_git','t08_physical_git'):
        raise ValueError('Unexpected removal target')
    if not path.is_dir() or path.is_symlink():raise ValueError('Missing or linked checkout')
    if git(path,'rev-parse','HEAD').decode().strip()!=head:raise ValueError('HEAD changed')
    if git(path,'status','--porcelain','--untracked-files=no'):raise ValueError('Tracked edits present')
    if git(path,'ls-files','--others','--exclude-standard'):raise ValueError('Untracked content present')
    ignored=git(path,'ls-files','--others','--ignored','--exclude-standard','-z').decode().split('\0')
    extras=[p for p in ignored if p and '__pycache__/' not in p and '.pytest_cache/' not in p and not p.endswith('.pyc')]
    if extras:raise ValueError('Non-cache ignored assets present')
    tracked=[p for p in git(path,'ls-files','-z').decode().split('\0') if p]
    if any(line.startswith(b'160000 ') for line in git(path,'ls-files','--stage').splitlines()):raise ValueError('Submodule present')
    timing=[]
    for name in tracked:
        target=path/name
        if not target.resolve().is_relative_to(path) or target.is_symlink():raise ValueError('Linked tracked asset')
        if TIMING.search(name):timing.append(dict(path=name,bytes=target.stat().st_size,sha256=digest(target)))
    return dict(path=str(path),relative=relative,head=head,timing_files=timing,
                tracked_file_count=len(tracked),ignored_cache_files=sum(bool(p) for p in ignored))

def finish_removal(receipt,report):
    """Resume only the two archived checkouts; preserve OneDrive leftovers."""
    for name,key in [('bundle','bundle_sha256'),('timing_zip','timing_zip_sha256')]:
        target=Path(receipt[name])
        if target.parent.resolve()!=ARCHIVE.resolve() or digest(target)!=receipt[key]:
            raise RuntimeError('Recovery archive changed')
    git(ROOT,'bundle','verify',receipt['bundle'])
    before_head=git(ROOT,'rev-parse','HEAD')
    before_status=git(ROOT,'status','--porcelain','--untracked-files=no')
    if git(ROOT,'diff','--cached','--name-only'):raise RuntimeError('Shared index is not empty')
    if before_head.decode().strip()!=receipt['original_head']:raise RuntimeError('Original HEAD changed')
    expected_status=receipt.get('original_status_sha256')
    if expected_status and hashlib.sha256(before_status).hexdigest()!=expected_status:
        raise RuntimeError('Original tracked edits changed')
    for row in receipt['rows']:
        relative=row['relative']
        if TARGETS.get(relative)!=row['head']:raise RuntimeError('Unexpected archived target')
        path=(ROOT/relative).resolve()
        if str(path)!=row['path'] or path.parent!=(ROOT/'.codex_tmp').resolve():raise RuntimeError('Target changed')
        if git(ROOT,'rev-parse',row['archive_ref']).decode().strip()!=row['head']:raise RuntimeError('Archive ref changed')
        registered=[line[9:] for line in git(ROOT,'worktree','list','--porcelain').decode().splitlines() if line.startswith('worktree ')]
        if path.as_posix() in registered:
            process_guard();inspect(relative,row['head'])
            result=subprocess.run(['git','-C',str(ROOT),'worktree','remove',str(path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            row['remove_exit']=result.returncode
            row['remove_message']=result.stdout.decode('utf-8',errors='replace')[-2000:]
            report.write_text(json.dumps(receipt,indent=2),encoding='utf8')
            registered=[line[9:] for line in git(ROOT,'worktree','list','--porcelain').decode().splitlines() if line.startswith('worktree ')]
            if path.as_posix() in registered:raise RuntimeError('Git refused removal; no force used')
        if path.exists():
            process_guard()
            if (path/'.git').exists():raise RuntimeError('Residual still has Git metadata')
            committed={}
            for entry in git(ROOT,'ls-tree','-r','-z',row['head']).split(b'\0'):
                if not entry:continue
                metadata,name=entry.split(b'\t',1)
                committed[name.decode()]=metadata.split()[2]
            files=0
            with subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as batch:
                try:
                    for remaining in path.rglob('*'):
                        if remaining.is_symlink():raise RuntimeError('Linked residual')
                        if not remaining.is_file():continue
                        rel=remaining.relative_to(path).as_posix()
                        if '__pycache__/' in rel or '.pytest_cache/' in rel or rel.endswith('.pyc'):continue
                        if rel not in committed:raise RuntimeError('Unknown residual file: '+rel)
                        batch.stdin.write(committed[rel]+b'\n');batch.stdin.flush()
                        header=batch.stdout.readline().split()
                        if len(header)!=3 or header[1]!=b'blob':raise RuntimeError('Invalid archived object')
                        blob=batch.stdout.read(int(header[2]))
                        if batch.stdout.read(1)!=b'\n':raise RuntimeError('Invalid batch separator')
                        # Git attributes can normalize source newlines on checkout.
                        actual=remaining.read_bytes()
                        if actual!=blob and actual.replace(b'\r\n',b'\n')!=blob:
                            raise RuntimeError('Residual content changed: '+rel)
                        files+=1
                finally:
                    batch.stdin.close()
            destination=ARCHIVE/'residual'/path.name
            if destination.exists():raise RuntimeError('Residual destination already exists')
            destination.parent.mkdir(exist_ok=True)
            # Native PowerShell end-to-end, explicit validated absolute paths;
            # keep any filesystem leftovers recoverable rather than deleting.
            quote=lambda value:"'"+str(value).replace("'","''")+"'"
            command='Move-Item -LiteralPath '+quote(path)+' -Destination '+quote(destination)+' -ErrorAction Stop'
            subprocess.run(['powershell','-NoProfile','-Command',command],check=True)
            row['residual_archive']=str(destination);row['verified_residual_source_files']=files
        if relative not in receipt['removed']:receipt['removed'].append(relative)
        common=Path(git(ROOT,'rev-parse','--path-format=absolute','--git-common-dir').decode().strip()).resolve()
        admin=(common/'worktrees'/path.name).resolve()
        if admin.parent!=(common/'worktrees').resolve():raise RuntimeError('Unexpected metadata path')
        if admin.exists():
            if any((admin/name).exists() for name in ('HEAD','gitdir','index','locked')):
                raise RuntimeError('Residual Git metadata still active')
            destination=ARCHIVE/'git_admin'/path.name
            if destination.exists():raise RuntimeError('Metadata archive already exists')
            destination.parent.mkdir(exist_ok=True)
            quote=lambda value:"'"+str(value).replace("'","''")+"'"
            subprocess.run(['powershell','-NoProfile','-Command',
                'Move-Item -LiteralPath '+quote(admin)+' -Destination '+quote(destination)+' -ErrorAction Stop'],check=True)
            row['residual_git_admin_archive']=str(destination)
        branch='refs/heads/'+BRANCHES[relative]
        if ('branch '+branch+'\n') in git(ROOT,'worktree','list','--porcelain').decode():
            raise RuntimeError('Retired branch still checked out')
        branch_head=git(ROOT,'for-each-ref','--format=%(objectname)',branch).decode().strip()
        if branch_head:
            if branch_head!=row['head']:raise RuntimeError('Retired branch changed')
            # Cancel only the unused name with expected-value comparison. Its
            # entire history remains reachable via verified archive ref/bundle.
            git(ROOT,'update-ref','-d',branch,row['head'])
        row['retired_branch_name']=BRANCHES[relative]
        report.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    if git(ROOT,'rev-parse','HEAD')!=before_head or git(ROOT,'status','--porcelain','--untracked-files=no')!=before_status:
        raise RuntimeError('Original checkout changed concurrently')
    receipt['status']='complete'
    report.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(dict(status='complete',removed=receipt['removed'],recoverable=True,
        timing_files_retained=sum(len(row['timing_files']) for row in receipt['rows']),receipt=str(report))))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.resume:
        if args.apply:raise ValueError('Use either apply or resume')
        report=ARCHIVE/'receipt.json'
        finish_removal(json.loads(report.read_text(encoding='utf8')),report);return
    if not args.apply and (ARCHIVE/'receipt.json').exists():
        receipt=json.loads((ARCHIVE/'receipt.json').read_text(encoding='utf8'))
        print(json.dumps(dict(read_only=True,status=receipt['status'],removed=receipt['removed'],
            bundle_sha_verified=digest(Path(receipt['bundle']))==receipt['bundle_sha256'],
            timing_zip_sha_verified=digest(Path(receipt['timing_zip']))==receipt['timing_zip_sha256'])))
        return
    process_guard()
    rows=[inspect(relative,head) for relative,head in TARGETS.items()]
    if not args.apply:
        print(json.dumps(dict(read_only=True,candidates=[{k:v for k,v in row.items() if k!='timing_files'} |
            {'timing_files_to_retain':len(row['timing_files'])} for row in rows]),indent=2));return
    if ARCHIVE.exists():raise FileExistsError('Archive already exists; inspect receipt, do not repeat')
    if git(ROOT,'diff','--cached','--name-only'):raise RuntimeError('Shared index is not empty')
    original_head=git(ROOT,'rev-parse','HEAD')
    original_status=git(ROOT,'status','--porcelain','--untracked-files=no')
    ARCHIVE.mkdir()
    refs=[]
    for row in rows:
        ref='refs/archive/retired-local-20261005/'+Path(row['relative']).name
        if subprocess.run(['git','show-ref','--verify','--quiet',ref]).returncode==0:raise RuntimeError('Archive ref exists')
        git(ROOT,'update-ref',ref,row['head']);row['archive_ref']=ref;refs.append(ref)
    bundle=ARCHIVE/'committed_sources.bundle'
    git(ROOT,'bundle','create',str(bundle),*refs)
    git(ROOT,'bundle','verify',str(bundle))
    timing_archive=ARCHIVE/'all_timing_files.zip'
    with zipfile.ZipFile(timing_archive,'w',zipfile.ZIP_DEFLATED) as out:
        for row in rows:
            for item in row['timing_files']:
                out.write(Path(row['path'])/item['path'],Path(row['relative']).name+'/'+item['path'])
    with zipfile.ZipFile(timing_archive) as saved:
        if saved.testzip() is not None:raise RuntimeError('Timing archive CRC failure')
        for row in rows:
            for item in row['timing_files']:
                blob=saved.read(Path(row['relative']).name+'/'+item['path'])
                if hashlib.sha256(blob).hexdigest()!=item['sha256']:raise RuntimeError('Timing bytes changed')
    receipt=dict(status='archived_before_removal',bundle=str(bundle),bundle_sha256=digest(bundle),
                 timing_zip=str(timing_archive),timing_zip_sha256=digest(timing_archive),
                 rows=rows,removed=[],original_head=original_head.decode().strip(),
                 original_status_sha256=hashlib.sha256(original_status).hexdigest(),
                 restore='Use Git bundle/archive ref to restore the exact original HEAD; ZIP retains all timing file bytes.')
    report=ARCHIVE/'receipt.json'
    report.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    if git(ROOT,'rev-parse','HEAD')!=original_head or git(ROOT,'status','--porcelain','--untracked-files=no')!=original_status:
        raise RuntimeError('Main working checkout changed concurrently')
    finish_removal(receipt,report)

if __name__=='__main__':main()
