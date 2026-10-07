"""Baseline handoff launcher; stdlib-only checks, original task code at execution."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT/'TASK.json').read_text(encoding='utf-8'))

def verify():
    m=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in m['files']:
        p=ROOT/'source'/row['path']
        if os.name=='nt' and not str(p).startswith('\\\\?\\'):
            p=Path('\\\\?\\'+str(p.resolve()))
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Source differs from snapshot: '+row['path'])
    print('Verified',len(m['files']),'original files at',m['source_commit'])

def initialize_git():
    source=ROOT/'source'
    if (source/'.git').exists(): return
    subprocess.run(['git','init','--quiet'],cwd=source,check=True)
    subprocess.run(['git','add','.'],cwd=source,check=True)
    subprocess.run(['git','-c','user.name=Baseline Handoff','-c','user.email=baseline@localhost',
                    'commit','--quiet','-m','Import pinned baseline source snapshot'],cwd=source,check=True)
    # This is a real local snapshot commit, not an impersonation of the original
    # repository commit. SOURCE_ORIGIN.json preserves the historical identity.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','list','init-source']+list(SPEC['actions']))
    p.add_argument('--allow-new-repository', action='store_true',
                   help='Explicit permission for a standalone handoff repository; not normal main workflow')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.action=='check': verify(); return
    if a.action=='list':
        for name,command in SPEC['actions'].items(): print(name,': python -m',' '.join(command))
        return
    if a.action=='init-source':
        if not a.allow_new_repository:
            p.error('Creating a nested repository is disabled. Use the unified main checkout; standalone reproduction requires explicit --allow-new-repository permission.')
        verify(); initialize_git(); return
    if not (ROOT/'source/.git').is_dir():
        p.error('First run: python run_baseline.py init-source')
    tail=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    command=[sys.executable,'-m',*SPEC['actions'][a.action],*tail]
    result=subprocess.run(command,cwd=ROOT/'source')
    raise SystemExit(result.returncode)
if __name__=='__main__': main()
