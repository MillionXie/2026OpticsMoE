"""Desktop-only bounded ABO capture batch and recoverable quarantine.

Batch manifests are immutable and created BEFORE capture. Only the named
previously-uncaptured sample/stage files may be quarantined. No recursive delete.
"""
import argparse
import json
import re
import os
import subprocess
import sys
from pathlib import Path
from phase_fingerprint import digest
ROOT=Path(__file__).resolve().parent
STAGES=('vision_router','vision_expert','vision_global','language_router','language_expert','language_global')
SUFFIXES=('.png','.raw.png','.json','.capture.json','.quality.json','.route.json','.record.json')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(tmp,p)

def paths(spec,root=ROOT):
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',spec['session']):raise ValueError('Invalid session')
    if not re.fullmatch('[0-9a-f]{32}',spec['batch_id']):raise ValueError('Invalid batch ID')
    session=(root/'sessions'/spec['session']).resolve()
    if not session.is_relative_to(root.resolve()):raise ValueError('Session path escape')
    return session,session/'phase_batches'/(spec['batch_id']+'.json')

def initialize(spec,root=ROOT):
    session,dest=paths(spec,root)
    if dest.exists():raise FileExistsError('Batch ID already exists')
    stage=spec['stage'];ids=spec['ids']
    if stage not in STAGES or not isinstance(ids,list) or not 1<=len(ids)<=128 or len(ids)!=len(set(ids)):raise ValueError('Invalid batch')
    mf=read(session/'play'/stage/'manifest.json');entries={e['id']:e for e in mf['entries']}
    receipt=spec['phase_receipt'];verified=receipt.get('optical_verification',{})
    if not verified.get('passed') or verified.get('target')!=stage or receipt['phase_sha256']!=mf['phase_sha256']:raise ValueError('Missing/mismatched optical verification')
    if any(read(f)['state']=='pending' for f in (session/'phase_batches').glob('*.json')):raise ValueError('Recover pending batch before starting another')
    for ident in ids:
        if not re.fullmatch('[A-Za-z0-9_-]{1,100}',ident) or ident not in entries:raise ValueError('Sample outside prepared manifest')
        folder=(session/'ccd'/ident).resolve()
        if not folder.is_relative_to(session):raise ValueError('Sample path escape')
        if any((folder/(stage+s)).exists() for s in SUFFIXES):raise ValueError('Batch would overwrite existing sample files; recover/audit first')
    obj={'state':'pending','stage':stage,'ids':ids,'batch_id':spec['batch_id'],'session':spec['session'],
         'phase_receipt':receipt,'manifest_sha256':digest(mf),'hardware_identity':mf['hardware_identity']}
    write(dest,obj);return dest

def quarantine(spec,root=ROOT):
    session,p=paths(spec,root)
    if not p.exists():return  # failed before creating/capturing batch: no files touched
    obj=read(p)
    if obj['stage'] not in STAGES or any(not re.fullmatch('[A-Za-z0-9_-]{1,100}',i) for i in obj['ids']):raise ValueError('Invalid saved batch identity')
    if obj['state']=='quarantined':return
    if obj['state']!='pending':raise ValueError('Refuse to quarantine accepted batch')
    stage=obj['stage'];moves=[]
    for ident in obj['ids']:
        for suffix in SUFFIXES:
            src=(session/'ccd'/ident/(stage+suffix)).resolve()
            dst=(session/'quarantine'/obj['batch_id']/ident/(stage+suffix)).resolve()
            if not src.is_relative_to(session/'ccd') or not dst.is_relative_to(session/'quarantine'):raise ValueError('Quarantine path escape')
            if src.exists():
                if dst.exists():raise FileExistsError('Quarantine destination exists; audit interrupted move')
                dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst);moves.append(str(dst.relative_to(session)))
    obj.update(state='quarantined',moved_files=moves);write(p,obj)

def accept(spec,root=ROOT):
    session,p=paths(spec,root);obj=read(p)
    if obj['state']!='pending':raise ValueError('Batch not pending')
    proof=spec['verification']
    if not proof['postcheck']['passed'] or proof['postcheck']['target']!=obj['stage'] or proof['bank_id']!=obj['phase_receipt']['optical_verification']['bank_id']:raise ValueError('Invalid postcheck proof')
    files={}
    from phase_hdmi import sha
    for ident in obj['ids']:
        record=session/'ccd'/ident/(obj['stage']+'.record.json')
        r=read(record)
        if r['phase_sha256']!=obj['phase_receipt']['phase_sha256'] or r['hardware_identity']!=obj['hardware_identity']:raise ValueError('Captured record identity mismatch')
        for name,expected in r['files'].items():
            path=(record.parent/name).resolve()
            if not path.is_relative_to(record.parent.resolve()) or sha(path)!=expected:raise ValueError('Captured file integrity failed')
        files[ident]=sha(record)
    obj.update(state='accepted',verification=proof,record_sha256=files);write(p,obj)

def assert_no_pending(session):
    for p in (Path(session)/'phase_batches').glob('*.json'):
        if read(p)['state']=='pending':raise ValueError('Unverified capture batch exists; resume guarded coordinator to quarantine it first')

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    path=a.spec.resolve()
    if not path.is_relative_to(ROOT/'results/dual_jobs'):raise ValueError('Spec outside controlled job directory')
    spec=read(path)
    if spec['action']=='capture_batch':
        batch=initialize(spec)
        cmd=[sys.executable,str(ROOT/'run.py'),'capture','--session',spec['session'],'--stage',spec['stage'],
             '--config','LAB.local.json','--yes','--guard-batch',str(batch)]
        subprocess.run(cmd,cwd=ROOT,check=True)
    elif spec['action']=='quarantine_batch':quarantine(spec)
    elif spec['action']=='accept_batch':accept(spec)
    else:raise ValueError('Unknown batch action')

if __name__=='__main__':main()
