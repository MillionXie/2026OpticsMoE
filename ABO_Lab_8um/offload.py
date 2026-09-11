"""Measured-data-only offload of language_global preparation. Never opens hardware."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
from common import ROOT, STAGES, CHECKPOINT_SHA, config, hardware_identity, read, sha, write, session_path

STAGE='language_global'


def normalized(path):
    p=PurePosixPath(str(path).replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p): raise ValueError('Unsafe relative path: '+str(p))
    return str(p)


def required(sample):
    return STAGES[3:5] if sample['kind']=='title' else STAGES[:5]


def file_hash(data): return hashlib.sha256(data).hexdigest()


def export_snapshot(name, target):
    c,_=config(); r=session_path(name); state=read(r/'session.json')
    if hardware_identity(c)!=state['hardware_identity']: raise ValueError('Session hardware changed')
    if state['checkpoint_sha256']!=CHECKPOINT_SHA: raise ValueError('Checkpoint mismatch')
    if c['amplitude_slm'].get('gray_lut_file'): raise ValueError('External LUT offload not supported; do not omit it silently')
    if any((r/'ccd'/s['id']/(STAGE+'.record.json')).exists() for s in state['samples']):
        raise ValueError('Target stage already has captures; do not replace its inputs')
    # Validate completion BEFORE making a ZIP, rather than failing halfway through inference.
    missing=[]
    for s in state['samples']:
        for stage in required(s):
            if not (r/'ccd'/s['id']/(stage+'.record.json')).exists(): missing.append([s['id'],stage])
    if missing: raise ValueError('Missing upstream real CCDs: '+str(missing[:10])+f'; total={len(missing)}')
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    inventory={}; source_records={}; counts={stage:0 for stage in STAGES[:5]}
    def add(z, path, name):
        name=normalized(name); data=Path(path).read_bytes()
        inventory[name]=file_hash(data); z.writestr(name,data)
    from phase_encoding import generated_root
    phase=generated_root(ROOT,c)/'P/06_language_global.bmp'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        add(z,r/'session.json','session.json'); add(z,r/'optical_contract.json','optical_contract.json')
        add(z,phase,'phase.bmp')
        for i,s in enumerate(state['samples']):
            sid=normalized(s['id']); d=r/'ccd'/sid
            if s['kind']=='image':
                path=ROOT/normalized(s['path'])
                if sha(path)!=s['source_sha256']: raise ValueError('Image source changed: '+sid)
                add(z,path,'images/'+sid+path.suffix)
            for stage in required(s):
                rp=d/(stage+'.record.json'); record=read(rp)
                if record['hardware_identity']!=state['hardware_identity'] or record.get('capture_mode')!='real':
                    raise ValueError('Not a matching real capture: '+str(rp))
                if record['stage']!=stage or record['sample']!=sid: raise ValueError('Record identity mismatch')
                src=stage+('.route.json' if stage.endswith('router') else '.png')
                expected=record['files'].get(src)
                if not expected or sha(d/src)!=expected: raise ValueError('Measured input changed: '+str(d/src))
                add(z,d/src,'ccd/'+sid+'/'+src); add(z,rp,'ccd/'+sid+'/'+rp.name)
                source_records['ccd/'+sid+'/'+rp.name]=sha(rp)
                counts[stage]+=1
            if (i+1)%250==0: print('Snapshot',i+1,'/',len(state['samples']),flush=True)
        # Pin frozen frontend and replay-critical code against existing server installation.
        fingerprints={}
        for base in [ROOT/'models/Qwen3-VL-Embedding-2B', ROOT/'runtime/abo_dual']:
            for p in sorted(base.rglob('*')):
                if p.is_file() and (p.suffix in ('.json','.py','.txt') or p.name=='native_student.safetensors'):
                    fingerprints[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
        for n in ['patterns.py','memory.py']:
            fingerprints[n]=sha(ROOT/n)
        manifest={'schema':1,'session':name,'target_stage':STAGE,'samples':len(state['samples']),
            'counts':counts,'hardware_identity':state['hardware_identity'],'checkpoint_sha256':CHECKPOINT_SHA,
            'phase_sha256':sha(phase),'source_records':source_records,'files':inventory,
            'runtime_fingerprints':fingerprints,
            'integrity_scope':'All consumed PNG/route JSON + record bytes verified; full raw TIFF hashes retained in source records, not recomputed or transferred.'}
        z.writestr('SNAPSHOT.json',json.dumps(manifest,indent=2))
    write(target.with_suffix('.report.json'),{'zip':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'manifest':manifest})
    print('SNAPSHOT_READY',target,target.stat().st_size,sha(target),flush=True)


def unpack_snapshot(archive, dest):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('SNAPSHOT.json'))
        for name,h in mf['files'].items():
            name=normalized(name); data=z.read(name)
            if file_hash(data)!=h: raise ValueError('Snapshot corruption: '+name)
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    write(dest/'SNAPSHOT.json',mf)
    return mf


def compute(archive, dest, device, limit=0):
    start=time.perf_counter(); dest=Path(dest)
    mf=unpack_snapshot(archive,dest/'snapshot'); snap=dest/'snapshot'
    for n,h in mf['runtime_fingerprints'].items():
        if sha(ROOT/normalized(n))!=h: raise ValueError('Server frontend/replay code mismatch: '+n)
    import torch
    from backend import create,forward,optical_contract
    from patterns import amplitude,save
    from common import setup_imports
    setup_imports();from abo_dual.backend import NeedCapture
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    b=create(device,force_fp32=True); b.guard_optics()
    if optical_contract(b)!=read(snap/'optical_contract.json'): raise ValueError('Optical contract mismatch')
    state=read(snap/'session.json');c=state['hardware_config']; entries=[]
    selected=state['samples'][:limit or None]
    play=dest/'play';play.mkdir()
    for i,original in enumerate(selected):
        s=dict(original); sid=s['id']; values={}
        if s['kind']=='image':
            matches=list((snap/'images').glob(sid+'.*'))
            if len(matches)!=1: raise ValueError('Missing/ambiguous source image')
            s['path']=str(matches[0])
        for stage in required(s):
            p=snap/'ccd'/sid
            if stage.endswith('router'): values[stage]=read(p/(stage+'.route.json'))
            else:
                with Image.open(p/(stage+'.png')) as im: values[stage]=np.asarray(im).copy()
        try: forward(b,s,values,release=True)
        except NeedCapture as e:
            if e.stage!=STAGE: raise RuntimeError('Missing measured dependency: '+e.stage)
            bmp,encoding=amplitude(e.amplitude,c);name=f'{i:05d}.bmp';save(play/name,bmp)
            entries.append({'id':sid,'bmp':name,'sha256':sha(play/name),'encoding':encoding})
        else: raise RuntimeError('Target was not guarded')
        if (i+1)%25==0:print('REMOTE_PREPARED',i+1,'/',len(selected),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    from phase_encoding import generated_root
    phase_rel=str((generated_root(ROOT,c)/'P/06_language_global.bmp').relative_to(ROOT))
    write(play/'manifest.json',{'stage':STAGE,'phase_file':phase_rel,
        'phase_sha256':mf['phase_sha256'],'hardware_identity':mf['hardware_identity'],'entries':entries})
    import subprocess
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'snapshot_sha256':sha(archive),'source_session_sha256':mf['files']['session.json'],
        'source_records':mf['source_records'],'source_consumed_files':{n:h for n,h in mf['files'].items() if n.startswith('ccd/')},
        'samples':len(entries),'complete':not limit or limit>=len(state['samples']),
        'precision':'FP32, TF32 disabled, real measured upstream stages only',
        'torch':torch.__version__,'numpy':np.__version__,'elapsed_s':time.perf_counter()-start}
    write(play/'offload_report.json',report)
    result=dest/'language_global_inputs.zip'
    with zipfile.ZipFile(result,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(play.iterdir()):z.write(p,p.name)
    print('REMOTE_READY',result,result.stat().st_size,sha(result),flush=True)


def install(name, archive):
    c,_=config();r=session_path(name); state=read(r/'session.json')
    if state['hardware_identity']!=hardware_identity(c): raise ValueError('Current hardware changed')
    target=r/'play'/STAGE
    if target.exists(): raise FileExistsError('Target already exists; refusing overwrite: '+str(target))
    if any((r/'ccd'/s['id']/(STAGE+'.record.json')).exists() for s in state['samples']):raise ValueError('Target CCD already exists')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('manifest.json')); report=json.loads(z.read('offload_report.json'))
        if mf['hardware_identity']!=hardware_identity(c) or mf['stage']!=STAGE:raise ValueError('Returned identity mismatch')
        if sha(ROOT/mf['phase_file'])!=mf['phase_sha256']:raise ValueError('Phase changed')
        if sha(r/'session.json')!=report['source_session_sha256']:raise ValueError('Session changed')
        if not report['complete'] or [e['id'] for e in mf['entries']]!=[s['id'] for s in state['samples']]:raise ValueError('Sample coverage/order mismatch')
        for n,h in report['source_consumed_files'].items():
            if sha(r/normalized(n))!=h:raise ValueError('Upstream source changed: '+n)
        names=[e['bmp'] for e in mf['entries']]
        if len(set(names))!=len(names):raise ValueError('Duplicate BMPs')
        stage=r/'play'/('language_global_offload_install_'+str(time.time_ns()));stage.mkdir(parents=True)
        for e in mf['entries']:
            n=normalized(e['bmp'])
            if '/' in n or not n.endswith('.bmp'):raise ValueError('Unsafe BMP filename')
            data=z.read(n)
            if file_hash(data)!=e['sha256']:raise ValueError('BMP corrupted: '+n)
            with Image.open(io.BytesIO(data)) as im:
                if list(im.size)!=c['amplitude_slm']['expected_resolution_wh'] or im.mode!='L':raise ValueError('Invalid SLM raster')
            (stage/n).write_bytes(data)
        write(stage/'manifest.json',mf);write(stage/'offload_report.json',report)
        stage.rename(target)
    print('INSTALLED',len(mf['entries']),'verified BMPs:',target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['export','compute','install'])
    p.add_argument('--session',default='pilot02');p.add_argument('--archive',required=True)
    p.add_argument('--out');p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.action=='export':export_snapshot(a.session,a.archive)
    elif a.action=='compute':
        if not a.out:p.error('--out required')
        compute(a.archive,a.out,a.device,a.limit)
    else:install(a.session,a.archive)
