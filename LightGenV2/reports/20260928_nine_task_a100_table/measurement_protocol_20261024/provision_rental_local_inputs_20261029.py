"""Same512 original bytes on local POSIX overlay, retain ALL metadata gates.

Failed timing01 is preserved, never promoted. Only our verified input symlinks
may be handed over; original shared-disk files and links remain recoverable.
No model/CUDA, training, placeholders, timestamp editing or checksum weakening.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

BASE=Path('/root/data/lgvq_20261029')
LOCAL=Path('/root/lgvq_inputs_local_20261029_01')
MANIFEST_SHA='daa3bf90db81077b5ef50ff0bf84d6f34b509e613a6006ea8d24a22cb8d4b849'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def metadata(p):
    s=Path(p).stat()
    return dict(size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)

def decode_metadata_control(source):
    # A SEPARATE, non-measured inode: never preload decoded TEST inputs for a
    # measured call. This tests OpenCV's metadata behaviour on the new storage.
    import cv2
    target=LOCAL/'metadata_decode_control01.mp4'
    with Path(source).open('rb') as f,target.open('xb') as out:shutil.copyfileobj(f,out)
    target.chmod(0o444);before=metadata(target);digest=sha(target)
    video=cv2.VideoCapture(str(target));total=int(video.get(cv2.CAP_PROP_FRAME_COUNT));positions=[]
    try:
        if total<=0:raise ValueError('Control video must decode')
        for fraction in (.1,.37,.63,.9):
            position=min(total-1,max(0,round((total-1)*fraction)));video.set(cv2.CAP_PROP_POS_FRAMES,position)
            ok,frame=video.read()
            if not ok:raise ValueError('Control frame decode failed')
            positions.append(position)
    finally:video.release()
    if metadata(target)!=before or sha(target)!=digest:raise ValueError('Local OpenCV metadata stability failed')
    return dict(path=str(target),separate_non_measured_inode=True,frames=4,positions=positions,
                metadata_stable=True,bytes_SHA256=digest,no_decoded_input_reused=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    if not 0<a.deadline-time.time()<=600:raise ValueError('One finite CPU data provision only')
    if LOCAL.is_symlink() or LOCAL.resolve()!=LOCAL:raise ValueError('Exact local directory only')
    LOCAL.mkdir(exist_ok=True);(LOCAL/'video_files').mkdir(exist_ok=False)
    receipt=BASE/'input_local_overlay_receipt01.json'
    if receipt.exists():raise FileExistsError('No duplicate handover')
    path=BASE/'input_data/transfer_manifest512.json'
    if sha(path)!=MANIFEST_SHA:raise ValueError('Original transfer manifest required')
    manifest=json.loads(path.read_text());files=manifest['files']
    if len(files)!=512 or len({r['sample_id'] for r in files})!=512:raise ValueError('Exactly512 original videos')
    if shutil.disk_usage(LOCAL).free<sum(r['bytes'] for r in files)+2*1024**3:raise ValueError('Keep2GiB local safety margin')
    stopped=BASE/'repo/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/rental_timing_only_20261029_01'
    state=json.loads((stopped/'suite_status.json').read_text())
    if state['status']!='stopped' or state['completed'] or state.get('mps_released') is not True:raise ValueError('Stopped/released failed timing01 only')
    if any(Path('/proc/'+str(pid)).exists() for pid in (3442,3443,3462)):raise ValueError('No input handover while any worker is alive')
    backup=BASE/'input_symlink_handover01/old_links';backup.mkdir(parents=True,exist_ok=False)
    records=[]
    # Copy and independently verify every target BEFORE changing any input link.
    for i,r in enumerate(files):
        if time.time()>=a.deadline:raise TimeoutError('Finite local input copy')
        src=BASE/'input_data'/r['path'];dst=LOCAL/r['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        link=Path(r['original_path'])
        if not str(link).startswith('/DATA/DATA1/lixinyue/xyli/data/LGVQ/videos/') or not link.is_symlink() or link.resolve()!=src.resolve():raise ValueError('Only our exact original input link')
        if src.stat().st_size!=r['bytes'] or sha(src)!=r['sha256']:raise ValueError('Original source bytes differ')
        with src.open('rb') as f,dst.open('xb') as out:shutil.copyfileobj(f,out,4194304)
        if dst.stat().st_size!=r['bytes'] or sha(dst)!=r['sha256']:raise ValueError('Local byte copy differs')
        dst.chmod(0o444)  # provisioning, before any new timing window; no timestamp fabrication
        records.append(dict(sample_id=r['sample_id'],original_path=str(link),old_target=str(src),new_target=str(dst),
            old_link_backup=str(backup/f'{i:05d}.link'),bytes=r['bytes'],sha256=r['sha256'],local_metadata=metadata(dst)))
    control=decode_metadata_control(records[0]['new_target'])
    for r in records:
        if time.time()>=a.deadline:raise TimeoutError('Finite link handover')
        link=Path(r['original_path']);old=Path(r['old_link_backup'])
        if not link.is_symlink() or link.resolve()!=Path(r['old_target']).resolve():raise ValueError('Input link changed externally')
        link.rename(old);link.symlink_to(Path(r['new_target']))
        if link.resolve()!=Path(r['new_target']).resolve():raise ValueError('New link target differs')
    # A second actual read verifies metadata stability on this local filesystem;
    # this is input transport verification, not model inference or GPU warmup.
    for r in records:
        dst=Path(r['new_target']);before=metadata(dst)
        if sha(dst)!=r['sha256'] or metadata(dst)!=before:raise ValueError('Local byte or metadata stability check failed')
    result=dict(status='same_original512_bytes_verified_on_local_overlay',time=time.time(),
        transfer_manifest_sha256=MANIFEST_SHA,total_video_bytes=sum(r['bytes'] for r in records),
        local_directory=str(LOCAL),storage_change_disclosed=True,no_model_or_CUDA_import=True,
        original_shared_files_and_old_symlinks_preserved=True,all_original_metadata_gates_retained=True,
        input_transport_SHA_reads_precede_measurement=True,CPU_OpenCV_control=control,records=records)
    with receipt.open('x') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
