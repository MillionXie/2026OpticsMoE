"""T21 native1016 intensity boundary and lossless BMP deployment.

Never changes coordinates, device LUT, exposure or model structure. Admission
uses calibration patterns only. Pilot uses deterministic TRAIN only. A phase
captures all optical layers before any terminal readout fitting is permitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from .model import build_model
from .train_native import sha,put,tensor_sha


def hardware_identity(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def legacy_session():
    # Published, read-only device session; old experiment runs are not restarted.
    previous=Path(__file__).parent.parent/'t18_optical_residual_ablation'
    if str(previous) not in sys.path:sys.path.insert(0,str(previous))
    from lab_caltech20_tucam_session_20261022 import Bench,LOCK,spot_centroid
    from lab_caltech10_20261015 import corners
    return Bench,LOCK,spot_centroid,corners


def native_image(gray,phase=False):
    gray=np.asarray(gray)
    if gray.shape!=(1016,1016) or gray.dtype!=np.uint8:raise ValueError('Native uint8 only; no implicit resize')
    height=1200 if phase else 1080
    panel=np.zeros((height,1920),np.uint8)
    value=np.flipud(np.fliplr(gray)) if phase else gray
    panel[(height-1016)//2:(height+1016)//2,452:1468]=value
    return 255-panel if phase else panel


def crop_native(raw,warp):
    if raw.shape!=(2048,2048) or raw.dtype!=np.uint16:raise ValueError('Changed sensor contract')
    # One intensity warp, fixed full scale; no uint8 CCD conversion or peak fit.
    value=cv2.warpPerspective(raw.astype(np.float32),warp,(1016,1016),
        flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)/65535.
    if not np.isfinite(value).all() or (value<0).any():raise ValueError('Invalid native CCD')
    return value


def hardware(cfg):
    Bench,LOCK,centroid,corners=legacy_session()
    for key,expected in (('coordinates','coordinates_SHA256'),('phase_lut','phase_LUT_SHA256'),
                         ('response_lut','response_lut_SHA256')):
        if sha(cfg[key])!=cfg[expected]:raise ValueError('Changed hardware identity: '+key)
    if cfg['orientation']!='identity' or cfg['phase_transform']!='hv_inverse':
        raise ValueError('Only independently admitted identity CCD / hv_inverse phase supported')
    response=json.loads(Path(cfg['response_lut']).read_text());drive=np.asarray(response['drive'],np.uint8)
    if not response['passed'] or drive.shape!=(256,) or drive[0]!=0:
        raise ValueError('Unadmitted amplitude response inverse')
    # Coordinates describe aperture boundaries, not edge-pixel centers.
    points=np.float32(corners(Path(cfg['coordinates'])))
    warp=cv2.getPerspectiveTransform(points,np.float32([[-.5,-.5],[1015.5,-.5],[1015.5,1015.5],[-.5,1015.5]]))
    return Bench,LOCK,centroid,drive,warp


def signal(value):
    return dict(mean=float(value.mean()),p99_gray=float(np.percentile(value,99)*255),
                maximum=float(value.max()),saturation_fraction=float((value>=1).mean()))


def guard(row):
    if row['p99_gray']<15 or row['saturation_fraction']>.15:
        raise RuntimeError('Pilot/capture dynamic-range stop; preserve evidence, no automatic exposure/retry: '+repr(row))


def native_admission(cfg,out):
    Bench,LOCK,centroid,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK owned by another session')
    out.mkdir(parents=True,exist_ok=False);put(LOCK,dict(pid=os.getpid(),task='T21',scope='native_admission',out=str(out)))
    rows=[];positions=((180,180),(800,180),(180,800),(800,800),(510,510),(180,180))
    try:
        with Bench(cfg,out) as bench:
            phase=out/'zero_phase.bmp';Image.fromarray(native_image(np.zeros((1016,1016),np.uint8),True)).save(phase);bench.stage(phase)
            def snap(name,gray):
                p=out/(name+'.bmp');Image.fromarray(native_image(drive[gray])).save(p)
                raw,metadata=bench.capture(p);np.save(out/(name+'_raw.npy'),raw)
                value=crop_native(raw,warp);np.save(out/(name+'_ccd.npy'),value)
                return value,dict(metadata=metadata,raw_sha256=sha(out/(name+'_raw.npy')),signal=signal(value))
            dark,bg=snap('background',np.zeros((1016,1016),np.uint8))
            # A fixed scale is used only for centroid diagnostics, never network frames.
            for k,(x,y) in enumerate(positions):
                gray=np.zeros((1016,1016),np.uint8);gray[y-35:y+35,x-35:x+35]=255
                value,row=snap('spot_'+str(k),gray)
                point=centroid(value*255,dark*255);error=float(np.linalg.norm(point-[x-.5,y-.5]))
                rows.append(dict(expected=[x-.5,y-.5],observed=point.tolist(),error_native_px=error,**row))
                put(out/'progress.json',dict(patterns=len(rows),latest=rows[-1]))
            repeat=float(np.linalg.norm(np.array(rows[0]['observed'])-rows[-1]['observed']))
            passed=max(r['error_native_px'] for r in rows)<=17 and repeat<=8
            put(out/'report.json',dict(passed=passed,rows=rows,background=bg,repeat_error_native_px=repeat,
                active_pixels=1016,sensor_full_scale=65535,network_background_subtraction=False,
                TEST_used=False,fit_points_reused=False,warp=warp.tolist(),SDK_released_after_return=True,
                hardware_payload_sha256=hardware_identity(cfg)))
            if not passed:raise RuntimeError('Independent native1016 spatial/temporal admission failed')
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK ownership changed')
        LOCK.unlink()


def load_checkpoint(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('PT SHA mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True)
    model.configure_stage(checkpoint['stage']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,checkpoint


def read_source(path,expected_sha,split,indices):
    if sha(path)!=expected_sha:raise ValueError('Source encoded input SHA mismatch')
    manifest=json.loads(Path(path).with_name('manifest.json').read_text())
    if manifest['split']!=split or manifest['inputs_sha256']!=expected_sha:raise ValueError('Changed split identity')
    with np.load(path,allow_pickle=False) as a:
        return dict(x=a['amplitudes'][indices].copy(),y=a['labels'][indices].copy(),
                    source_indices=a['source_indices'][indices].copy(),manifest=manifest)


@torch.no_grad()
def capture(package,cfg,out,pilot=False):
    Bench,LOCK,_,drive,warp=hardware(cfg)
    admission=json.loads(Path(package['native_admission']).read_text())
    if not admission['passed'] or sha(package['native_admission'])!=package['native_admission_sha256']:
        raise ValueError('Native admission missing or changed')
    if admission['hardware_payload_sha256']!=hardware_identity(cfg):raise ValueError('Admission uses another hardware session')
    if LOCK.exists():raise RuntimeError('SDK already owned')
    model,checkpoint=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.json',dict(package=package,hardware=cfg,hardware_sha256=package['hardware_sha256'],
        pilot=pilot,source_sha256=sha(__file__),interlayer_tuning=False,TEST_gradients=False))
    entries=[]
    for spec in package['datasets']:
        if pilot and spec['split']!='train':continue
        ix=np.asarray(spec['indices'],int)
        if pilot:ix=ix[np.linspace(0,len(ix)-1,min(24,len(ix)),dtype=int)]
        data=read_source(spec['path'],spec['sha256'],spec['split'],ix)
        for k in range(len(ix)):
            entries.append(dict(task=spec['task'],split=spec['split'],local_index=int(ix[k]),
                source_index=int(data['source_indices'][k]),label=int(data['y'][k]),amplitude=data['x'][k],
                manifest_sha256=data['manifest']['identity_sha256']))
    order=model.STAGES if model.architecture=='moe' else model.STAGES[1:]
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='pilot' if pilot else 'capture',out=str(out)))
    started=time.time();rows=[];frozen={k:tensor_sha(v) for k,v in model.state_dict().items()}
    try:
        with Bench(cfg,out) as bench:
            for stage in order:
                layer=out/stage;layer.mkdir();phase_file=layer/'phase.bmp';phase_sha=None
                for i,entry in enumerate(entries):
                    key=f'{i:05d}';folder=layer/key;folder.mkdir()
                    prefix={s:torch.from_numpy(np.load(out/s/key/'ccd.npy'))[None] for s in order[:order.index(stage)]}
                    x=torch.from_numpy(entry['amplitude'])[None]
                    amplitude,phase,_=model.payload(x,stage,prefix)
                    gray=np.rint(amplitude[0].numpy()*255).astype(np.uint8)
                    phase_gray=np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8)
                    candidate=native_image(phase_gray,True)
                    digest=hashlib.sha256(candidate.tobytes()).hexdigest()
                    if phase_sha is None:
                        Image.fromarray(candidate).save(phase_file);bench.stage(phase_file);phase_sha=digest
                    elif digest!=phase_sha:raise ValueError('Input-dependent phase violates frozen layer contract')
                    amplitude_file=folder/'input.bmp';Image.fromarray(native_image(drive[gray])).save(amplitude_file)
                    raw,metadata=bench.capture(amplitude_file);np.save(folder/'raw.npy',raw)
                    value=crop_native(raw,warp);np.save(folder/'ccd.npy',value);quality=signal(value)
                    row={k:v for k,v in entry.items() if k!='amplitude'}
                    row.update(index=i,stage=stage,checkpoint_sha256=package['checkpoint_sha256'],
                        input_canonical_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
                        input_BMP_sha256=sha(amplitude_file),phase_BMP_sha256=sha(phase_file),
                        prefix_CCD_sha256={s:sha(out/s/key/'ccd.npy') for s in prefix},
                        raw_sha256=sha(folder/'raw.npy'),ccd_sha256=sha(folder/'ccd.npy'),
                        signal=quality,metadata=metadata,hardware_sha256=package['hardware_sha256'])
                    put(folder/'receipt.json',row);rows.append(row)
                    put(out/'progress.json',dict(stage=stage,stage_frames=i+1,stage_total=len(entries),
                        total_frames=len(rows),expected_frames=len(entries)*len(order),latest=row,updated=time.time()))
                    guard(quality)
            features=[]
            for i in range(len(entries)):
                value=torch.from_numpy(np.load(out/'final'/f'{i:05d}'/'ccd.npy'))[None]
                features.append(model.features_from_ccd(value)[0].numpy())
            np.savez_compressed(out/'features.npz',features=np.stack(features),labels=[r['label'] for r in entries],
                tasks=[r['task'] for r in entries],splits=[r['split'] for r in entries],source_indices=[r['source_index'] for r in entries])
        if {k:tensor_sha(v) for k,v in model.state_dict().items()}!=frozen:raise RuntimeError('Frozen deployed PT changed')
        put(out/'report.json',dict(passed=True,pilot=pilot,records=len(entries),frames=len(rows),
            stages=list(order),features_sha256=sha(out/'features.npz'),checkpoint_sha256=package['checkpoint_sha256'],
            seconds=time.time()-started,seconds_per_frame=(time.time()-started)/len(rows),
            TRAIN_records=sum(r['split']=='train' for r in entries),EVAL_records=sum(r['split']=='test' for r in entries),
            SDK_released=True,interlayer_training=False,full_physical_prefix=True))
    except BaseException:
        put(out/'failure.json',dict(traceback=traceback.format_exc(),valid_frames=len(rows),
            automatic_retry=False,all_valid_and_bad_evidence_preserved=True));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK ownership changed')
        LOCK.unlink()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=('admit','pilot','capture'),required=True)
    p.add_argument('--hardware',type=Path,required=True);p.add_argument('--package',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);cfg=json.loads(a.hardware.read_text())
    if a.mode=='admit':native_admission(cfg,a.out)
    else:
        package=json.loads(a.package.read_text())
        if sha(a.hardware)!=package['hardware_sha256']:raise ValueError('Hardware JSON SHA mismatch')
        capture(package,cfg,a.out,pilot=a.mode=='pilot')


if __name__=='__main__':main()
