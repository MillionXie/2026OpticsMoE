"""Four fixed TRAIN payloads, three repeats each, two darks; no new network run.

Replays exact saved BMP and phase, without recoding, exposure/geometry/LUT/PT
changes. Quantifies old-to-current drift against contemporaneous repetitions.
No correction is applied to the existing network or reported accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from .hardware_bridge import hardware,crop_native,native_image,load_checkpoint
from .train_native import put,sha


def drift_summary(old,repeats):
    values=np.stack(repeats).astype(np.float64);old=np.asarray(old,np.float64)
    center=values.mean(0);size=max(1e-20,float(np.linalg.norm(old)))
    repeat=max(1e-20,float(np.sqrt(np.mean(np.sum((values-center)**2,axis=1)))))
    return dict(old_to_new_relative_L2=float(np.linalg.norm(center-old)/size),
        current_repeat_relative_RMS=float(repeat/size),old_to_new_over_current_repeat=float(np.linalg.norm(center-old)/repeat))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('hardware','capture','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.capture/'protocol.json').read_text());report=json.loads((a.capture/'report.json').read_text())
    package=protocol['package'];cfg=json.loads(a.hardware.read_text())
    if not report['passed'] or report['pilot'] or sha(a.hardware)!=package['hardware_sha256']:raise ValueError('Completed exact hardware family required')
    if cfg!=protocol['hardware']:raise ValueError('Hardware configuration changed')
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    layer=a.capture/'final';indices=[0,250,500,750];sources=[]
    for i in indices:
        f=layer/f'{i:05d}';receipt=json.loads((f/'receipt.json').read_text())
        if receipt['split']!='train':raise ValueError('TRAIN payloads only')
        for name,key in (('ccd.npy','ccd_sha256'),('raw.npy','raw_sha256'),('input.bmp','input_BMP_sha256')):
            if sha(f/name)!=receipt[key]:raise ValueError('Saved TRAIN artifact changed')
        if sha(layer/'phase.bmp')!=receipt['phase_BMP_sha256']:raise ValueError('Phase changed')
        sources.append((i,f,receipt))
    Bench,LOCK,_,_,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK owned')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(source_capture=str(a.capture),hardware_SHA256=sha(a.hardware),
        checkpoint_SHA256=package['checkpoint_sha256'],stage='final',TRAIN_indices=indices,
        repeats=3,maximum_frames=14,TEST_used=False,labels_for_fitting=False,
        exposure_coordinates_LUT_PT_unchanged=True,network_background_subtraction=False,
        exact_saved_input_BMP_replay=True,source_SHA256=sha(__file__)))
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='bounded_saved_TRAIN_temporal_recheck'))
    rows=[]
    try:
        with Bench(cfg,a.out) as bench:
            bench.stage(layer/'phase.bmp')
            dark=a.out/'dark.bmp';Image.fromarray(native_image(np.zeros((1016,1016),np.uint8))).save(dark)
            raw,meta=bench.capture(dark);np.save(a.out/'dark_before_raw.npy',raw);put(a.out/'dark_before_metadata.json',meta)
            for i,folder,receipt in sources:
                current=[];measurements=[]
                for repeat in range(3):
                    raw,meta=bench.capture(folder/'input.bmp');name=f'TRAIN_{i:04d}_repeat{repeat}'
                    np.save(a.out/(name+'_raw.npy'),raw);value=crop_native(raw,warp);np.save(a.out/(name+'_ccd.npy'),value)
                    current.append(model.features_from_ccd(torch.from_numpy(value)[None])[0].numpy())
                    measurements.append(dict(metadata=meta,raw_SHA256=sha(a.out/(name+'_raw.npy')),ccd_SHA256=sha(a.out/(name+'_ccd.npy'))))
                old=model.features_from_ccd(torch.from_numpy(np.load(folder/'ccd.npy'))[None])[0].numpy()
                rows.append(dict(TRAIN_index=i,source_index=receipt['source_index'],source_input_SHA256=receipt['input_BMP_sha256'],
                    original_frame_age_seconds=time.time()-(folder/'ccd.npy').stat().st_mtime,
                    feature_drift=drift_summary(old,current),measurements=measurements))
                put(a.out/'progress.json',dict(completed=len(rows),rows=rows,updated=time.time()))
            raw,meta=bench.capture(dark);np.save(a.out/'dark_after_raw.npy',raw);put(a.out/'dark_after_metadata.json',meta)
        put(a.out/'report.json',dict(completed=True,frames=14,rows=rows,SDK_released=True,TEST_used=False,
            no_network_result_changed=True,no_automatic_recapture=True,exposure_coordinates_LUT_PT_unchanged=True))
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK owner changed')
        LOCK.unlink()


if __name__=='__main__':main()
