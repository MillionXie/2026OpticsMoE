"""Five bounded TRAIN/control frames to diagnose sparse 16-bit router signal.

No change to exposure/coordinates/device LUT/PT. Dark subtraction is used ONLY
for this diagnostic, never for network frames. No test records or labels read.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import torch

from .hardware_bridge import hardware,native_image,crop_native,load_checkpoint,read_source,signal,hardware_identity
from .train_native import sha,put


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--hardware',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads(a.hardware.read_text());package=json.loads(a.package.read_text())
    if sha(a.hardware)!=package['hardware_sha256']:raise ValueError('Hardware identity changed')
    Bench,LOCK,_,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK busy')
    spec=next(s for s in package['datasets'] if s['split']=='train')
    data=read_source(spec['path'],spec['sha256'],'train',np.array([spec['indices'][0],spec['indices'][-1]]))
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    if model.architecture!='moe':raise ValueError('This bounded diagnostic is router only')
    ampA,phase,_=model.payload(torch.from_numpy(data['x'][0])[None],'router',{})
    ampB,_,_=model.payload(torch.from_numpy(data['x'][1])[None],'router',{})
    images={'dark':np.zeros((1016,1016),np.uint8),'A1':np.rint(ampA[0].numpy()*255).astype(np.uint8),
            'A2':np.rint(ampA[0].numpy()*255).astype(np.uint8),'B':np.rint(ampB[0].numpy()*255).astype(np.uint8),
            'A3':np.rint(ampA[0].numpy()*255).astype(np.uint8)}
    mask=np.zeros((1016,1016),bool)
    for i in model.active_indices[:int(model.active_count)].tolist():
        y,x=model.router_centers[i];y-=model.border;x-=model.border;h=model.router_side//2;mask[y-h:y+h,x-h:x+h]=True
    a.out.mkdir(parents=True,exist_ok=False);put(LOCK,dict(pid=os.getpid(),task='T21',scope='bounded_TRAIN_signal_diagnostic'))
    values={};rows={}
    try:
        with Bench(cfg,a.out) as bench:
            phasefile=a.out/'phase.bmp';Image.fromarray(native_image(np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8),True)).save(phasefile);bench.stage(phasefile)
            for name,gray in images.items():
                f=a.out/(name+'.bmp');Image.fromarray(native_image(drive[gray])).save(f)
                raw,metadata=bench.capture(f);np.save(a.out/(name+'_raw.npy'),raw)
                value=crop_native(raw,warp);np.save(a.out/(name+'_ccd.npy'),value);values[name]=value
                rows[name]=dict(full=signal(value),metadata=metadata,raw_sha256=sha(a.out/(name+'_raw.npy')),
                    roi_p99_counts=float(np.percentile(value[mask]*65535,99)))
        dark=values['dark']*65535;A1=values['A1']*65535;A2=values['A2']*65535;A3=values['A3']*65535;B=values['B']*65535
        noise=max(1.,float(np.std((A1-A2)[mask])/np.sqrt(2)))
        delta99=float(np.percentile((A1-dark)[mask],99));snr=delta99/noise
        correlation=float(np.corrcoef((A1-dark)[mask],(A3-dark)[mask])[0,1])
        q=[model.route_from_ccd(torch.from_numpy(values[n])[None])[0].numpy() for n in ('A1','A2','A3','B')]
        repeat_q_delta=float(max(np.abs(q[0]-v).sum() for v in q[1:3]))
        changed_input_rms=float(np.sqrt(np.mean((B-A1)[mask]**2)))
        passed=(snr>=10 and delta99>=64 and correlation>=.95 and repeat_q_delta<=.05
                and changed_input_rms>=3*noise and max(v['full']['saturation_fraction'] for v in rows.values())<=.15)
        receipt=dict(passed=bool(passed),hardware_payload_sha256=hardware_identity(cfg),checkpoint_sha256=package['checkpoint_sha256'],
            rows=rows,read_noise_upper_counts=noise,dark_global_p99_counts=float(np.percentile(dark,99)),
            dark_router_ROI_p99_counts=float(np.percentile(dark[mask],99)),signal_ROI_delta_p99_counts=delta99,
            SNR=snr,repeat_correlation=correlation,repeat_q_L1=repeat_q_delta,changed_input_RMS_counts=changed_input_rms,
            quantization='uint16 preserved as float32, NOT logical8bit',exposure_unchanged=True,
            coordinates_unchanged=True,LUT_unchanged=True,TEST_used=False,network_background_subtraction=False,
            SDK_released=True,patterns=5,scope='diagnostic only; not an accuracy or model-response parity result')
        put(a.out/'report.json',receipt)
        if not passed:raise RuntimeError('Bounded signal diagnostic failed; preserve evidence and stop')
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK owner changed')
        LOCK.unlink()


if __name__=='__main__':main()
