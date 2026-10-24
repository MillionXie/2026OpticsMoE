"""One seven-frame TRAIN/control recheck of the admitted MoE expert boundary.

Same published stage_signal_admission, thresholds, PT, hardware and input IDs.
No network frames, TEST inputs, exposure changes, automatic retry or fitting.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import torch

from .hardware_bridge import (hardware, load_checkpoint, read_source,
                              native_image, stage_signal_admission)
from .train_native import sha, put


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('hardware','package','source_capture','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads(a.hardware.read_text());package=json.loads(a.package.read_text())
    if sha(a.hardware)!=package['hardware_sha256']:
        raise ValueError('Changed hardware identity')
    source=json.loads((a.source_capture/'protocol.json').read_text())
    if source['package']!=package:
        raise ValueError('Different original capture package')
    previous=json.loads((a.source_capture/'first/TRAIN_signal_admission/report.json').read_text())
    if previous['passed']:
        raise ValueError('This is a single diagnosed failure recheck, not repeated admission sampling')
    spec=next(s for s in package['datasets'] if s['split']=='train')
    if len(spec['indices'])!=1000:
        raise ValueError('Original full TRAIN1000 identity required')
    data=read_source(spec['path'],spec['sha256'],'train',np.array([spec['indices'][0],spec['indices'][-1]]))
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    if model.architecture!='moe':raise ValueError('MoE only')
    examples=[];phase_image=None
    for j,index in enumerate((0,999)):
        folder=a.source_capture/'router'/f'{index:05d}'
        receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['split']!='train' or receipt['source_index']!=int(data['source_indices'][j]):
            raise ValueError('TRAIN source identity mismatch')
        if receipt['checkpoint_sha256']!=package['checkpoint_sha256'] or receipt['hardware_sha256']!=package['hardware_sha256']:
            raise ValueError('Different prefix checkpoint/hardware')
        for filename,key in (('raw.npy','raw_sha256'),('ccd.npy','ccd_sha256'),('input.bmp','input_BMP_sha256')):
            if sha(folder/filename)!=receipt[key]:raise ValueError('Prefix data SHA mismatch')
        if sha(a.source_capture/'router/phase.bmp')!=receipt['phase_BMP_sha256']:
            raise ValueError('Prefix phase SHA mismatch')
        prefix={'router':torch.from_numpy(np.load(folder/'ccd.npy'))[None]}
        amplitude,phase,_=model.payload(torch.from_numpy(data['x'][j])[None],'first',prefix)
        examples.append(np.rint(amplitude[0].numpy()*255).astype(np.uint8))
        candidate=native_image(np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8),True)
        if phase_image is not None and not np.array_equal(phase_image,candidate):raise ValueError('Input-dependent phase')
        phase_image=candidate
    Bench,LOCK,_,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK already owned')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(package=package,source_capture=str(a.source_capture),frames=7,
        single_recheck=True,automatic_retry=False,exposure_changed=False,TEST_used=False,network_frames=0,
        previous_repeat_correlation=previous['repeat_correlation'],correlation_threshold=.95))
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='one_bounded_expert_TRAIN_control_recheck'))
    try:
        with Bench(cfg,a.out) as bench:
            phasefile=a.out/'phase.bmp';Image.fromarray(phase_image).save(phasefile);bench.stage(phasefile)
            stage_signal_admission(bench,a.out/'control',examples,drive,warp,None,'first',
                                   previous['hardware_payload_sha256'],model)
        put(a.out/'completion.json',dict(passed=True,SDK_released=True,network_frames=0,TEST_used=False))
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK owner changed')
        LOCK.unlink()


if __name__=='__main__':main()
