"""Read-only audit of complete T20 raw/CCD/BMP/ID/phase and cached features."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .build_lab_package import PINS,load_locked,sha,put
from .lab_capture import read_entries,router_mask
from LightGenV2.tasks.t21_continual_learning_hardware.hardware_bridge import hardware,native_image,crop_native,signal,guard
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import tensor_sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('package','capture','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    report=json.loads((a.capture/'report.json').read_text());protocol=json.loads((a.capture/'protocol.json').read_text())
    manifest=json.loads((a.package/'manifest.json').read_text());case=report['case']
    if not report['passed'] or sha(a.package/'manifest.json')!=protocol['package_sha256']:
        raise ValueError('Only a complete immutable same-package capture may be audited')
    if sha(a.capture/'features.npz')!=report['features_sha256']:raise ValueError('Changed measured features')
    model,cp=load_locked(case,a.package/manifest['cases'][case]['checkpoint'])
    frozen={n:tensor_sha(v) for n,v in model.state_dict().items()}
    entries=read_entries(a.package,manifest,case,report['pilot']);order=manifest['cases'][case]['stages']
    if report['records']!=len(entries) or report['frames']!=len(entries)*len(order):raise ValueError('Missing records/stages')
    _,_,_,drive,warp=hardware(protocol['hardware']) # creates no camera/SLM instance
    a.out.mkdir(parents=True,exist_ok=False);began=time.monotonic();rows=[];features=[]
    for stage in order:
        layer=a.capture/stage;phase_file=layer/'phase.bmp';phase_sha=sha(phase_file)
        noise=json.loads((layer/'TRAIN_signal_admission/report.json').read_text())
        if not noise['passed'] or noise['TEST_used']:raise ValueError('Invalid TRAIN signal admission')
        dark=np.load(layer/'TRAIN_signal_admission/dark_ccd.npy')
        if sha(layer/'TRAIN_signal_admission/dark_ccd.npy')!=noise['dark_CCD_sha256']:raise ValueError('Fixed control dark changed')
        mask=router_mask(model) if stage=='router' else None
        for i,entry in enumerate(entries):
            folder=layer/f'{i:05d}';r=json.loads((folder/'receipt.json').read_text())
            if not r['passed']:raise ValueError('Failed evidence is not a valid network frame')
            for field in ('split','local_index','source_index','source_id','domain','label','source_identity_sha256'):
                if r[field]!=entry[field]:raise ValueError('Captured source identity mismatch: '+field)
            if r['checkpoint_sha256']!=PINS[case]['sha256'] or r['hardware_payload_sha256']!=report['hardware_payload_sha256']:
                raise ValueError('Wrong upstream or camera identity')
            for name,key in (('raw.npy','raw_sha256'),('ccd.npy','ccd_sha256'),('input.bmp','input_BMP_sha256')):
                if sha(folder/name)!=r[key]:raise ValueError('Changed raw/CCD/BMP file')
            if r['phase_BMP_sha256']!=phase_sha:raise ValueError('Changed phase file')
            prefix={s:torch.from_numpy(np.load(a.capture/s/f'{i:05d}'/'ccd.npy'))[None] for s in order[:order.index(stage)]}
            for s in prefix:
                if sha(a.capture/s/f'{i:05d}'/'ccd.npy')!=r['prefix_CCD_sha256'][s]:raise ValueError('Mixed optical prefix')
            amp,phase,_=model.payload(torch.from_numpy(entry['amplitude'])[None],stage,prefix)
            gray=np.rint(amp[0].numpy()*255).clip(0,255).astype(np.uint8)
            if hashlib.sha256(gray.tobytes()).hexdigest()!=r['input_canonical_sha256']:raise ValueError('Wrong bounded amplitude')
            np.testing.assert_array_equal(np.asarray(Image.open(folder/'input.bmp')),native_image(drive[gray]))
            phase_gray=np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8)
            np.testing.assert_array_equal(np.asarray(Image.open(phase_file)),native_image(phase_gray,True))
            raw=np.load(folder/'raw.npy');ccd=np.load(folder/'ccd.npy')
            np.testing.assert_array_equal(crop_native(raw,warp),ccd)
            guard(signal(ccd,mask,dark),noise,router=stage=='router',policy='pointwise_dark_p99')
            metadata=r['metadata']
            if (abs(metadata['exposure_us']-protocol['hardware']['exposure_us'])>.5 or metadata['gain']!=0 or
                metadata['wait_ms']!=240 or metadata['discard_frames']!=6 or metadata['auto_exposure'] is not False):
                raise ValueError('Changed camera exposure/gain/timing')
            if stage=='final':features.append(model.features_from_ccd(torch.from_numpy(ccd)[None])[0].numpy())
            rows.append(dict(stage=stage,index=i,source_id=entry['source_id'],split=entry['split'],
                raw_sha256=r['raw_sha256'],ccd_sha256=r['ccd_sha256'],input_BMP_sha256=r['input_BMP_sha256'],phase_BMP_sha256=phase_sha))
            if (i+1)%100==0:put(a.out/'progress.json',dict(stage=stage,stage_rows=i+1,verified_frames=len(rows),updated=time.time()))
    with np.load(a.capture/'features.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(np.stack(features),saved['features'])
        for key,field in (('ids','source_id'),('labels','label'),('domains','domain'),('splits','split'),('source_indices','source_index')):
            if saved[key].tolist()!=[e[field] for e in entries]:raise ValueError('Feature cache source order mismatch')
    if frozen!={n:tensor_sha(v) for n,v in model.state_dict().items()}:raise ValueError('Audit mutated the model')
    put(a.out/'frames_manifest.json',rows)
    put(a.out/'report.json',dict(passed=True,pilot=report['pilot'],case=case,verified_frames=len(rows),
        records=len(entries),stages=order,TRAIN=sum(e['split']=='train' for e in entries),TEST=sum(e['split']=='test' for e in entries),
        checkpoint_sha256=PINS[case]['sha256'],features_sha256=report['features_sha256'],capture_report_sha256=sha(a.capture/'report.json'),
        frame_manifest_sha256=sha(a.out/'frames_manifest.json'),source_sha256=sha(__file__),SDK=False,GPU=False,
        all_raw_CCD_BMP_phase_prefix_ID_and_timing_checked=True,original_CPU_feature_cache_bitwise=True,seconds=time.monotonic()-began))


if __name__=='__main__':main()
