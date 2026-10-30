"""Bounded non-model phase-depth diagnostic using an already admitted camera map."""
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
from .scene_lab_hardware import Bench
from .scene_lab_protocol import sha,put


def main():
    import cv2
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--aperture',choices=('gaussian4','square44'),default='gaussian4');p.add_argument('--cycles',type=int,default=64);a=p.parse_args()
    if a.cycles not in (64,128):raise ValueError('Bounded predeclared carrier frequencies only')
    cfg=json.loads(a.config.read_text());admission=json.loads((a.calibration/'report.json').read_text())
    if not admission['passed'] or admission['orientation']!='identity':raise ValueError('Admitted identity camera map required')
    mapping=json.loads(Path(cfg['derived_mapping']).read_text());assert sha(cfg['derived_mapping'])==cfg['derived_mapping_sha256']
    warp=np.asarray(mapping['canonical_warp'],np.float64);inverse=json.loads((a.calibration/'amplitude_inverse.json').read_text())['drive']
    root=Path(__file__).resolve().parents[3];lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Other device owner')
    a.out.mkdir(parents=True,exist_ok=False);fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out))).encode());os.close(fd)
    yy,xx=np.indices((478,478));zero=np.zeros((478,478),np.uint8)
    # Narrow in X resolves diffraction orders. The measured fixed inverse is
    # applied once, to amplitude drive only; no model/sample-wise normalization.
    target=np.exp(-.5*((xx-239)/4.)**2)*(np.abs(yy-239)<22)
    if a.aperture=='square44':target=((np.abs(xx-239)<22)&(np.abs(yy-239)<22)).astype(float)
    code=np.rint(target*255).astype(np.uint8);drive=np.asarray(inverse,np.uint8)[code]
    patterns=[('dark',zero,zero),('white',np.full_like(zero,255),zero),('flat',drive,zero)]
    for depth in (.25,.5,.75,1.):
        for sign in (1,-1):
            phase=np.floor(np.remainder(sign*xx*a.cycles/478,1)*depth*256).clip(0,255).astype(np.uint8)
            patterns.append((f'depth{depth:g}_{sign:+d}',drive,phase))
    put(a.out/'protocol.json',dict(config=cfg,calibration_report_sha256=sha(a.calibration/'report.json'),inverse_sha256=sha(a.calibration/'amplitude_inverse.json'),
        cycles=a.cycles,aperture=a.aperture,phase_depth_fractions=[.25,.5,.75,1.],heldout_prior_32cycle_patterns_not_fitted=True,model_inputs_used=False,TEST_used=False,physical_contract_changed=False,source_sha256=sha(__file__)))
    rows=[]
    try:
        with Bench(cfg,a.out)as bench:
            background=None
            for tag,amp,phase in patterns:
                raw,receipt=bench.capture(tag,amp,phase)
                value=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                np.save(a.out/tag/'canonical.npy',value,allow_pickle=False)
                if background is None:background=value
                signal=(value-background).clip(0);profile=signal[195:284].sum(axis=0)
                row=dict(tag=tag,raw_sha256=receipt['raw_file_sha256'],canonical_sha256=sha(a.out/tag/'canonical.npy'),profile=profile.tolist(),saturation_fraction=float((value>=65534).mean()))
                rows.append(row);print(json.dumps(dict(tag=tag,peak_x=int(profile.argmax()),signal_peak=float(signal.max()),saturation_fraction=row['saturation_fraction'])),flush=True)
                if row['saturation_fraction']>.001:raise ValueError('Phase probe saturated; no automatic exposure edit')
                if tag in ('white','flat') and signal.max()<500:raise ValueError('Low signal in phase canary; no phase-depth inference or blind retry')
        put(a.out/'report.json',dict(complete=True,rows=rows,native_frames=len(rows),SDK_released=True,model_TRAIN_CCD=0,TEST_used=False,no_distance_or_LUT_fit=True))
    except BaseException as error:
        put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,frames=len(rows)));raise
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(own_lock_released=True,closed_context=True,pid=os.getpid()))


if __name__=='__main__':main()
