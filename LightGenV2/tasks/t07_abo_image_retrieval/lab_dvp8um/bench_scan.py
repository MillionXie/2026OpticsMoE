"""Bounded exposure scan for the 8 um Holoeye + DVP bench.

Saves linear raw statistics and canonical 478x478 previews.  It never performs
per-image contrast normalization and does not alter the trained model.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "ABO_I2I_Lab_DVP_8um" else HERE.parents[4]
OLD = ROOT / "ABO_Lab_SHS_8um"
ADAPTER = ROOT / "ABO_I2I_DVP_adapter_20260922"
sys.path[:0] = [str(OLD), str(ADAPTER)]
from lab_dvp import Camera  # noqa: E402
from phase_hdmi import PhaseHDMI  # noqa: E402
from vendor_driver import HoloeyeSLM  # noqa: E402

CAMERA_DLL = ROOT.parent / "小相机/SDK二次开发包/DVP2  SDK 中性版本/DVP2 SDK/library/Visual C++/bin/x64/DVPCamera64.dll"
PHASE_SDK = Path(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK")
PHASE_LUT = Path(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\19x12_8bit_linearVoltage.lut")
AMP_SDK = OLD / "vendor/holoeye_python"
AMP_BIN = Path(r"C:\Program Files\HOLOEYE SLM SDK SlideshowPlayer 2.0")
DEFAULT_CORNERS = np.float32([[900,142],[4310,142],[4297,3551],[874,3544]])  # TL,TR,BR,BL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(frame,corners):
    dst=np.float32([[0,0],[477,0],[477,477],[0,477]])
    matrix=cv2.getPerspectiveTransform(corners,dst)
    return cv2.warpPerspective(frame,matrix,(478,478),flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def amplitude(gray):
    a=np.zeros((1080,1920),np.uint8)
    size=1016;x=(1920-size)//2;y=(1080-size)//2
    a[y:y+size,x:x+size]=int(gray)
    return a


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exposures',type=float,nargs='+',default=[10,20,40,80,120,160,240,320,480,640,1000])
    p.add_argument('--grays',type=int,nargs='+',default=[0,128,255])
    p.add_argument('--corners-tltrbrbl',type=float,nargs=8,default=DEFAULT_CORNERS.reshape(-1).tolist())
    p.add_argument('--phase-bmp',type=Path)
    p.add_argument('--settle-ms',type=float,default=240)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    corners=np.float32(a.corners_tltrbrbl).reshape(4,2)
    if len(set(a.grays))!=len(a.grays) or any(not 0<=g<=255 for g in a.grays) or 0 not in a.grays or 255 not in a.grays:
        raise ValueError('Gray points must be distinct uint8 values including 0 and 255')
    if not 0<=a.settle_ms<=2000:raise ValueError('settle-ms out of bounded range')
    patterns=out/'patterns';previews=out/'previews';patterns.mkdir();previews.mkdir()
    amps={}
    for gray in a.grays:
        path=patterns/f'amplitude_{gray:03d}.bmp';Image.fromarray(amplitude(gray)).save(path);amps[gray]=path
    # raw=0 -> pi. Inverted hardware encoding therefore uses approximately127.
    phase_path=patterns/'phase_flat_pi_inverted.bmp'
    Image.fromarray(np.full((1200,1920),127,np.uint8)).save(phase_path)
    if a.phase_bmp is not None:phase_path=a.phase_bmp.resolve()
    exposures=[float(v) for v in a.exposures]
    rows=[]
    amp=HoloeyeSLM(AMP_SDK,AMP_BIN,(1920,1080),None,True,True,5)
    with PhaseHDMI(PHASE_SDK,PHASE_LUT,settle_s=.8,pixel_format='rgba') as phase, amp, Camera(CAMERA_DLL) as camera:
        phase_receipt=phase.show(phase_path)
        amp.preload_files(list(amps.values()))
        for exposure in exposures:
            actual=camera.settings(exposure=exposure,gain=1.0)
            for gray,path in amps.items():
                amp.display_file(path);time.sleep(a.settle_ms/1000)
                frames=[];metas=[]
                for _ in range(5):
                    frame,meta=camera.capture();frames.append(frame);metas.append(meta)
                frame=frames[-1];roi=canonical(frame,corners);v=roi.astype(np.float32)
                sat=float(np.mean(roi==np.iinfo(roi.dtype).max))
                row=dict(requested_exposure_us=exposure,actual_exposure_us=actual['exposure_us'],gain=actual['gain'],
                         gray=gray,mean=float(v.mean()),std=float(v.std()),p01=float(np.percentile(v,1)),
                         p50=float(np.percentile(v,50)),p99=float(np.percentile(v,99)),maximum=int(v.max()),
                         saturation_fraction=sat,frame_ids=[m['frame_id'] for m in metas],capture_ms=[m['capture_ms'] for m in metas])
                rows.append(row);print(json.dumps(row),flush=True)
                Image.fromarray(roi).save(previews/f'e{int(round(exposure)):05d}_g{gray:03d}.png')
        amp.display_file(amps[0])
    grouped={e:{r['gray']:r for r in rows if r['requested_exposure_us']==e} for e in exposures}
    valid=[]
    for e,g in grouped.items():
        high=g[255];low=g[0]
        if high['saturation_fraction']<=.001 and high['p99']<=250 and high['mean']>low['mean']+2:
            valid.append((e,high['mean']-low['mean']))
    selected=max(valid,key=lambda x:(x[0],x[1]))[0] if valid else min(exposures,key=lambda e:grouped[e][255]['saturation_fraction'])
    report=dict(schema=1,status='complete',selected_exposure_us=selected,selection_rule='highest tested exposure with gray255 saturation<=0.1%, p99<=250 and positive contrast; otherwise minimum saturation',
                corners_order='TL,TR,BR,BL',corners_full_sensor_xy=corners.tolist(),canonical_size_wh=[478,478],
                gray_points=a.grays,settle_ms=a.settle_ms,phase_file=str(phase_path.resolve()),
                no_per_image_normalization=True,phase_receipt=phase_receipt,
                amplitude_sha256={str(k):sha(v) for k,v in amps.items()},phase_sha256=sha(phase_path),rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
