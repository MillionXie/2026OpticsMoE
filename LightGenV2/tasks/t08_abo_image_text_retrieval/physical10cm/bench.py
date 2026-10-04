"""T08 adopted DVP bench, extracted from the exact laboratory six-stage flow.

No torch/Qwen/model import. Hardware opens only when Bench is constructed and
entered explicitly; configure() validates external machine assets without SDK
calls. Historical numerical warp, frame draining and device ordering are kept.
"""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from LightGenV2.hardware_common.dvp_legacy import Camera
from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI
from experiments.hardware_sdk.devices import HoloeyeSLM

BASE_CORNERS = np.float32([[900,142],[4310,142],[4297,3551],[874,3544]])
CAMERA_DLL = PHASE_SDK = PHASE_LUT = AMP_SDK = AMP_BIN = None
_configured = False


def configure(config_path):
    """Require an explicit external JSON config; never infer a sibling project."""
    global CAMERA_DLL, PHASE_SDK, PHASE_LUT, AMP_SDK, AMP_BIN, _configured
    _configured = False
    config_path = Path(config_path).expanduser().resolve()
    values = json.loads(config_path.read_text(encoding='utf-8-sig'))
    required = {'camera_dll': 'file', 'phase_sdk': 'directory', 'phase_lut': 'file',
                'amplitude_sdk': 'directory', 'amplitude_bin': 'directory'}
    resolved = {}
    for name, kind in required.items():
        value = values.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Missing explicit machine path: ' + name)
        location = Path(value).expanduser()
        location = (config_path.parent / location).resolve() if not location.is_absolute() else location.resolve()
        if not (location.is_file() if kind == 'file' else location.is_dir()):
            raise FileNotFoundError(location)
        resolved[name] = location
    CAMERA_DLL, PHASE_SDK, PHASE_LUT = (resolved[k] for k in ('camera_dll', 'phase_sdk', 'phase_lut'))
    AMP_SDK, AMP_BIN = (resolved[k] for k in ('amplitude_sdk', 'amplitude_bin'))
    _configured = True
    return {name: str(value) for name, value in resolved.items()}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def pcc(a,b):
    x=np.asarray(a,dtype=np.float64).ravel();y=np.asarray(b,dtype=np.float64).ravel()
    x-=x.mean();y-=y.mean();d=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x.dot(y)/d) if d else 0.0


def camera_variants(a):
    return {'identity':a,'flip_h':np.fliplr(a),'flip_v':np.flipud(a),'rot180':np.rot90(a,2),
            'transpose':a.T,'rot90':np.rot90(a,1),'rot270':np.rot90(a,3),'anti_transpose':np.rot90(a.T,2)}


def warp(frame):
    dst=np.float32([[0,0],[477,0],[477,477],[0,477]])
    matrix=cv2.getPerspectiveTransform(BASE_CORNERS,dst)
    return cv2.warpPerspective(frame,matrix,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def orient(a,name):
    return camera_variants(a)[name].copy()


class Bench:
    def __init__(self,out,exposure,wait_ms,phase_paths):
        if not _configured:raise RuntimeError('Configure explicit machine SDK paths before constructing a bench')
        self.out=out;self.exposure=exposure;self.wait=wait_ms/1000;self.phase_paths=phase_paths
        self.amp=HoloeyeSLM(AMP_SDK,AMP_BIN,(1920,1080),None,True,True,5)
        self.phase=PhaseHDMI(PHASE_SDK,PHASE_LUT,settle_s=.8,pixel_format='rgba')
        self.camera=Camera(CAMERA_DLL);self.rows=[]
    def __enter__(self):
        self.phase.__enter__();self.amp.__enter__();self.camera.__enter__()
        self.settings=self.camera.settings(exposure=self.exposure,gain=1.0);return self
    def __exit__(self,*args):
        errors=[]
        for device in (self.camera,self.amp,self.phase):
            try:device.__exit__(*args)
            except Exception as e:errors.append(repr(e))
        if errors and args[0] is None:raise RuntimeError('; '.join(errors))
    def capture(self,stage,phase_path,amplitude_paths,ids,camera_orientation,save=True):
        receipt=self.phase.show(phase_path);self.amp.preload_files(amplitude_paths);values=[]
        folder=self.out/'ccd'/stage
        if save:folder.mkdir(parents=True,exist_ok=True)
        for sample_id,path in zip(ids,amplitude_paths):
            self.amp.display_file(path);time.sleep(self.wait)
            frames=[];metas=[]
            for _ in range(6):
                image,meta=self.camera.capture();frames.append(image);metas.append(meta)
            image=orient(warp(frames[-1]),camera_orientation);values.append(image)
            row={'stage':stage,'sample_id':sample_id,'phase_sha256':sha(phase_path),'amplitude_sha256':sha(path),
                 'exposure':self.settings,'wait_ms':self.wait*1000,'frame_ids':[m['frame_id'] for m in metas],
                 'mean':float(image.mean()),'p99':float(np.percentile(image,99)),'maximum':int(image.max()),
                 'saturation_fraction':float(np.mean(image==255)),'canonical_orientation':camera_orientation,
                 'no_photometric_normalization':True}
            self.rows.append(row);print(json.dumps(row),flush=True)
            if row['saturation_fraction'] > .01:
                raise RuntimeError(f"Capture saturated at {stage}/{sample_id}: {row['saturation_fraction']:.4%}")
            if save:
                Image.fromarray(image).save(folder/(sample_id+'.png'));write(folder/(sample_id+'.json'),row)
        return np.stack(values),receipt
