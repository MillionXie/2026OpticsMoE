"""Holoeye + legacy DVP only. Raw files never get contrast/log processing."""
import contextlib
import time
import numpy as np
from PIL import Image
from common import ROOT,setup_imports,write,hardware_identity

def geometry(c,shape):
    import cv2
    names=('top_left','top_right','bottom_right','bottom_left')
    pts=[c['logical_corners_full_sensor_xy'][n] for n in names]
    if not c.get('geometry_confirmed') or any(p is None for p in pts):
        raise ValueError('Fill four LOGICAL full-sensor corners and set geometry_confirmed=true after asymmetric-pattern check.')
    points=np.asarray(pts,np.float32)
    roi=c['camera'].get('device_roi_xywh')
    if roi: points-=np.asarray(roi[:2],np.float32)
    h,w=shape
    if points.shape!=(4,2) or not np.isfinite(points).all() or (points<0).any() or (points[:,0]>=w).any() or (points[:,1]>=h).any():
        raise ValueError('Corners outside captured sensor region')
    cross=[]
    for i in range(4):
        u=points[(i+1)%4]-points[i]; v=points[(i+2)%4]-points[(i+1)%4]
        cross.append(u[0]*v[1]-u[1]*v[0])
    if not (all(v>1e-3 for v in cross) or all(v<-1e-3 for v in cross)):
        raise ValueError('Logical corner order is self-intersecting: identify TL,TR,BR,BL with one-corner patterns. Do not sort by camera coordinates.')
    target=np.array([[-.5,-.5],[477.5,-.5],[477.5,477.5],[-.5,477.5]],np.float32)
    return cv2.getPerspectiveTransform(points,target)

def canonical(raw,c):
    import cv2
    if raw.ndim!=2 or not np.isfinite(raw).all(): raise ValueError('Need monochrome finite raw frame')
    bounds=c.get('capture_input_range')
    if bounds is None: raise ValueError('Set capture_input_range after probe (usually [0,255] for Mono8; never infer sensor bit depth from brightness).')
    low,high=map(float,bounds)
    if not 0<=low<high: raise ValueError('Invalid sensor input range')
    H=geometry(c,raw.shape)
    rect=cv2.warpPerspective(raw.astype(np.float32),H,(478,478),flags=cv2.INTER_LINEAR)
    # Fixed LINEAR storage scale only; no min/max per image, gamma, log or CLAHE.
    return np.rint(np.clip((rect-low)/(high-low),0,1)*255).astype(np.uint8)

class Bench:
    def __init__(self,c,base=ROOT,slm=True):
        setup_imports()
        from experiments.hardware_sdk.devices import build_slm,build_camera
        self.c=c; self.slm=build_slm(c['amplitude_slm'],base) if slm else None
        self.camera=build_camera(c['camera'],base)
        self.stack=contextlib.ExitStack()
    def __enter__(self):
        try:
            if self.slm: self.stack.enter_context(self.slm)
            self.stack.enter_context(self.camera)
            self.info={'amplitude':self.slm.device_info() if self.slm else None,'camera':self.camera.device_info()}
            print(self.info,flush=True)
            return self
        except BaseException:
            self.stack.close(); raise
    def __exit__(self,*args): return self.stack.__exit__(*args)
    def capture(self,bmp,path,rectify=True):
        path=__import__('pathlib').Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        t0=time.perf_counter()
        if bmp is not None:
            # Preload exactly one; no thousands-of-BMP GPU preload on 3GB card.
            self.slm.preload_files([bmp]); self.slm.display_file(bmp)
            time.sleep(self.c['settle_delay_ms']/1000)
        t1=time.perf_counter()
        self.camera.capture(path.with_suffix('.npy'))
        raw=np.load(path.with_suffix('.npy'),allow_pickle=False)
        if raw.ndim!=2: raise ValueError('DVP must output MONO')
        # Lossless raw TIFF, not auto-enhanced. NPY retained for numerical audit.
        Image.fromarray(raw).save(path.with_suffix('.tif'))
        meta={'hardware_identity':hardware_identity(self.c),'raw_shape':list(raw.shape),'raw_dtype':str(raw.dtype),
              'min':float(raw.min()),'max':float(raw.max()),'mean':float(raw.mean()),
              'display_and_settle_ms':(t1-t0)*1000,'capture_ms':(time.perf_counter()-t1)*1000,'devices':self.info}
        if rectify:
            actual=self.camera.device_info().get('device_roi_xywh')
            if actual is None:
                actual=self.camera.device_info().get('camera',{}).get('device_roi_xywh')
            expected=self.c['camera'].get('device_roi_xywh')
            if actual is not None and expected is None and list(actual[:2])!=[0,0]:
                raise ValueError(f'Camera retained cropped ROI {actual}; set explicit device_roi_xywh, with four corners in full-sensor coordinates.')
            out=canonical(raw,self.c); Image.fromarray(out).save(path.with_suffix('.png'))
        else: out=raw
        write(path.with_suffix('.json'),meta)
        return out
