"""Future Holoeye display -> settle -> fresh SHS frame. Camera-only tests need no SLM."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import time
from PIL import Image
import numpy as np
from sdk import Camera
from capture import snapshot,save_json,restore_settings

ROOT=Path(__file__).resolve().parent


def slm_driver(c):
    if not c['amplitude_slm'].get('connected'):raise RuntimeError('SLM not connected: set connected=true ONLY after connecting/checking display identity')
    path=ROOT/'vendor_driver.py'
    if not path.exists():path=ROOT.parent/'experiments/hardware_sdk/devices.py'
    spec=importlib.util.spec_from_file_location('shs_holoeye_driver',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.build_slm(c['amplitude_slm'],ROOT)


class Controller:
    def __init__(self,c):self.c=c;self.stack=contextlib.ExitStack()
    def __enter__(self):
        try:
            self.slm=self.stack.enter_context(slm_driver(self.c))
            self.camera=self.stack.enter_context(Camera(self.c['camera']))
            self.before=snapshot(self.camera)
            self.stack.callback(self.restore)
            self.camera.set('AcquisitionFrameRate',self.c['camera']['frame_rate_hz'])
            self.camera.set('ExposureTime',self.c['camera']['exposure_us'])
            if self.c['camera'].get('gain') is not None:self.camera.set('Gain',self.c['camera']['gain'])
            if self.camera.get('TestPattern')!='Normal' or self.camera.get('SyncMode')!='InternalSync':
                raise RuntimeError('Require real-image Normal + InternalSync for validated continuous fallback')
            if self.camera.get('OffsetX')!='0' or self.camera.get('OffsetY')!='0':raise RuntimeError('Current profile requires full-sensor ROI; reset offsets in Viewer first')
            self.camera.start()
            self.camera_settings=snapshot(self.camera)
            self.info={'amplitude':self.slm.device_info(),'camera':self.camera_settings}
            return self
        except BaseException:self.stack.close();raise
    def restore(self):
        errors=restore_settings(self.camera,self.before,['Gain','ExposureTime','AcquisitionFrameRate'])
        if errors:raise RuntimeError('Camera restore failed: '+str(errors))
    def __exit__(self,*args):return self.stack.__exit__(*args)
    def capture(self,bmp):
        begin=time.perf_counter()
        path=Path(bmp).resolve()
        with Image.open(path) as im:
            if im.format!='BMP' or im.mode!='L' or list(im.size)!=self.c['amplitude_slm']['expected_resolution_wh']:
                raise ValueError('Expected native-size 1920x1080 grayscale BMP')
        t0=time.perf_counter()
        self.slm.preload_files([path]);loaded=time.perf_counter()
        self.slm.display_file(path)
        visible=time.perf_counter()
        # Sleeping lets the camera/card queue accumulate old optical frames.
        # Fixed-count fresh() alone was experimentally shown to return the
        # previous pattern. Drain continuously throughout the settling window.
        drained=0
        while time.perf_counter()-visible<self.c['settle_delay_ms']/1000:
            self.camera.grab();drained+=1
        settled=time.perf_counter()
        frame,meta=self.camera.fresh()
        end=time.perf_counter()
        meta.update(amplitude_file=str(path),settle_delay_ms=self.c['settle_delay_ms'],
                    settle_method='continuous_drain_then_buffer_count_plus_2',settle_drained_frames=drained,
                    bmp_validate_ms=(t0-begin)*1000,slm_preload_ms=(loaded-t0)*1000,
                    slm_show_to_visible_ms=(visible-loaded)*1000,settle_actual_ms=(settled-visible)*1000,
                    final_fresh_ms=(end-settled)*1000,capture_total_ms=(end-begin)*1000,
                    display_visible_ms=(visible-t0)*1000,
                    visible_to_capture_ms=(end-visible)*1000,
                    camera=self.camera_settings,startup_warmup=self.camera.startup_warmup,
                    phase_control='manual, unchanged')
        return frame,meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json')
    p.add_argument('--bmp',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    if not c['amplitude_slm'].get('connected'):raise RuntimeError('No SLM currently connected; use capture.py for camera-only')
    a.out.mkdir(parents=True,exist_ok=False)
    with Controller(c) as hw:
        raw,meta=hw.capture(a.bmp);Image.fromarray(raw).save(a.out/'raw.png')
        save_json(a.out/'capture.json',meta)
    print(a.out.resolve())


if __name__=='__main__':main()
