"""SHS adapter for the unchanged six-stage ABO-I2I model.

Continuous camera draining is required during SLM settling. Never reuse DVP
run directories, frame queues, exposure units, or sensor coordinates.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import four_image_flow as flow
import full_query_flow as pipeline
from sdk import Camera as SHSCamera
from capture import snapshot, restore_settings

CORNERS = np.float32([[586,147],[1379,159],[1369,946],[573,932]])
CAMERA_CONFIG = json.loads((flow.OLD / 'LAB.local.json').read_text(encoding='utf-8-sig'))['camera']


class SHSBench(flow.Bench):
    def __init__(self, out, exposure, wait_ms, phase_paths):
        super().__init__(out, exposure, wait_ms, phase_paths)
        self.camera = SHSCamera(CAMERA_CONFIG)

    def __enter__(self):
        try:
            self.phase.__enter__()
            self.amp.__enter__()
            self.camera.__enter__()
            self.before = snapshot(self.camera)
            self.camera.set('AcquisitionFrameRate',100)
            self.camera.set('ExposureTime',self.exposure)
            self.camera.set('Gain','Gain_X1')
            if self.camera.get('TestPattern') != 'Normal' or self.camera.get('SyncMode') != 'InternalSync':
                raise RuntimeError('Require Normal real image and InternalSync')
            if self.camera.get('OffsetX') != '0' or self.camera.get('OffsetY') != '0':
                raise RuntimeError('Require full-sensor zero offsets')
            self.camera.start()
            self.settings = {'exposure_us':float(self.camera.get('ExposureTime')),
                             'gain':self.camera.get('Gain'), 'frame_rate_hz':self.camera.get('AcquisitionFrameRate'),
                             'snapshot':snapshot(self.camera)}
            print(json.dumps({'camera_connected':self.settings}),flush=True)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self,*args):
        errors=[]
        if hasattr(self,'before'):
            errors.extend(restore_settings(self.camera,self.before,['Gain','ExposureTime','AcquisitionFrameRate']))
        for device in (self.camera,self.amp,self.phase):
            try: device.__exit__(*args)
            except Exception as ex: errors.append(str(ex))
        if errors and args[0] is None: raise RuntimeError('; '.join(errors))

    def capture(self,stage,phase_path,amplitude_paths,ids,camera_orientation,save=True):
        receipt=self.phase.show(phase_path)
        self.amp.preload_files(amplitude_paths)
        values=[]
        folder=self.out/'ccd'/stage
        if save: folder.mkdir(parents=True,exist_ok=True)
        for sample_id,path in zip(ids,amplitude_paths):
            start=time.perf_counter()
            self.amp.display_file(path)
            visible=time.perf_counter()
            drained=0
            while time.perf_counter()-visible<self.wait:
                self.camera.grab();drained+=1
            raw,meta=self.camera.fresh()
            if raw.shape != (1080,1920): raise RuntimeError(f'Unexpected SHS frame shape: {raw.shape}')
            image=flow.orient(flow.warp(raw),camera_orientation)
            row={'stage':stage,'sample_id':sample_id,'phase_sha256':flow.sha(phase_path),
                 'amplitude_sha256':flow.sha(path),'exposure':self.settings,'wait_ms':self.wait*1000,
                 'frame_id':meta['frame_id'],'timestamp_ns':meta.get('timestamp_ns'),
                 'settle_drained_frames':drained,'capture_total_ms':(time.perf_counter()-start)*1000,
                 'mean':float(image.mean()),'p99':float(np.percentile(image,99)),
                 'maximum':int(image.max()),'saturation_fraction':float(np.mean(image==255)),
                 'canonical_orientation':camera_orientation,'no_photometric_normalization':True}
            self.rows.append(row)
            print(json.dumps(row),flush=True)
            if row['saturation_fraction'] > .01: raise RuntimeError('SHS capture saturated')
            if save:
                Image.fromarray(image).save(folder/(sample_id+'.png'))
                flow.write(folder/(sample_id+'.json'),row)
            values.append(image)
        return np.stack(values),receipt


def health(out):
    out.mkdir(parents=True,exist_ok=False)
    patterns=flow.PROJECT/'runs/abo_i2i_20260924/05_brightness_health/patterns'
    flat=patterns/'phase_flat_pi_inverted.bmp'
    rows=[]
    with SHSBench(out,150,240,{}) as bench:
        for exposure in (100,150,300,400):
            bench.camera.set('ExposureTime',exposure)
            bench.settings['exposure_us']=float(bench.camera.get('ExposureTime'))
            for gray in (0,64,128,192,255):
                path=out/f'amplitude_{gray}.bmp'
                Image.fromarray(flow.active_to_native(np.full((478,478),gray,np.uint8))).save(path)
                values,_=bench.capture('gray_response',flat,[path],[f'e{exposure}_g{gray}'],'identity')
                rows.append(bench.rows[-1])
    flow.write(out/'report.json',{'camera':'SHS','corners_screen_TL_TR_BR_BL':CORNERS.tolist(),'rows':rows})


if __name__ == '__main__':
    flow.BASE_CORNERS=CORNERS.copy()
    if '--health-output' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--health-output',type=Path,required=True)
        health(p.parse_args().health_output)
    else:
        pipeline.Bench=SHSBench
        pipeline.main()
