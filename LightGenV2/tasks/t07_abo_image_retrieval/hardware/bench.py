"""Actual rank72 SHS bench with explicit external machine paths.
No model or device opening on import; preserves capture and close ordering.
This is not a standalone acquisition runner or authorization to recapture.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from . import geometry as flow
from LightGenV2.hardware_common.shs.capture import snapshot
from LightGenV2.hardware_common.shs.slm_camera import Controller
from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI
class SHSBench:
    def __init__(self, out, exposure, wait_ms, phase_paths, *, machine_config, phase_sdk, phase_lut, amplitude_sdk=None):
        self.out=out;self.exposure=exposure;self.wait=wait_ms/1000
        self.phase_paths=phase_paths;self.rows=[]
        machine_config=Path(machine_config).resolve()
        config=json.loads(machine_config.read_text(encoding='utf-8-sig'))
        if amplitude_sdk is not None:
            amplitude_sdk=Path(amplitude_sdk).resolve()
            if not amplitude_sdk.is_dir():raise FileNotFoundError(amplitude_sdk)
            config['amplitude_slm']['sdk_path']=str(amplitude_sdk)
        config['camera']['exposure_us']=exposure
        config['camera']['gain']='Gain_X4'
        config['settle_delay_ms']=wait_ms
        self.controller=Controller(config,config_base=machine_config.parent)
        self.phase=PhaseHDMI(phase_sdk,phase_lut,settle_s=.8,pixel_format='rgba')
        self.current_phase=None

    def __enter__(self):
        try:
            self.phase.__enter__()
            self.controller.__enter__()
            self.camera=self.controller.camera;self.amp=self.controller.slm
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
        for device in (self.controller,self.phase):
            try: device.__exit__(*args)
            except Exception as ex: errors.append(str(ex))
        if errors and args[0] is None: raise RuntimeError('; '.join(errors))

    def capture(self,stage,phase_path,amplitude_paths,ids,camera_orientation,save=True):
        digest=flow.sha(phase_path)
        if self.current_phase != digest:
            self.receipt=self.phase.show(phase_path)
            self.current_phase=digest
            # The phase SDK blocks during its own settling; discard queued
            # frames before the next amplitude Controller cycle starts.
            self.camera.fresh()
        receipt=self.receipt
        values=[]
        folder=self.out/'ccd'/stage
        if save: folder.mkdir(parents=True,exist_ok=True)
        for sample_id,path in zip(ids,amplitude_paths):
            start=time.perf_counter()
            self.controller.c['settle_delay_ms']=self.wait*1000
            raw,meta=self.controller.capture(path)
            drained=meta['settle_drained_frames']
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
