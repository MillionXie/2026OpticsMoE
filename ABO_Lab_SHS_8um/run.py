"""ABO six-stage adapter; same model/electronics, new SHS hardware identity.

Requires importing the fixed ABO runtime/model/dataset before init/prepare.
Camera-only bring-up uses probe.py/capture.py and does NOT need these weights.
"""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    compat=ROOT/'compat_abo'
    if not compat.exists():compat=ROOT.parent/'ABO_Lab_8um'
    sys.path.insert(0,str(compat))
    import common
    common.ROOT=ROOT
    if '--config' not in sys.argv:
        local=ROOT/'LAB.local.json'
        sys.argv+=['--config',str(local if local.exists() else ROOT/'config.json')]
    import hardware
    from slm_camera import Controller
    class Bench(Controller):
        raw_suffix='.raw.png'
        def __init__(self,c):
            super().__init__(c)
            # Legacy capture records hash raw_suffix unconditionally. Alias it
            # to the canonical PNG in minimal mode; there is only one image.
            self.raw_suffix='.raw.png' if c.get('save_raw_frames',False) else '.png'
        def capture(self,bmp,path,rectify=True):
            if bmp is None:raise ValueError('Camera-only capture: use capture.py')
            raw,meta=super().capture(bmp)
            out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
            keep_raw=self.c.get('save_raw_frames',False) or not rectify
            if keep_raw:
                rawpath=out.with_suffix('.raw.png');Image.fromarray(raw).save(rawpath,compress_level=1)
            meta['hardware_identity']=common.hardware_identity(self.c)
            meta['raw_frame_saved']=keep_raw
            common.write(out.with_suffix('.json'),meta)
            if not rectify:return raw
            frame=hardware.canonical(raw,self.c)
            Image.fromarray(frame).save(out.with_suffix('.png'),compress_level=1)
            return frame
    hardware.Bench=Bench
    spec=importlib.util.spec_from_file_location('abo_shs_task',compat/'run.py');task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    # Its ROOT/run.py subprocess entry resolves to THIS adapter, never legacy DVP.
    if len(sys.argv)>1 and sys.argv[1] in ('probe','exposure'):
        raise ValueError('Use capture.py for standalone camera tests; optical exposure scan is not validated yet')
    task.main()


if __name__=='__main__':main()
