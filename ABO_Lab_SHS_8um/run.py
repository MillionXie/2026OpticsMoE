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
    if '--config' not in sys.argv:sys.argv+=['--config',str(ROOT/'config.json')]
    import hardware
    from slm_camera import Controller
    class Bench(Controller):
        raw_suffix='.raw.png'
        def capture(self,bmp,path,rectify=True):
            if bmp is None:raise ValueError('Camera-only capture: use capture.py')
            raw,meta=super().capture(bmp)
            out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
            rawpath=out.with_suffix(self.raw_suffix);Image.fromarray(raw).save(rawpath)
            with Image.open(rawpath) as im:
                if not np.array_equal(raw,np.asarray(im)):raise RuntimeError('Raw PNG changed pixels')
            meta['hardware_identity']=common.hardware_identity(self.c)
            common.write(out.with_suffix('.json'),meta)
            if not rectify:return raw
            frame=hardware.canonical(raw,self.c);Image.fromarray(frame).save(out.with_suffix('.png'));return frame
    hardware.Bench=Bench
    spec=importlib.util.spec_from_file_location('abo_shs_task',compat/'run.py');task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    # Its ROOT/run.py subprocess entry resolves to THIS adapter, never legacy DVP.
    if len(sys.argv)>1 and sys.argv[1] in ('probe','exposure'):
        raise ValueError('Use capture.py for standalone camera tests; optical exposure scan is not validated yet')
    task.main()


if __name__=='__main__':main()
