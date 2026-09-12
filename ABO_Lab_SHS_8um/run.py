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


def save_capture(raw,meta,path,c,canonical,write,identity,rectify=True):
    """Keep one canonical PNG by default; full-sensor diagnostics are opt-in."""
    out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
    keep_raw=c.get('save_raw_frames',False) or not rectify
    if keep_raw:Image.fromarray(raw).save(out.with_suffix('.raw.png'),compress_level=1)
    meta.update(hardware_identity=identity,raw_frame_saved=keep_raw)
    frame=canonical(raw,c) if rectify else raw
    if rectify:Image.fromarray(frame).save(out.with_suffix('.png'),compress_level=1)
    write(out.with_suffix('.json'),meta)
    return frame


def main():
    batch=None
    if '--guard-batch' in sys.argv:
        index=sys.argv.index('--guard-batch');batch_path=Path(sys.argv[index+1]).resolve()
        del sys.argv[index:index+2]
        if not batch_path.is_relative_to(ROOT/'sessions'):raise ValueError('Batch manifest outside sessions')
        batch=json.loads(batch_path.read_text(encoding='utf-8'))
        if batch['state']!='pending' or len(sys.argv)<2 or sys.argv[1]!='capture':raise ValueError('Guard batch only for pending capture')
        for flag,key in (('--stage','stage'),('--session','session')):
            if flag not in sys.argv or sys.argv[sys.argv.index(flag)+1]!=batch[key]:raise ValueError('Batch CLI identity mismatch')
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
            return save_capture(raw,meta,path,self.c,hardware.canonical,common.write,
                                common.hardware_identity(self.c),rectify)
    hardware.Bench=Bench
    spec=importlib.util.spec_from_file_location('abo_shs_task',compat/'run.py');task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    original_load=task.load_session
    def guarded_load(name,c):
        root,state=original_load(name,c)
        from guarded_batch import assert_no_pending
        if batch is None:assert_no_pending(root)
        elif name!=batch['session'] or common.hardware_identity(c)!=batch['hardware_identity']:raise ValueError('Batch/session/config mismatch')
        return root,state
    task.load_session=guarded_load
    if batch is not None:
        original_read=task.read
        expected=(ROOT/'sessions'/batch['session']/'play'/batch['stage']/'manifest.json').resolve()
        from phase_fingerprint import digest
        def batch_read(path):
            value=original_read(path)
            if Path(path).resolve()==expected:
                if digest(value)!=batch['manifest_sha256']:raise ValueError('Prepared manifest changed during batch')
                value=dict(value);value['entries']=[e for e in value['entries'] if e['id'] in batch['ids']]
                if len(value['entries'])!=len(batch['ids']):raise ValueError('Batch entries missing')
            return value
        task.read=batch_read
    # Its ROOT/run.py subprocess entry resolves to THIS adapter, never legacy DVP.
    if len(sys.argv)>1 and sys.argv[1] in ('probe','exposure'):
        raise ValueError('Use capture.py for standalone camera tests; optical exposure scan is not validated yet')
    task.main()


if __name__=='__main__':main()
