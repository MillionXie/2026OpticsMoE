"""DVP2 x64 native capture; raw mono only, no display normalization or hardware reset."""
import ctypes as C
import json
import time
from pathlib import Path
import numpy as np


class Frame(C.Structure):
    _fields_=[('format',C.c_int),('bits',C.c_int),('bytes',C.c_uint32),
              ('width',C.c_int32),('height',C.c_int32),('id',C.c_uint64),('timestamp',C.c_uint64),
              ('exposure',C.c_double),('gain',C.c_float),('position',C.c_int),
              ('flip_h',C.c_bool),('flip_v',C.c_bool),('rotate',C.c_bool),('opposite',C.c_bool),
              ('flags',C.c_uint32),('value',C.c_uint32),('trigger_id',C.c_uint64),
              ('user',C.c_uint64),('reserved',C.c_uint32*24)]


class Camera:
    def __init__(self,dll):
        self.dll=C.WinDLL(str(Path(dll).resolve())); self.handle=C.c_uint32();self.opened=False
        signatures={'Refresh':[C.POINTER(C.c_uint32)],'Open':[C.c_uint32,C.c_int,C.POINTER(C.c_uint32)],
          'Close':[C.c_uint32],'Start':[C.c_uint32],'Stop':[C.c_uint32],
          'GetFrame':[C.c_uint32,C.POINTER(Frame),C.POINTER(C.c_void_p),C.c_uint32],
          'GetExposure':[C.c_uint32,C.POINTER(C.c_double)],'SetExposure':[C.c_uint32,C.c_double],
          'GetAnalogGain':[C.c_uint32,C.POINTER(C.c_float)],'SetAnalogGain':[C.c_uint32,C.c_float],
          'SetTriggerState':[C.c_uint32,C.c_bool],'SetAeOperation':[C.c_uint32,C.c_int]}
        for name,args in signatures.items():
            fn=getattr(self.dll,'dvp'+name);fn.argtypes=args;fn.restype=C.c_int
    def call(self,name,*args):
        status=getattr(self.dll,'dvp'+name)(*args)
        if status!=1:raise RuntimeError(f'DVP {name}: {status}')
    def __enter__(self):
        n=C.c_uint32();self.call('Refresh',C.byref(n))
        if n.value!=1:raise RuntimeError(f'Expected one camera, found {n.value}')
        self.call('Open',0,1,C.byref(self.handle));self.opened=True
        try:
            self.call('SetTriggerState',self.handle,False)
            self.call('SetAeOperation',self.handle,0)
            self.call('Start',self.handle)
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def settings(self,exposure=None,gain=None):
        if exposure is not None:self.call('SetExposure',self.handle,float(exposure))
        if gain is not None:self.call('SetAnalogGain',self.handle,float(gain))
        e=C.c_double();g=C.c_float()
        self.call('GetExposure',self.handle,C.byref(e));self.call('GetAnalogGain',self.handle,C.byref(g))
        return dict(exposure_us=e.value,gain=g.value)
    def capture(self):
        f=Frame();buf=C.c_void_p();t=time.perf_counter()
        self.call('GetFrame',self.handle,C.byref(f),C.byref(buf),5000)
        elapsed=(time.perf_counter()-t)*1000
        if f.format!=0 or f.bits not in (0,1,2,3,4) or not (0<f.width<=10000 and 0<f.height<=10000):
            raise RuntimeError('Unexpected DVP frame layout or non-mono output')
        count=f.width*f.height
        if f.bytes not in (count,count*2):raise RuntimeError('Unsupported packed/stride frame')
        a=np.frombuffer(C.string_at(buf,f.bytes),dtype=np.uint8 if f.bytes==count else np.uint16).reshape(f.height,f.width).copy()
        return a,dict(frame_id=f.id,timestamp=f.timestamp,exposure_us=f.exposure,gain=f.gain,
                      format=f.format,bits=f.bits,shape=list(a.shape),capture_ms=elapsed,
                      flip_h=f.flip_h,flip_v=f.flip_v,rotate=f.rotate,opposite=f.opposite)
    def __exit__(self,*args):
        if self.opened:
            try:self.call('Stop',self.handle)
            finally:self.call('Close',self.handle);self.opened=False


def main():
    import argparse
    from PIL import Image
    p=argparse.ArgumentParser();p.add_argument('--dll',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with Camera(a.dll) as camera:
        original=camera.settings()
        for i in range(8):
            image,meta=camera.capture();rows.append(meta)
        Image.fromarray(image).save(out/'raw.png')
    (out/'report.json').write_text(json.dumps(dict(original=original,frames=rows,normalization='none',frame_struct_bytes=C.sizeof(Frame)),indent=2))
    print(json.dumps(dict(original=original,final=rows[-1],p99=float(np.percentile(image,99)),maximum=int(image.max()))))


if __name__=='__main__':main()
