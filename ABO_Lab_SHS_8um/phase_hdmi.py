"""Local Meadowlark HDMI phase display. No implicit flip/inversion/LUT fitting.

Only final native BMP bytes are displayed; encoding belongs to the exporter.
Never change coverglass voltage or timing ramps. Close Blink GUI before use.
"""
import argparse,ctypes as C,hashlib,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_native(path,expected_sha=None):
    if expected_sha and sha(path)!=expected_sha:raise ValueError('Phase SHA mismatch')
    with Image.open(path) as im:
        if im.format!='BMP' or im.mode!='L' or im.size!=(1920,1200):
            raise ValueError('Require FINAL 1920x1200 8-bit grayscale BMP; no implicit resize/flip')
        return np.asarray(im).copy()

class PhaseHDMI:
    def __init__(self,sdk,lut,settle_s=1.0):
        self.sdk=Path(sdk).resolve();self.lut=Path(lut).resolve();self.settle_s=float(settle_s)
        if not .5<=self.settle_s<=10:raise ValueError('Phase settle must be 0.5..10 seconds')
        self.dll=None;self.created=False;self.dir=None
    def __enter__(self):
        if os.name!='nt':raise RuntimeError('Windows x64 required')
        if not self.lut.is_file():raise FileNotFoundError(self.lut)
        # SDK requires per-monitor DPI awareness for unscaled native addressing.
        C.windll.shcore.SetProcessDpiAwareness(2)
        self.dir=os.add_dll_directory(str(self.sdk))
        self.dll=C.CDLL(str(self.sdk/'Blink_C_wrapper.dll'))
        for name in ['Create_SDK','Delete_SDK']:
            fn=getattr(self.dll,name);fn.argtypes=[];fn.restype=None
        for name in ['Get_Width','Get_Height','Get_Depth','Get_SLMFound','Get_COMFound']:
            fn=getattr(self.dll,name);fn.argtypes=[];fn.restype=C.c_int
        self.dll.Load_lut.argtypes=[C.c_char_p];self.dll.Load_lut.restype=C.c_int
        self.dll.Write_image.argtypes=[C.POINTER(C.c_ubyte),C.c_int];self.dll.Write_image.restype=C.c_int
        try:
            self.dll.Create_SDK();self.created=True
            self.info={k:int(getattr(self.dll,'Get_'+k)()) for k in ['Width','Height','Depth','SLMFound','COMFound']}
            if (self.info['Width'],self.info['Height'],self.info['Depth'])!=(1920,1200,8):
                raise RuntimeError('Unexpected phase panel: '+str(self.info))
            if not self.info['SLMFound'] or not self.info['COMFound']:raise RuntimeError('Phase display/USB controller not found')
            if self.dll.Load_lut(str(self.lut).encode('mbcs'))<=0:raise RuntimeError('Phase LUT load failed')
            self.info.update(lut_sha256=sha(self.lut),lut=str(self.lut),lut_scope='linear voltage; NOT verified linear phase')
            return self
        except BaseException:self.close();raise
    def show(self,path,expected_sha=None):
        a=np.ascontiguousarray(load_native(path,expected_sha));t=time.perf_counter()
        if self.dll.Write_image(a.ctypes.data_as(C.POINTER(C.c_ubyte)),1)<=0:raise RuntimeError('Write_image failed')
        written=time.perf_counter();time.sleep(self.settle_s)
        self.current={'phase_file':str(Path(path).resolve()),'phase_sha256':sha(path),
            'write_call_ms':(written-t)*1000,'settle_s':self.settle_s,'sdk_ack_only':True,
            'no_extra_flip_or_inversion':True,'panel':self.info}
        return self.current
    def close(self):
        if self.created:self.dll.Delete_SDK();self.created=False
        if self.dir:self.dir.close();self.dir=None
    def __exit__(self,*args):self.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--sdk',default=str(ROOT/'vendor/phase_hdmi'))
    p.add_argument('--lut',default=str(ROOT/'vendor/phase_hdmi/19x12_8bit_linearVoltage.lut'))
    p.add_argument('--bmp',type=Path,required=True);p.add_argument('--hold-s',type=float,default=5)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if not 0<=a.hold_s<=60:raise ValueError('Bounded test hold: 0..60 seconds')
    with PhaseHDMI(a.sdk,a.lut) as phase:
        r=phase.show(a.bmp);print(json.dumps(r,indent=2),flush=True)
        a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
        time.sleep(a.hold_s)
    print('SDK closed; no guarantee last image persists after closing')

if __name__=='__main__':main()
