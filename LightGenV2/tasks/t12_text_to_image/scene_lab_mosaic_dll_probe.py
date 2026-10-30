"""Isolated two-pattern Mosaic-shipped DLL comparison; never replaces a driver.

The provided v2 bindings/structures are reused only after ProductVersion and
required-export admission. Configuration XML is copied as DATA to this run.
Camera caps, exposure, device ROI, LUTs and original SDK files are untouched.
"""
import argparse,ctypes,json,os,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
from .scene_lab_protocol import put,sha
from .scene_lab_hardware import Bench,native
from .scene_lab_vendor_frame_probe import header_fields
from .scene_lab_sdk_canary import signal_pair


def version_tuple(info,prefix='ProductVersion'):
    ms,ls=info[prefix+'MS'],info[prefix+'LS']
    return ((ms>>16)&65535,ms&65535,(ls>>16)&65535,ls&65535)


class IsolatedMosaicCamera(TucamCamera):
    def __init__(self,sdk_path,*,mosaic_dll,config_copy,**kwargs):
        super().__init__(Path(sdk_path),**kwargs);self.binding_root=Path(sdk_path);self.mosaic_dll=Path(mosaic_dll);self.config_copy=Path(config_copy);self.native_directory=None;self.admission={}
    def _load_vendor_module(self):
        import win32api
        primary=self.binding_root/'lib/x64/TUCam.dll';a=win32api.GetFileVersionInfo(str(primary),'\\');b=win32api.GetFileVersionInfo(str(self.mosaic_dll),'\\')
        if version_tuple(a)!=version_tuple(b)or version_tuple(a)!=(2,0,7,0):raise ValueError('ProductVersion admission failed; no camera API init')
        if sha(self.mosaic_dll)!='d1fae23cd2700073b8d00b4b59ebe2c27234ef8ca1dd2829cbd913375d907bdc':raise ValueError('Mosaic DLL identity changed')
        module=super()._load_vendor_module() # binding types only; primary API not initialized
        self.native_directory=os.add_dll_directory(str(self.mosaic_dll.parent));dll=ctypes.OleDLL(str(self.mosaic_dll.resolve()))
        required=('TUCAM_Api_Init','TUCAM_Api_Uninit','TUCAM_Dev_Open','TUCAM_Dev_Close','TUCAM_Dev_GetInfo','TUCAM_Capa_GetValue','TUCAM_Prop_GetValue','TUCAM_Cap_GetROI',
                  'TUCAM_Buf_Alloc','TUCAM_Buf_WaitForFrame','TUCAM_Buf_AbortWait','TUCAM_Buf_Release','TUCAM_Cap_Start','TUCAM_Cap_Stop','TUCAM_File_SaveImage')
        if any(not hasattr(dll,n)for n in required):raise ValueError('Required vendor export missing; no camera API init')
        if ctypes.sizeof(module.TUCAM_FRAME)!=56 or ctypes.sizeof(module.TUCAM_FILE_SAVE)!=24:raise ValueError('Unexpected declared v2 ABI; no camera API init')
        def unavailable(*args):raise RuntimeError('Unexported function is disabled in isolated SDK comparison')
        exports=[]
        for name in dir(module):
            old=getattr(module,name)
            if not name.startswith(('TUCAM_','TUIMG_'))or not isinstance(old,ctypes._CFuncPtr):continue
            if hasattr(dll,name):
                function=getattr(dll,name);function.argtypes=old.argtypes;function.restype=ctypes.c_int32;setattr(module,name,function);exports.append(name)
            else:setattr(module,name,unavailable)
        module.TUSDKdll=dll;self.sdk_path=self.config_copy
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32];kernel.GetModuleFileNameW.restype=ctypes.c_uint32
        buf=ctypes.create_unicode_buffer(32768);assert kernel.GetModuleFileNameW(dll._handle,buf,len(buf))
        if Path(buf.value).resolve()!=self.mosaic_dll.resolve():raise ValueError('Loaded wrong alternate DLL')
        self.admission=dict(actual_loaded_DLL=buf.value,SHA256=sha(self.mosaic_dll),product_version=list(version_tuple(b)),file_version=list(version_tuple(b,'FileVersion')),
            primary_API_initialized=False,required_exports=list(required),exported_count=len(exports),frame_bytes=56,file_save_bytes=24,config_copy=str(self.config_copy),
            ABI_admission_scope='same declared ProductVersion and required symbols; not a proof of vendor internal semantic equivalence')
        return module
    def close(self):
        try:super().close()
        finally:
            if self.native_directory:self.native_directory.close();self.native_directory=None


def main():
    from LightGenV2.hardware_common.shs.phase_owner import message_pump
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3];lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Other owner')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out))).encode());os.close(fd)
    originals={};rows=[];frames={}
    try:
        config_copy=a.out/'camera_profile_copy';config_copy.mkdir()
        for source in Path(cfg['camera_sdk']).glob('*.xml'):
            originals[str(source)]=sha(source);shutil.copy2(source,config_copy/source.name)
        if not originals:raise ValueError('No original camera XML to isolate; no API init')
        put(a.out/'config_copy_manifest.json',dict(originals=originals,only_XML_data_copied=True,no_source_or_DLL_copied=True))
        factory=lambda path,**kw:IsolatedMosaicCamera(path,mosaic_dll='E:/Mosaic3.0/TUCam.dll',config_copy=config_copy,**kw)
        with Bench(cfg,a.out,camera_factory=factory)as bench:
            camera=bench.camera;put(a.out/'DLL_admission.json',camera.admission);pump=message_pump();zero=np.zeros((478,478),np.uint8)
            for tag,amp in [('dark',zero),('white',np.full_like(zero,255))]:
                folder=a.out/tag;folder.mkdir();ap=folder/'amplitude.bmp';pp=folder/'phase.bmp'
                Image.fromarray(native(amp,'amplitude')).save(ap);Image.fromarray(native(zero,'phase')).save(pp);bench.phase.show(pp,pump=pump);bench.slm.preload_files([ap]);bench.slm.display_file(ap)
                until=time.monotonic()+.8
                while time.monotonic()<until:pump();time.sleep(.01)
                for _ in range(6):camera._grab_array()
                raw=camera._grab_array();np.save(folder/'native.npy',raw,allow_pickle=False);frames[tag]=raw
                row=dict(tag=tag,SHA256=sha(folder/'native.npy'),mean=float(raw.mean()),p99=float(np.percentile(raw,99)),maximum=int(raw.max()),header=header_fields(camera._frame))
                m=camera._module;save=m.TUCAM_FILE_SAVE();save.nSaveFmt=m.TUIMG_FORMATS.TUFMT_TIF.value;save.pFrame=ctypes.pointer(camera._frame);save.pstrSavePath=str(folder/'vendor_native').encode();camera._check(m.TUCAM_File_SaveImage(camera._handle(),save),'Mosaic DLL native save')
                row['vendor_files']={p.name:sha(p)for p in folder.glob('vendor_native*')};put(folder/'receipt.json',row);rows.append(row);print(json.dumps(row),flush=True)
        if any(sha(path)!=digest for path,digest in originals.items()):raise ValueError('Original SDK XML changed; preserve and investigate')
        put(a.out/'report.json',dict(complete=True,rows=rows,comparison=signal_pair(frames['dark'],frames['white']),original_SDK_files_unchanged=True,no_DLL_replaced=True,
            camera_caps_exposure_ROI_LUT_unchanged=True,SDK_released=True,model_TRAIN_CCD=0,TEST_used=False))
    except BaseException as error:
        import traceback
        put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),rows=rows,automatic_retry=False));raise
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(pid=os.getpid(),own_lock_released=True,closed_context=True))


if __name__=='__main__':main()
