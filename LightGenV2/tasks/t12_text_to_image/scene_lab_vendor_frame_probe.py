"""Two-pattern native vendor TIFF/NumPy frame parsing and DLL identity comparison."""
import argparse,ctypes,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from .scene_lab_hardware import Bench,native
from .scene_lab_protocol import put,sha
from .scene_lab_sdk_canary import signal_pair


def header_fields(frame):
    result={}
    for name,*_ in frame._fields_:
        value=getattr(frame,name)
        result[name]=value.hex()if isinstance(value,bytes)else(value if value is not None else 0)
    return result


def compare_file(path,adapter):
    with Image.open(path)as image:
        mode=image.mode;value=np.asarray(image).copy()
    return dict(path=str(path),SHA256=sha(path),mode=mode,shape=list(value.shape),dtype=str(value.dtype),
        mean=float(value.mean()),p99=float(np.percentile(value,99)),maximum=int(value.max()),
        same_pixel_values=value.shape==adapter.shape and bool(np.array_equal(value,adapter)))


def main():
    from LightGenV2.hardware_common.shs.phase_owner import message_pump
    import win32api
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3];lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Other owner')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out))).encode());os.close(fd)
    rows=[];patterns={}
    try:
        with Bench(cfg,a.out)as bench:
            camera=bench.camera;module=camera._module;dll=module.TUSDKdll
            kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32];kernel.GetModuleFileNameW.restype=ctypes.c_uint32
            text=ctypes.create_unicode_buffer(32768)
            if not kernel.GetModuleFileNameW(dll._handle,text,len(text)):raise OSError('Cannot identify actual loaded DLL')
            actual=Path(text.value);expected=Path(cfg['camera_sdk'])/'lib/x64/TUCam.dll'
            if actual.resolve()!=expected.resolve()or sha(actual)!=sha(expected):raise ValueError('Unexpected actual loaded camera DLL')
            version=win32api.GetFileVersionInfo(str(actual),'\\');identity=dict(actual_loaded_DLL=str(actual),actual_loaded_SHA256=sha(actual),version=version,binding_source_SHA256=sha(Path(cfg['camera_sdk'])/'TUCam.py'),
                frame_structure_bytes=ctypes.sizeof(module.TUCAM_FRAME),field_offsets={name:getattr(module.TUCAM_FRAME,name).offset for name,*_ in module.TUCAM_FRAME._fields_})
            put(a.out/'DLL_and_ABI.json',identity);pump=message_pump();zero=np.zeros((478,478),np.uint8)
            for tag,amp in [('dark',zero),('white',np.full_like(zero,255))]:
                folder=a.out/tag;folder.mkdir();ap=folder/'amplitude.bmp';pp=folder/'phase.bmp'
                Image.fromarray(native(amp,'amplitude')).save(ap);Image.fromarray(native(zero,'phase')).save(pp)
                bench.phase.show(pp,pump=pump);bench.slm.preload_files([ap]);bench.slm.display_file(ap)
                until=time.monotonic()+.8
                while time.monotonic()<until:pump();time.sleep(.01)
                for _ in range(6):camera._grab_array()
                # One WaitForFrame, copied via the current adapter; the vendor
                # TIFF writer receives the exact same original TUCAM_FRAME.
                before=camera._grab_array();np.save(folder/'adapter.npy',before,allow_pickle=False);patterns[tag]=before
                row=dict(tag=tag,frame_header=header_fields(camera._frame),adapter_SHA256=sha(folder/'adapter.npy'),amplitude_SHA256=sha(ap),phase_SHA256=sha(pp),adapter_mean=float(before.mean()),adapter_p99=float(np.percentile(before,99)))
                save=module.TUCAM_FILE_SAVE();save.nSaveFmt=module.TUIMG_FORMATS.TUFMT_TIF.value;save.pFrame=ctypes.pointer(camera._frame);save.pstrSavePath=str(folder/'vendor_native').encode('utf-8')
                camera._check(module.TUCAM_File_SaveImage(camera._handle(),save),'official same-frame TIFF save')
                files=list(folder.glob('vendor_native*'))
                if len(files)!=1:raise ValueError(('Unexpected vendor output',list(map(str,files))))
                row['vendor_TIFF']=compare_file(files[0],before)
                after=camera.frame_to_array(camera._frame);row['same_buffer_after_vendor_save']=bool(np.array_equal(before,after))
                # A separate published vendor acquisition function is bounded
                # to one call per unchanged pattern; no capture-setting writes.
                result=module.TUCAM_Buf_CopyFrame(camera._handle(),ctypes.pointer(camera._frame));row['CopyFrame_return']=camera._result_value(result)
                if row['CopyFrame_return']==module.TUCAMRET.TUCAMRET_SUCCESS.value:
                    copy=camera.frame_to_array(camera._frame);np.save(folder/'copy_API.npy',copy,allow_pickle=False);row['CopyFrame_header']=header_fields(camera._frame)
                    row['CopyFrame_stats']=dict(mean=float(copy.mean()),p99=float(np.percentile(copy,99)),maximum=int(copy.max()),SHA256=sha(folder/'copy_API.npy'))
                put(folder/'receipt.json',row);rows.append(row);print(json.dumps(row),flush=True)
        put(a.out/'report.json',dict(complete=True,rows=rows,comparison=signal_pair(patterns['dark'],patterns['white']),
            WaitForFrame_patterns=2,CopyFrame_max_calls=2,TIFF_same_frame_not_extra_exposure=True,exposure_ROI_LUT_changed=False,SDK_released=True,model_TRAIN_CCD=0,TEST_used=False))
    except BaseException as error:
        import traceback
        put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),rows=rows,automatic_retry=False));raise
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(pid=os.getpid(),own_lock_released=True,closed_context=True))


if __name__=='__main__':main()
