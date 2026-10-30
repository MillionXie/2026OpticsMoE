"""Four-pattern maximum RAW/USUAL buffer-format diagnostic; no camera cap edits."""
import argparse,ctypes,json,os,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from .scene_lab_hardware import Bench,native
from .scene_lab_protocol import put,sha
from .scene_lab_vendor_frame_probe import header_fields


def set_buffer_format(camera,code):
    m=camera._module;handle=camera._handle()
    if camera._capture_started:
        m.TUCAM_Buf_AbortWait(handle);camera._check(m.TUCAM_Cap_Stop(handle),'stop for buffer-format comparison');camera._capture_started=False
    if camera._buffer_allocated:
        camera._check(m.TUCAM_Buf_Release(handle),'release old buffer');camera._buffer_allocated=False
    frame=m.TUCAM_FRAME();frame.pBuffer=0;frame.ucFormatGet=code;frame.uiRsdSize=1;camera._frame=frame
    camera._check(m.TUCAM_Buf_Alloc(handle,ctypes.pointer(frame)),'allocate new buffer format');camera._buffer_allocated=True
    camera._check(m.TUCAM_Cap_Start(handle,m.TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value),'restart same sequence mode');camera._capture_started=True


def main():
    from LightGenV2.hardware_common.shs.phase_owner import message_pump
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3];lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Other owner')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out))).encode());os.close(fd)
    rows=[]
    try:
        with Bench(cfg,a.out)as bench:
            pump=message_pump();camera=bench.camera;m=camera._module;zero=np.zeros((478,478),np.uint8)
            for name,code in [('USUAL',m.TUFRM_FORMATS.TUFRM_FMT_USUAl.value),('RAW',m.TUFRM_FORMATS.TUFRM_FMT_RAW.value)]:
                if name=='RAW':set_buffer_format(camera,code)
                for tag,amp in [('dark',zero),('white',np.full_like(zero,255))]:
                    folder=a.out/(name+'_'+tag);folder.mkdir();ap=folder/'amplitude.bmp';pp=folder/'phase.bmp'
                    Image.fromarray(native(amp,'amplitude')).save(ap);Image.fromarray(native(zero,'phase')).save(pp)
                    bench.phase.show(pp,pump=pump);bench.slm.preload_files([ap]);bench.slm.display_file(ap)
                    until=time.monotonic()+.8
                    while time.monotonic()<until:pump();time.sleep(.01)
                    for _ in range(7):camera._check(m.TUCAM_Buf_WaitForFrame(camera._handle(),ctypes.pointer(camera._frame),10000),'vendor wait, no pixel conversion')
                    row=dict(format=name,format_request=code,pattern=tag,header=header_fields(camera._frame),amplitude_SHA256=sha(ap),phase_SHA256=sha(pp))
                    save=m.TUCAM_FILE_SAVE();save.nSaveFmt=m.TUIMG_FORMATS.TUFMT_TIF.value;save.pFrame=ctypes.pointer(camera._frame);save.pstrSavePath=str(folder/'vendor_native').encode()
                    ret=m.TUCAM_File_SaveImage(camera._handle(),save);row['vendor_save_return']=camera._result_value(ret)
                    if row['vendor_save_return']==m.TUCAMRET.TUCAMRET_SUCCESS.value:
                        files=list(folder.glob('vendor_native*'))
                        if len(files)!=1:raise ValueError('Unexpected vendor file count')
                        with Image.open(files[0])as im:value=np.asarray(im).copy();row['vendor_TIFF']=dict(path=str(files[0]),mode=im.mode,dtype=str(value.dtype),shape=list(value.shape),mean=float(value.mean()),p99=float(np.percentile(value,99)),maximum=int(value.max()),SHA256=sha(files[0]))
                    try:
                        arr=camera.frame_to_array(camera._frame);np.save(folder/'adapter.npy',arr,allow_pickle=False);row['adapter']=dict(dtype=str(arr.dtype),shape=list(arr.shape),mean=float(arr.mean()),p99=float(np.percentile(arr,99)),maximum=int(arr.max()),SHA256=sha(folder/'adapter.npy'))
                    except Exception as error:row['adapter_decode_error']=repr(error)
                    row['live_exposure_us']=camera._get_property(m.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000;row['live_gain']=camera._get_property(m.TUCAM_IDPROP.TUIDP_GLOBALGAIN)
                    if abs(row['live_exposure_us']-cfg['exposure_us'])>2 or row['live_gain']!=cfg['gain']:raise ValueError('Parameter drift')
                    put(folder/'receipt.json',row);rows.append(row);print(json.dumps(row),flush=True)
        put(a.out/'report.json',dict(complete=True,rows=rows,max_patterns=4,buffer_format_only=True,camera_caps_exposure_ROI_LUT_unchanged=True,SDK_released=True,model_TRAIN_CCD=0,TEST_used=False))
    except BaseException as error:put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),rows=rows,automatic_retry=False));raise
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(pid=os.getpid(),own_lock_released=True,closed_context=True))


if __name__=='__main__':main()
