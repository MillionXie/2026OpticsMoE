"""Two white frames in a camera-only process while an existing SLM holder runs.

No SLM modules/SDKs are initialized in this process. No exposure/gain/ROI writes,
DLL replacements, model inference or calibration fitting are performed.
"""
import argparse
from datetime import datetime,timezone
import json,os,traceback
from pathlib import Path
import numpy as np
from .scene_lab_protocol import put,sha


def validate_holder(ready,lock,now):
    if not ready['SLMs_held']or ready['camera_SDK_opened']or lock.get('scope')!='SLMs_only'or lock.get('camera_SDK_owned')is not False:
        raise ValueError('An explicit SLM-only owner is required')
    if(datetime.fromisoformat(ready['deadline_UTC'])-now).total_seconds()<20:
        raise ValueError('Insufficient remaining white-holder time; no camera init')


def audit_vendor_save(camera,path,decoded):
    """Save the SAME acquired frame via the official SDK, with no new capture."""
    import ctypes
    from PIL import Image
    path=Path(path)
    if path.exists()or path.with_suffix('.tif').exists():raise FileExistsError('Preserve prior vendor frame audit')
    m=camera._module
    request=m.TUCAM_FILE_SAVE(m.TUIMG_FORMATS.TUFMT_TIF.value,str(path).encode('mbcs'if os.name=='nt'else'utf-8'),ctypes.pointer(camera._frame))
    camera._check(m.TUCAM_File_SaveImage(camera._handle(),request),'vendor same-frame TIFF save')
    candidates=[p for p in (path,path.with_suffix('.tif'),Path(str(path)+'.tif'))if p.is_file()]
    candidates=list(dict.fromkeys(candidates))
    if len(candidates)!=1:raise ValueError('Expected exactly one official saved TIFF')
    file=candidates[0]
    with Image.open(file)as im:value=np.asarray(im).copy()
    equal=value.shape==decoded.shape and np.array_equal(value,decoded)
    return dict(file=file.name,SHA256=sha(file),shape=list(value.shape),dtype=str(value.dtype),
                mean=float(value.mean()),p99=float(np.percentile(value,99)),maximum=int(value.max()),
                equals_decoded_array=equal,new_camera_frame=False,settings_written=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--holder',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--vendor-save-audit',action='store_true');a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(__file__).resolve().parents[3];holder_lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    validate_holder(json.loads((a.holder/'ready.json').read_text()),json.loads(holder_lock.read_text()),datetime.now(timezone.utc))
    if(root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock').exists():raise RuntimeError('Another camera owner')
    a.out.mkdir(parents=True,exist_ok=False);own=a.out/'camera_owner.lock';fd=os.open(own,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid())).encode());os.close(fd);rows=[]
    try:
        for name,digest in cfg['fixed_files'].items():
            if sha(name)!=digest:raise ValueError('Device/calibration identity changed')
        from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
        put(a.out/'protocol.json',dict(source_SHA256=sha(__file__),holder_ready_SHA256=sha(a.holder/'ready.json'),camera_only_process=True,SLM_SDK_initialized=False,parameter_changes=False,max_raw_frames=2,TEST_used=False))
        with TucamCamera(cfg['camera_sdk'],timeout_ms=10000,auto_exposure=None,exposure_us=None,analog_gain=None,warmup_frames=3,discard_frames_after_display=6,saved_frame_size_wh=None,saved_frame_resize_mode='none',saved_frame_bit_depth=None)as camera:
            m=camera._module
            actual=dict(exposure_us=camera._get_property(m.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000,gain=camera._get_property(m.TUCAM_IDPROP.TUIDP_GLOBALGAIN),auto=camera._get_capability(m.TUCAM_IDCAPA.TUIDC_ATEXPOSURE))
            put(a.out/'camera_settings.json',dict(actual=actual,device=camera.device_info()))
            if abs(actual['exposure_us']-cfg['exposure_us'])>2 or actual['gain']!=cfg['gain']or actual['auto']!=0:raise ValueError('Camera setting mismatch; no automatic reset')
            for i in range(2):
                validate_holder(json.loads((a.holder/'ready.json').read_text()),json.loads(holder_lock.read_text()),datetime.now(timezone.utc))
                file=a.out/f'white_{i}.npy';camera.capture(file);v=np.load(file,allow_pickle=False)
                if v.shape!=(2048,2048)or v.dtype!=np.uint16:raise ValueError('Unexpected native format')
                row=dict(file=file.name,SHA256=sha(file),capture_UTC=datetime.now(timezone.utc).isoformat(),mean=float(v.mean()),p99=float(np.percentile(v,99)),maximum=int(v.max()),saturation_fraction=float((v==65535).mean()))
                if a.vendor_save_audit:row['official_same_frame_TIFF']=audit_vendor_save(camera,a.out/f'vendor_white_{i}',v)
                rows.append(row);put(a.out/f'white_{i}_receipt.json',row);print(json.dumps(row),flush=True)
        put(a.out/'report.json',dict(complete=True,rows=rows,clear_white_signal=all(r['mean']>1000 and r['p99']>4000 for r in rows),SDK_released=True,camera_only_process=True,SLM_SDK_initialized=False,model_TRAIN_CCD=0,TEST_used=False))
    except BaseException as error:
        put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),frames_saved=len(rows),automatic_retry=False));raise
    finally:
        assert json.loads(own.read_text())['pid']==os.getpid();own.unlink();put(a.out/'release.json',dict(pid=os.getpid(),own_camera_lock_released=True,SLM_owner_not_stopped=True))


if __name__=='__main__':main()
