"""At most nine human-authorized dark/white SDK canaries; no model or TEST.

Separate unchanged BMP bytes from camera buffering and retained display state.
No exposure/gain/ROI/LUT edits, display-origin moves or unbounded recovery.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from .scene_lab_hardware import Bench,native
from .scene_lab_protocol import sha,put


def signal_pair(dark,white):
    delta=white.astype(np.float32)-dark.astype(np.float32)
    return dict(mean_delta=float(delta.mean()),p99_delta=float(np.percentile(delta,99)),
                bright_fraction=float((delta>1000).mean()),passed=bool(np.percentile(delta,99)>1000 and delta.mean()>500),
                exactly_same_array=bool(np.array_equal(dark,white)))


def run(cfg,out):
    from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
    from LightGenV2.hardware_common.shs.phase_owner import message_pump,PhaseOwner
    out=Path(out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[3]
    lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    others=[root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']
    if any(p.exists()for p in others):raise RuntimeError('Another device owner')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(out))).encode());os.close(fd)
    put(out/'protocol.json',dict(config=cfg,human_confirmed_source_and_SLM_illumination=True,max_raw_frames=9,
        sequence=['dark_standard','white_standard','white_held','dark_repeat','white_direct_file','white_pumped_reassert','white_camera_reopen','white_phase_owner','dark_final'],
        no_parameter_changes=True,no_model_or_TEST=True,no_recurring=True,source_sha256=sha(__file__)))
    zero=np.zeros((478,478),np.uint8);white=np.full_like(zero,255);rows=[];comparisons={};outcome='no_effective_white'
    try:
        with Bench(cfg,out)as bench:
            pump=message_pump()
            def wait(seconds,pumped=False):
                until=time.monotonic()+seconds
                while time.monotonic()<until:
                    if pumped:pump()
                    time.sleep(.01)
            def snap(tag,amp,method='preload',pumped=False,phase_owner=False,extra_wait_s=0):
                folder=out/tag;folder.mkdir();ap=folder/'amplitude.bmp';pp=folder/'phase.bmp'
                Image.fromarray(native(amp,'amplitude')).save(ap);Image.fromarray(native(zero,'phase')).save(pp)
                if not phase_owner:phase=bench.phase.show(pp,pump=pump if pumped else None)
                else:phase=bench.phase.show(pp)
                if method=='preload':bench.slm.preload_files([ap])
                else:bench.slm._release_handles()
                bench.slm.display_file(ap);wait(cfg['wait_ms']/1000+extra_wait_s,pumped)
                # Read several native frames via the original drain-six backend.
                file=folder/'native.npy';bench.camera.capture(file);raw=np.load(file,allow_pickle=False)
                if raw.shape!=(2048,2048)or raw.dtype!=np.uint16:raise ValueError('Native format changed')
                module=bench.camera._module
                actual=dict(exposure_us=bench.camera._get_property(module.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000,
                            gain=bench.camera._get_property(module.TUCAM_IDPROP.TUIDP_GLOBALGAIN),auto=bench.camera._get_capability(module.TUCAM_IDCAPA.TUIDC_ATEXPOSURE))
                if abs(actual['exposure_us']-cfg['exposure_us'])>2 or actual['gain']!=cfg['gain']or actual['auto']!=0:raise ValueError('Actual camera setting drift')
                row=dict(tag=tag,raw_sha256=sha(file),amplitude_sha256=sha(ap),phase_sha256=sha(pp),actual_camera=actual,
                    camera_metadata=bench.camera.device_info(),phase_receipt=phase,amplitude_method=method,message_pump=pumped,extra_wait_s=extra_wait_s,
                    mean=float(raw.mean()),p99=float(np.percentile(raw,99)),maximum=int(raw.max()),saturation_fraction=float((raw==65535).mean()))
                put(folder/'receipt.json',row);rows.append(row);print(json.dumps({k:v for k,v in row.items()if k not in('phase_receipt','camera_metadata')}),flush=True)
                if row['saturation_fraction']>.001:raise ValueError('Saturated canary; no exposure edit')
                return raw
            dark=snap('dark_standard',zero)
            observed=snap('white_standard',white);comparisons['standard']=signal_pair(dark,observed)
            if comparisons['standard']['passed']:
                held=snap('white_held',white,extra_wait_s=2);comparisons['held']=signal_pair(dark,held)
                latest=snap('dark_repeat',zero);comparisons['repeat_dark_to_white']=signal_pair(latest,held)
                outcome='standard_restored' if comparisons['held']['passed']and comparisons['repeat_dark_to_white']['passed']else'nonrepeatable_standard'
            else:
                held=snap('white_held',white,extra_wait_s=2);comparisons['held']=signal_pair(dark,held)
                latest=snap('dark_repeat',zero)
                direct=snap('white_direct_file',white,method='direct');comparisons['direct_file']=signal_pair(latest,direct)
                pumped=snap('white_pumped_reassert',white,method='direct',pumped=True,extra_wait_s=2);comparisons['pumped_reassert']=signal_pair(latest,pumped)
                if comparisons['pumped_reassert']['passed']:
                    outcome='direct_or_message_pump_restored'
                else:
                    # One camera-only reconnect, preserving both displayed BMPs.
                    bench.camera.close()
                    camera=TucamCamera(cfg['camera_sdk'],timeout_ms=10000,auto_exposure=None,exposure_us=None,analog_gain=None,warmup_frames=3,discard_frames_after_display=6,
                        saved_frame_size_wh=None,saved_frame_resize_mode='none',saved_frame_bit_depth=None)
                    bench.camera=bench.stack.enter_context(camera)
                    reopened=snap('white_camera_reopen',white,method='direct',pumped=True,extra_wait_s=1);comparisons['camera_reopen']=signal_pair(latest,reopened)
                    if comparisons['camera_reopen']['passed']:outcome='camera_reopen_restored'
                    else:
                        # One phase-SDK owner change; identical flat bytes/LUT,
                        # no monitor relayout, firmware or voltage setting calls.
                        flat=out/'white_camera_reopen/phase.bmp';bench.phase.close()
                        link=dict(phase_sdk=cfg['phase_sdk'],phase_lut=cfg['phase_lut'],phase_settle_s=.8,phase_pixel_format='rgba',phase_startup_cycles=0,phase_display_align_top=False)
                        bench.phase=bench.stack.enter_context(PhaseOwner(link,flat,flat))
                        steady=snap('white_phase_owner',white,method='direct',pumped=True,phase_owner=True,extra_wait_s=2);comparisons['phase_owner']=signal_pair(latest,steady)
                        if comparisons['phase_owner']['passed']:outcome='persistent_phase_owner_restored'
                final=snap('dark_final',zero,method='direct',pumped=True,phase_owner=isinstance(bench.phase,PhaseOwner));comparisons['final_dark_vs_initial']=dict(mean_change=float(final.mean()-dark.mean()),different_array=not np.array_equal(final,dark))
        put(out/'report.json',dict(complete=True,outcome=outcome,signal_restored=outcome in('standard_restored','direct_or_message_pump_restored','camera_reopen_restored','persistent_phase_owner_restored'),
            comparisons=comparisons,rows=rows,raw_frames=len(rows),camera_reconnects=int('camera_reopen'in comparisons),phase_owner_changes=int('phase_owner'in comparisons),
            SDK_released=True,actual_model_TRAIN_CCD=0,physical_contract_changed=False,TEST_used=False))
    except BaseException as error:
        put(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),rows=rows,comparisons=comparisons,automatic_retry=False));raise
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(out/'release.json',dict(own_lock_released=True,closed_context=True,pid=os.getpid()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.out)


if __name__=='__main__':main()
