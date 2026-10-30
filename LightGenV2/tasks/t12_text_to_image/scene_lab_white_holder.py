"""Finite SLM-only white-field holder for human Mosaic comparison, no camera SDK."""
import argparse,json,os,time
from contextlib import ExitStack
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from PIL import Image
from .scene_lab_protocol import put,sha
from .scene_lab_hardware import native


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=300);a=p.parse_args()
    if not 30<=a.seconds<=300:raise ValueError('Finite holder must be30–300seconds')
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[3]
    lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Other SLM owner')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out),scope='SLMs_only',camera_SDK_owned=False,manual_Mosaic_preview_allowed=True)).encode());os.close(fd)
    reason='finite_deadline'
    try:
        for path,digest in cfg['fixed_files'].items():
            if sha(path)!=digest:raise ValueError('Device/calibration identity changed')
        amp=a.out/'amplitude.bmp';phase=a.out/'phase.bmp';zero=np.zeros((478,478),np.uint8)
        Image.fromarray(native(np.full_like(zero,255),'amplitude')).save(amp);Image.fromarray(native(zero,'phase')).save(phase)
        if sha(amp)!='4f60cc426d04a433ba30571eca9df128b7f954d90262318adbdca5a5ea9dd193'or sha(phase)!='b4a6b5b0a9ce7428c3496bdc5325f8e0796d6b033bbb5115d26bb35602b1a65a':raise ValueError('Not exact original white/flat BMPs')
        from experiments.hardware_sdk.devices import build_slm
        from LightGenV2.hardware_common.shs.phase_owner import PhaseOwner,message_pump
        m=json.loads(Path(cfg['machine_config']).read_text(encoding='utf-8-sig'));m['amplitude_slm']['sdk_path']=cfg['amplitude_sdk']
        link=dict(phase_sdk=cfg['phase_sdk'],phase_lut=cfg['phase_lut'],phase_settle_s=.8,phase_pixel_format='rgba',phase_startup_cycles=0,phase_display_align_top=False)
        with ExitStack()as stack:
            owner=stack.enter_context(PhaseOwner(link,phase,phase));slm=stack.enter_context(build_slm(m['amplitude_slm'],Path(cfg['machine_config']).parent));pump=message_pump()
            phase_receipt=owner.show(phase);slm.preload_files([amp]);slm.display_file(amp)
            started=datetime.now(timezone.utc);until=time.monotonic()+a.seconds
            put(a.out/'ready.json',dict(SLMs_held=True,camera_SDK_opened=False,manual_Mosaic_preview_allowed=True,started_UTC=started.isoformat(),deadline_UTC=(started+timedelta(seconds=a.seconds)).isoformat(),
                holder_seconds=a.seconds,amplitude_sha256=sha(amp),phase_sha256=sha(phase),source_sha256=sha(__file__),SLM=slm.device_info(),phase=phase_receipt,display_origin_changed=False,recurring=False,
                requested_manual_camera_settings=dict(exposure_us=1999.8,gain=0,auto_exposure=False),physical_light_proof=False))
            print('Finite SLM white holder ready; camera SDK is not opened.',flush=True)
            while time.monotonic()<until:
                pump();time.sleep(.05)
                if (a.out/'stop.request').exists():reason='explicit_stop_request';break
        put(a.out/'report.json',dict(complete=True,reason=reason,SLM_contexts_closed=True,camera_SDK_opened=False,source_sha256=sha(__file__),physical_light_verified=False))
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(pid=os.getpid(),own_lock_released=True,SLM_contexts_closed=True,camera_SDK_opened=False,reason=reason,ended_UTC=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
