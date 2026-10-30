"""Read-only camera modes/display inventory after a bounded failed SDK canary."""
import argparse,ctypes,json,os,re
from pathlib import Path
from .scene_lab_protocol import put,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--display-only',action='store_true');a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3];lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Another owner')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out))).encode());os.close(fd)
    result=dict(read_only_modes=True,SLM_SDK_opened=False,display_configuration_changed=False,saved_raw_frames=0,TEST_used=False)
    try:
        import win32api,win32con
        displays=[]
        for i in range(32):
            try:d=win32api.EnumDisplayDevices(None,i)
            except Exception:break
            if not d.StateFlags&1:continue
            m=win32api.EnumDisplaySettings(d.DeviceName,win32con.ENUM_CURRENT_SETTINGS);monitors=[]
            for k in range(8):
                try:
                    q=win32api.EnumDisplayDevices(d.DeviceName,k);monitors.append(dict(name=q.DeviceName,label=q.DeviceString,ID=q.DeviceID,flags=q.StateFlags))
                except Exception:break
            displays.append(dict(device=d.DeviceName,label=d.DeviceString,flags=d.StateFlags,x=m.Position_x,y=m.Position_y,width=m.PelsWidth,height=m.PelsHeight,Hz=m.DisplayFrequency,monitors=monitors))
        result['displays']=displays
        if a.display_only:
            import win32ts
            result.update(interactive_session=win32ts.ProcessIdToSessionId(os.getpid()),camera_SDK_opened=False,SDK_closed=True)
            put(a.out/'report.json',result);print(json.dumps(result));return
        from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
        with TucamCamera(cfg['camera_sdk'],auto_exposure=None,exposure_us=None,analog_gain=None,warmup_frames=3,discard_frames_after_display=6,
                         saved_frame_size_wh=None,saved_frame_resize_mode='none',saved_frame_bit_depth=None)as camera:
            result['camera']=camera.device_info();module=camera._module
            pattern=r'TRIG|SHUT|TEST|DARK|DSNU|FPN|GAIN|EXPOS|MIRROR|FLIP|RESOL|BLACK|HDR|BIN|MODE|TEMPER'
            result['capabilities']={member.name:camera._get_capability(member)for member in module.TUCAM_IDCAPA if re.search(pattern,member.name)}
            result['properties']={member.name:camera._get_property(member)for member in module.TUCAM_IDPROP if re.search(pattern,member.name)}
            if hasattr(module,'TUCAM_Cap_GetTrigger')and hasattr(module,'TUCAM_TRIGGER_ATTR'):
                try:
                    trigger=module.TUCAM_TRIGGER_ATTR();camera._check(module.TUCAM_Cap_GetTrigger(camera._handle(),ctypes.pointer(trigger)),'read trigger only')
                    result['trigger']={name:getattr(trigger,name)for name,*_ in trigger._fields_}
                except Exception as error:result['trigger_read_error']=repr(error)
        wrappers=[]
        for file in Path(cfg['amplitude_sdk']).rglob('*.py'):
            if len(file.relative_to(cfg['amplitude_sdk']).parts)>4:continue
            text=file.read_text(errors='replace')
            if 'class SLMInstance'in text:
                lines=text.splitlines();excerpts=[]
                for i,line in enumerate(lines):
                    if re.search(r'^\s+def (open|__init__|showDataFromFile|showDatahandle|.*[Dd]isplay.*|.*[Dd]evice.*)\(',line):excerpts.append(dict(line=i+1,text='\n'.join(lines[i:i+25])))
                wrappers.append(dict(path=str(file),SHA256=sha(file),excerpts=excerpts))
        result['amplitude_wrapper_public_readback']=wrappers
        result['SDK_closed']=True;put(a.out/'report.json',result)
        print(json.dumps({k:v for k,v in result.items()if k!='amplitude_wrapper_public_readback'}))
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(own_lock_released=True,closed_context=True,pid=os.getpid()))


if __name__=='__main__':main()
