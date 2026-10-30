"""Two-frame diagnostic: SDK-owned window rectangles, not screen manipulation."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from .scene_lab_hardware import Bench,native
from .scene_lab_protocol import put,sha
from .scene_lab_sdk_canary import signal_pair


def owned_windows():
    import psutil,win32gui,win32process
    pids={os.getpid(),*(p.pid for p in psutil.Process().children(recursive=True))};rows=[]
    def collect(handle,_):
        tid,pid=win32process.GetWindowThreadProcessId(handle)
        if pid in pids:rows.append(dict(pid=pid,thread=tid,title=win32gui.GetWindowText(handle),class_name=win32gui.GetClassName(handle),rectangle=list(win32gui.GetWindowRect(handle)),visible=bool(win32gui.IsWindowVisible(handle)),minimized=bool(win32gui.IsIconic(handle))))
    win32gui.EnumWindows(collect,None);return rows


def main():
    from LightGenV2.hardware_common.shs.phase_owner import message_pump
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    root=Path(__file__).resolve().parents[3];lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    if any(q.exists()for q in [root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']):raise RuntimeError('Other owner')
    a.out.mkdir(parents=True,exist_ok=False);fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(a.out))).encode());os.close(fd)
    try:
        windows={};frames={};rows=[];zero=np.zeros((478,478),np.uint8)
        with Bench(cfg,a.out)as bench:
            pump=message_pump();windows['after_SDK_open']=owned_windows()
            for tag,amp in [('dark',zero),('white',np.full_like(zero,255))]:
                folder=a.out/tag;folder.mkdir();ap=folder/'amplitude.bmp';pp=folder/'phase.bmp'
                Image.fromarray(native(amp,'amplitude')).save(ap);Image.fromarray(native(zero,'phase')).save(pp)
                phase=bench.phase.show(pp,pump=pump);bench.slm.preload_files([ap]);bench.slm.display_file(ap)
                until=time.monotonic()+1
                while time.monotonic()<until:pump();time.sleep(.01)
                windows[tag]=owned_windows()
                handles=[dict(path=str(path),state=int(h.state),error_code=int(h.errorCode),canvas=int(h.canvas))for path,h in bench.slm._handles.items()]
                file=folder/'native.npy';bench.camera.capture(file);raw=np.load(file,allow_pickle=False);frames[tag]=raw
                row=dict(tag=tag,raw_sha256=sha(file),amplitude_sha256=sha(ap),phase_sha256=sha(pp),data_handles=handles,
                         native_mean=float(raw.mean()),native_p99=float(np.percentile(raw,99)),camera=bench.camera.device_info(),phase=phase)
                put(folder/'receipt.json',row);rows.append(row)
        put(a.out/'report.json',dict(complete=True,SDK_released=True,own_windows=windows,rows=rows,comparison=signal_pair(frames['dark'],frames['white']),
            windows_moved=False,display_origin_changed=False,model_TRAIN_CCD=0,max_raw_frames=2,TEST_used=False))
        print(json.dumps(dict(windows=windows,comparison=signal_pair(frames['dark'],frames['white']),SDK_released=True)))
    finally:
        assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();put(a.out/'release.json',dict(pid=os.getpid(),own_lock_released=True,closed_context=True))


if __name__=='__main__':main()
