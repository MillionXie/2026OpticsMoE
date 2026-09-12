"""Bounded native SLM display / continuous-camera isolation, not formal timing."""
import argparse,ctypes as C,json,time
from ctypes import wintypes as W
from pathlib import Path
import numpy as np
from PIL import Image
from slm_camera import Controller
from capture import save_json


def renderer_windows():
    u=C.windll.user32;k=C.windll.kernel32
    k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenProcess.restype=W.HANDLE
    k.QueryFullProcessImageNameW.argtypes=[W.HANDLE,W.DWORD,W.LPWSTR,C.POINTER(W.DWORD)]
    k.CloseHandle.argtypes=[W.HANDLE]
    u.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
    u.GetWindowRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
    rows=[]
    cb=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
    def visit(hwnd,param):
        pid=W.DWORD();u.GetWindowThreadProcessId(hwnd,C.byref(pid))
        h=k.OpenProcess(0x1000,False,pid.value)
        if not h:return True
        try:
            buf=C.create_unicode_buffer(2048);size=W.DWORD(2048)
            if k.QueryFullProcessImageNameW(h,0,buf,C.byref(size)) and 'holoeye' in buf.value.lower():
                rect=W.RECT();u.GetWindowRect(hwnd,C.byref(rect))
                rows.append(dict(pid=pid.value,exe=buf.value,visible=bool(u.IsWindowVisible(hwnd)),
                                 rect=[rect.left,rect.top,rect.right,rect.bottom]))
        finally:k.CloseHandle(h)
        return True
    callback=cb(visit);u.EnumWindows(callback,0)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--method',choices=['native','handle','file'],default='native')
    p.add_argument('--config',default='LAB.local.json');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    c['camera']['exposure_us']=100;c['camera']['frame_rate_hz']=100
    report={'complete':False,'frames':[],'method':a.method,'hold_s':3}
    with Controller(c) as hw:
        report['devices']=hw.info;slm=hw.slm._slm
        report['renderer_windows']=renderer_windows()
        checker=Path(__file__).resolve().parent/'generated/phase_inverted/cal/A_CHECK_64.bmp'
        files={'checker':checker}
        for name,g in (('black',0),('white',255)):
            files[name]=a.out/(name+'.bmp');Image.fromarray(np.full((1080,1920),g,np.uint8)).save(files[name])
        for index,label in enumerate(('black','white','checker','black')):
            if a.method=='handle':
                hw.slm.preload_files([files[label]]);hw.slm.display_file(files[label])
            else:
                result=slm.showDataFromFile(str(files[label])) if a.method=='file' or label=='checker' else slm.showBlankscreen(0 if label=='black' else 255)
                hw.slm._check(result,'direct '+label)
            started=time.perf_counter();next_record=0;count=0
            if a.method!='native':
                time.sleep(.2);early,early_meta=hw.camera.fresh()
                Image.fromarray(early).save(a.out/(f'{index}_{label}_early.png'))
                report.setdefault('early',[]).append(dict(pattern=label,elapsed_s=time.perf_counter()-started,
                    mean=float(early.mean()),frame_id=early_meta['frame_id']))
            while time.perf_counter()-started<3:
                frame,meta=hw.camera.grab();count+=1
                elapsed=time.perf_counter()-started
                if elapsed>=next_record:
                    report['frames'].append(dict(pattern=label,elapsed_s=elapsed,frame_id=meta['frame_id'],
                        mean=float(frame.mean()),p99=float(np.percentile(frame,99)),max=int(frame.max())))
                    next_record+=.25
            Image.fromarray(frame).save(a.out/(f'{index}_{label}.png'))
            print(label,'last mean',float(frame.mean()),'drained frames',count,flush=True)
            save_json(a.out/'report.json',report)
        report['renderer_windows_after']=renderer_windows()
        report['complete']=True
        # Keep the last checker until the normal SDK context closes; no phase access.
    report['camera_restored']=True;save_json(a.out/'report.json',report)


if __name__=='__main__':main()
