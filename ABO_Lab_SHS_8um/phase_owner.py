"""One persistent, message-pumped vendor SDK owner. NOT an optical acknowledgement.

Vendor-example RGBA retained buffer, repeated on owner/message-pump thread.
No explicit VCom/ramp/firmware calls; Create_SDK itself has vendor side effects.
"""
import ctypes as C
from ctypes import wintypes as W
from concurrent.futures import Future
import queue
import threading
import time
import subprocess
import os
from pathlib import Path
from phase_hdmi import PhaseHDMI, load_native
from phase_display import DisplayOrigin


def message_pump():
    u=C.WinDLL('user32',use_last_error=True)
    u.PeekMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT,W.UINT];u.PeekMessageW.restype=W.BOOL
    u.TranslateMessage.argtypes=[C.POINTER(W.MSG)];u.TranslateMessage.restype=W.BOOL
    u.DispatchMessageW.argtypes=[C.POINTER(W.MSG)];u.DispatchMessageW.restype=C.c_ssize_t
    msg=W.MSG()
    def pump():
        for _ in range(1000):
            if not u.PeekMessageW(C.byref(msg),None,0,0,1):break
            if msg.message==0x12:raise RuntimeError('SDK window received WM_QUIT')
            u.TranslateMessage(C.byref(msg));u.DispatchMessageW(C.byref(msg))
    return pump


class PhaseOwner:
    def __init__(self,link,black,lens):
        self.link=link;self.black=Path(black);self.lens=Path(lens)
        self.q=queue.Queue();self.ready=Future();self.stop=threading.Event();self.error=None
        self.audit=[];self.thread=None
        self.lock=None;self.display=DisplayOrigin(bool(link.get('phase_display_align_top',False)))
    def __enter__(self):
        load_native(self.black);load_native(self.lens)
        tasks=subprocess.check_output(['tasklist','/FI','IMAGENAME eq BlinkHdmi.exe','/FO','CSV'],text=True)
        if 'blinkhdmi.exe' in tasks.lower():raise RuntimeError('Close Blink GUI before SDK ownership')
        lock=Path(__file__).resolve().parent/'results/phase_sdk_owner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        try:os.write(fd,str(os.getpid()).encode())
        finally:os.close(fd)
        self.lock=lock
        try:
            self.display.__enter__()
            self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
            self.info=self.ready.result(timeout=60)
            self.recover(int(self.link.get('phase_startup_cycles',0)))
            return self
        except BaseException:self.close();raise
    def _loop(self):
        current=None
        try:
            pump=message_pump()
            with PhaseHDMI(self.link['phase_sdk'],self.link['phase_lut'],self.link.get('phase_settle_s',1),
                           pixel_format=self.link.get('phase_pixel_format','rgba')) as phase:
                last=0;current_path=self.black
                def show(path,expected=None):
                    nonlocal last,current_path
                    result=phase.show(path,expected,pump=pump);last=time.monotonic();current_path=path
                    result['optical_display_verified_by_this_call']=False
                    result['repeat_interval_s']=1
                    return result
                self.ready.set_result(phase.info)
                try:
                    while not self.stop.is_set():
                        pump()
                        if phase.pixels is not None and time.monotonic()-last>=1:
                            phase.repeat();last=time.monotonic()
                        try:kind,arg,current=self.q.get(timeout=.01)
                        except queue.Empty:continue
                        if kind=='show':result=show(*arg)
                        elif kind=='recover':
                            result=[]
                            for i in range(arg):
                                if self.stop.is_set():raise RuntimeError('Recovery cancelled')
                                result.append(show(current_path))
                                if (i+1)%10==0:print(f'Phase startup/recovery {i+1}/{arg}',flush=True)
                            self.audit.append({'kind':'reassert_current_retained_buffer','writes':result})
                        else:raise ValueError('Unknown SDK command')
                        current.set_result(result);current=None
                finally:
                    self.audit.append({'kind':'final_flat_write','receipt':show(self.black)})
        except BaseException as e:
            self.error=e
            if not self.ready.done():self.ready.set_exception(e)
            if current is not None and not current.done():current.set_exception(e)
    def _call(self,kind,arg,timeout):
        if not self.thread or not self.thread.is_alive():raise RuntimeError(f'SDK owner stopped: {self.error}')
        f=Future();self.q.put((kind,arg,f));return f.result(timeout=timeout)
    def show(self,path,expected_sha=None):
        load_native(path,expected_sha)
        return self._call('show',(path,expected_sha),30)
    def recover(self,cycles=4):
        if not 0<=cycles<=30:raise ValueError('Recovery cycles must be 0..30')
        return self._call('recover',cycles,180)
    def restart(self):
        """Bounded caller-controlled SDK reconnect; retain monitor origin/lock.

        This repeats vendor Create_SDK, with its normal internal side effects.
        No explicit channel/VCom/ramp/firmware writes. Not an optical proof.
        """
        self.stop.set()
        if self.thread:self.thread.join(timeout=30)
        if self.thread and self.thread.is_alive():raise RuntimeError('Old SDK owner did not stop; no second owner started')
        self.audit.append({'kind':'sdk_reconnect_without_display_layout_change','previous_error':str(self.error) if self.error else None})
        self.stop=threading.Event();self.q=queue.Queue();self.ready=Future();self.error=None
        time.sleep(1)
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
        self.info=self.ready.result(timeout=60)
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=30)
        if self.thread and self.thread.is_alive():raise RuntimeError('SDK owner did not exit; do not start another owner')
        try:self.display.close()
        finally:
            if self.lock:self.lock.unlink(missing_ok=True);self.lock=None
        if self.error:raise self.error
    def __exit__(self,*args):self.close()
