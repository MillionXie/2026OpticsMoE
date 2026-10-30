"""Camera-only subprocess transport for the T12 bench; no SLM SDK in worker.

RPC transports file identities and metadata, not simulated detector values.
The original TucamCamera is used without camera-parameter writes.
"""
import argparse,json,os,queue,subprocess,sys,threading,traceback
from pathlib import Path

PREFIX='T12_CAMERA_RPC '


def checked_frame_path(value,root):
    path=Path(value).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root)or path.suffix!='.npy' or path.exists():raise ValueError('Capture must create a new npy inside the owned run')
    path.parent.mkdir(parents=True,exist_ok=True);return path


class CameraProcess:
    def __init__(self,sdk_path,*,allowed_root,**kwargs):
        self.sdk_path=str(sdk_path);self.allowed_root=Path(allowed_root).resolve();self.kwargs=kwargs;self.process=None;self.responses=queue.Queue();self.serial=0;self.info={};self.log=None;self.worker_pid=None
    def _read(self):
        for line in self.process.stdout:
            index=line.find(PREFIX)
            if index>=0:
                try:self.responses.put(json.loads(line[index+len(PREFIX):]))
                except json.JSONDecodeError:pass
        self.responses.put(dict(transport_EOF=True))
    def _call(self,op,timeout=30,**data):
        self.serial+=1;number=self.serial;self.process.stdin.write(json.dumps(dict(id=number,op=op,**data))+'\n');self.process.stdin.flush()
        try:reply=self.responses.get(timeout=timeout)
        except queue.Empty:raise TimeoutError('Camera-only worker RPC timed out')
        if reply.get('transport_EOF'):raise RuntimeError('Camera-only worker exited; inspect its stderr log')
        if reply.get('id')!=number or not reply.get('ok'):raise RuntimeError(('Camera-only worker failure',reply))
        return reply['result']
    def open(self):
        self.allowed_root.mkdir(parents=True,exist_ok=True);self.log=(self.allowed_root/'camera_worker_stderr.log').open('x',encoding='utf-8')
        root=Path(__file__).resolve().parents[3]
        self.process=subprocess.Popen([sys.executable,'-u','-m','LightGenV2.tasks.t12_text_to_image.scene_lab_camera_process','--worker'],cwd=root,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        threading.Thread(target=self._read,daemon=True).start()
        try:
            result=self._call('open',timeout=60,sdk_path=self.sdk_path,allowed_root=str(self.allowed_root),expected_parent_pid=os.getpid(),kwargs=self.kwargs)
            self.info=result['device'];self.worker_pid=result['worker_pid']
        except BaseException:self.close();raise
    def capture(self,path):
        result=self._call('capture',path=str(Path(path).resolve()));self.info=result['device'];return result
    def device_info(self):return dict(self.info,camera_process_isolated=True,camera_worker_PID=self.worker_pid)
    def live_settings(self):return self._call('info')['actual']
    def close(self):
        try:
            if self.process is not None:
                if self.process.poll()is None:
                    try:self._call('close',timeout=15)
                    except(Exception,):
                        if self.process.poll()is None:self.process.terminate()
                self.process.wait(timeout=15)
                for stream in(self.process.stdin,self.process.stdout):
                    if stream:stream.close()
        finally:
            if self.log:self.log.close();self.log=None
    def __enter__(self):self.open();return self
    def __exit__(self,*args):self.close()


def worker():
    camera=None;request={}
    def reply(result):print(PREFIX+json.dumps(dict(id=request['id'],ok=True,result=result)),flush=True)
    def status():
        module=camera._module
        actual=dict(exposure_us=camera._get_property(module.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000,
            gain=camera._get_property(module.TUCAM_IDPROP.TUIDP_GLOBALGAIN),auto=camera._get_capability(module.TUCAM_IDCAPA.TUIDC_ATEXPOSURE))
        return dict(device=camera.device_info(),actual=actual,worker_pid=os.getpid(),SLM_SDK_initialized=False)
    try:
        for line in sys.stdin:
            request=json.loads(line);op=request['op']
            if op=='open':
                if camera is not None:raise ValueError('Camera already open')
                import psutil
                if request['expected_parent_pid']not in{p.pid for p in psutil.Process().parents()}:raise ValueError('Unexpected camera-worker parent')
                root=Path(request['allowed_root']).resolve();hardware=Path(__file__).resolve().parent/'runs/hardware'
                if not root.is_relative_to(hardware):raise ValueError('Owned hardware run required')
                kwargs=request['kwargs']
                if any(kwargs.get(k)is not None for k in('auto_exposure','exposure_us','analog_gain','device_roi_xywh','resolution_mode')):raise ValueError('Parameter writes are forbidden in isolated capture')
                from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
                camera=TucamCamera(request['sdk_path'],**kwargs);camera.open();reply(status())
            elif op=='capture':
                if camera is None:raise ValueError('Camera not open')
                path=checked_frame_path(request['path'],root);camera.capture(path);reply(status())
            elif op=='info':reply(status())
            elif op=='close':
                if camera is not None:camera.close();camera=None
                reply(dict(closed=True));break
            else:raise ValueError('Unknown camera RPC')
    except BaseException as error:
        print(PREFIX+json.dumps(dict(id=request.get('id'),ok=False,error=repr(error))),flush=True);traceback.print_exc();raise
    finally:
        if camera is not None:camera.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true',required=True);p.parse_args();worker()


if __name__=='__main__':main()
