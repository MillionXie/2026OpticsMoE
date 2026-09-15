"""One-PC, persistent phase SDK + Holoeye/SHS, guarded at optical checkpoints.

Run on the logged-in Windows desktop using pythonw (not Session0; no SW_HIDE).
No network phase lease, no GUI clicks, no VCom/ramp/firmware writes.
"""
import os
from pathlib import Path
import sys
import time
import json
import subprocess
import queue
import threading
from types import SimpleNamespace
import numpy as np
from PIL import Image
from . import lab_bench as bench
from .lab_runtime import STAGES
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read,write,sha
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import stage_config,raster,verified_ccd


_PHASE_WORKER = r'''
import sys,json,traceback
sys.path.insert(0,sys.argv[1])
from phase_owner import PhaseOwner
def reply(value):
    print('PHASE_IPC:'+json.dumps(value),flush=True)
try:
    cfg=json.loads(sys.stdin.readline())
    with PhaseOwner(cfg['config'],cfg['flat'],cfg['lens']) as owner:
        reply(dict(ok=True,info=owner.info,display=owner.display.audit))
        for line in sys.stdin:
            msg=json.loads(line)
            if msg['action']=='close':break
            if msg['action']!='show':raise ValueError('Unknown phase command')
            receipt=owner.show(msg['path'],msg.get('sha256'))
            reply(dict(ok=True,receipt=receipt))
    reply(dict(ok=True,closed=True))
except BaseException:
    reply(dict(ok=False,error=traceback.format_exc()))
    raise
'''


class ProcessPhaseOwner:
    """Isolate vendor graphics/DLL state from Holoeye and CUDA on this PC.

    Normal desktop window visibility is retained; no SW_HIDE/CREATE_NO_WINDOW.
    A pipe acknowledgement is still NOT proof of optical correctness.
    """
    def __init__(self,config,flat,lens,module_root):
        self.config=config;self.flat=flat;self.lens=lens;self.module_root=module_root
        self.process=None;self.messages=queue.Queue()
    def _receive(self,timeout=75):
        try:message=self.messages.get(timeout=timeout)
        except queue.Empty:raise RuntimeError('Phase subprocess response timed out')
        if not message.get('ok'):raise RuntimeError('Phase subprocess: '+message.get('error','closed unexpectedly'))
        return message
    def _reader(self):
        for line in self.process.stdout:
            if line.startswith('PHASE_IPC:'):
                try:self.messages.put(json.loads(line[len('PHASE_IPC:'):]))
                except ValueError:self.messages.put(dict(ok=False,error='Invalid phase IPC response'))
            else:print('[phase process] '+line.rstrip(),flush=True)
        self.messages.put(dict(ok=False,error='Phase process EOF'))
    def _send(self,value):
        self.process.stdin.write(json.dumps(value)+'\n');self.process.stdin.flush()
    def __enter__(self):
        self.process=subprocess.Popen([sys.executable,'-u','-c',_PHASE_WORKER,str(self.module_root)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1)
        threading.Thread(target=self._reader,daemon=True).start()
        try:
            self._send(dict(config=self.config,flat=str(self.flat),lens=str(self.lens)))
            ready=self._receive();self.info=ready['info'];self.info['sdk_host_pid']=self.process.pid
            self.info['sdk_process_isolated']=True;self.display=SimpleNamespace(audit=ready['display'])
            return self
        except BaseException:self.close();raise
    def show(self,path,expected_sha=None):
        self._send(dict(action='show',path=str(path),sha256=expected_sha))
        return self._receive()['receipt']
    def close(self):
        if self.process is None:return
        if self.process.poll() is None:
            try:self._send(dict(action='close'))
            except (BrokenPipeError,OSError):pass
            try:self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                # Stop only our child if graceful SDK shutdown hangs. Keep any
                # stale owner lock for explicit inspection, never bypass it.
                self.process.terminate();self.process.wait(timeout=10)
                raise RuntimeError('Phase child forced to stop after graceful close timed out; inspect SDK lock')
        self.process.stdin.close();self.process.stdout.close()
    def __exit__(self,*args):self.close()


def pcc(a,b):
    x=np.asarray(a,dtype=np.float64).ravel(); y=np.asarray(b,dtype=np.float64).ravel()
    x=x-x.mean(); y=y-y.mean(); den=np.linalg.norm(x)*np.linalg.norm(y)
    if den<1e-12:return 0.0
    return float(x@y/den)


def quality(im):
    return dict(p99=float(np.percentile(im,99)),std=float(im.std()),saturation=float((im>=255).mean()))


def gate(im):
    q=quality(im)
    if (q['p99']<=8 and q['std']<1.5) or q['saturation']>.01:
        raise RuntimeError('Near-dark/saturated CCD; no threshold bypass: '+str(q))
    return q


def patterns(folder,c):
    folder.mkdir(parents=True,exist_ok=True)
    white=folder/'A_white.bmp'; flat=folder/'P_flat.bmp'; lens=folder/'P_lens10cm.bmp'
    Image.fromarray(raster(np.ones((478,478),np.float32),c['amplitude_slm'],'amplitude')).save(white)
    yy,xx=np.indices((478,478)); radius=((xx-238.5)*17e-6)**2+((yy-238.5)*17e-6)**2
    Image.fromarray(raster(np.zeros((478,478)),c['phase_slm'],'phase')).save(flat)
    Image.fromarray(raster(-np.pi*radius/(532e-9*.1),c['phase_slm'],'phase')).save(lens)
    return white,flat,lens


def warp_matrix(c):
    import cv2
    pts=np.float32([c['logical_corners_full_sensor_xy'][k] for k in ('top_left','top_right','bottom_right','bottom_left')])
    return cv2.getPerspectiveTransform(pts,np.float32([[-.5,-.5],[477.5,-.5],[477.5,477.5],[-.5,477.5]]))


def get_roi(hw,bmp,H):
    import cv2
    frame,meta=hw.capture(bmp)
    if meta.get('incomplete',False):raise RuntimeError('Incomplete camera frame')
    im=np.rint(np.clip(cv2.warpPerspective(frame.astype(np.float32),H,(478,478)),0,255)).astype(np.uint8)
    return im,meta


def phase_check(owner,hw,H,white,flat,target,folder,reference=None):
    """Physical repeated-image test, not mere SDK SHA/return acknowledgement."""
    folder.mkdir(parents=True,exist_ok=True)
    from capture import snapshot
    exposure=hw.camera.get('ExposureTime')
    rows=[]; images=[]
    try:
        probe_us=float(hw.c.get('phase_probe_exposure_us',150.0))
        if not 20<=probe_us<=1600:raise ValueError('Probe exposure must be 20..1600 us')
        hw.camera.set('ExposureTime',probe_us)
        hw.camera_settings=snapshot(hw.camera)
        # First upload on a newly opened Holoeye connection can lag its visible
        # callback. Warm up using the same optical probe while draining frames.
        # This cost is per connection, not per experimental sample.
        if not getattr(hw,'_openmoji_optical_warmed',False):
            owner.show(flat,sha(flat))
            started=time.monotonic();warmup=[]
            duration=float(hw.c.get('connection_optical_warmup_s',10.0))
            if not 0<=duration<=30:raise ValueError('Connection warmup must be 0..30 s')
            while time.monotonic()-started<duration:
                image,meta=get_roi(hw,white,H)
                warmup.append(dict(elapsed_s=time.monotonic()-started,quality=quality(image),frame_id=meta['frame_id']))
            write(folder/'connection_warmup.json',dict(duration_s=duration,frames=warmup))
            hw._openmoji_optical_warmed=True
        for i,path in enumerate((flat,target,target)):
            receipt=owner.show(path,sha(path))
            time.sleep(1)
            if hw.c.get('diagnostic_save_sensor',False):
                import cv2
                raw,meta=hw.capture(white)
                Image.fromarray(raw).save(folder/f'{i}_sensor.png')
                meta['full_sensor_quality']=quality(raw)
                im=np.rint(np.clip(cv2.warpPerspective(raw.astype(np.float32),H,(478,478)),0,255)).astype(np.uint8)
            else:
                im,meta=get_roi(hw,white,H)
            Image.fromarray(im).save(folder/f'{i}.png')
            rows.append(dict(phase=receipt,camera=meta,quality=quality(im)))
            images.append(im)
        cross=pcc(images[0],images[1]); repeat=pcc(images[1],images[2])
        ref=pcc(reference,images[2]) if reference is not None else None
        report=dict(flat_target_pcc=cross,target_repeat_pcc=repeat,target_reference_pcc=ref,frames=rows,
                    thresholds=dict(max_flat_target_pcc=.99,min_repeat_pcc=.95,min_reference_pcc=.95),passed=False)
        write(folder/'report.json',report)
        for im in images:gate(im)
        if cross>=.99 or repeat<.95 or (ref is not None and ref<.95):
            raise RuntimeError('Optical phase check failed; evidence retained: '+str(folder))
        report['passed']=True;write(folder/'report.json',report)
        return images[2],report
    finally:
        hw.camera.set('ExposureTime',float(exposure))
        hw.camera_settings=snapshot(hw.camera)


def run(a):
    root=Path(a.project).resolve(); c=bench.config(a.config)
    out=bench.session(root,a.session)
    if a.action=='auto' and (out/'status.json').exists():
        prior=read(out/'status.json')
        if prior.get('status')=='stopped' and list((out/'ccd').rglob('*.record.json')):
            raise RuntimeError('Stopped session contains CCDs: audit/quarantine suspect last batch before resuming; no automatic acceptance')
    if a.action=='auto' and not out.exists():bench.initialize(a)
    if a.action=='probe':out.mkdir(parents=True,exist_ok=True)
    phase_cfg=read(a.phase_config)
    sys.path.insert(0,str(Path(a.bench_root).resolve()))
    sys.path.insert(0,str(root/'runtime/phase_control'))
    from phase_owner import PhaseOwner
    from slm_camera import Controller
    white,flat,lens=patterns(out/'checks/patterns',c);H=warp_matrix(c)
    lock=Path(a.bench_root).resolve()/'results/dual_jobs/ACTIVE.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    def stop():
        if (out/'STOP').exists():raise RuntimeError('User STOP requested; existing CCD retained')
    try:
        write(out/'status.json',dict(status='starting_sdk',pid=os.getpid()))
        phase_host=ProcessPhaseOwner(phase_cfg,flat,lens,root/'runtime/phase_control') if c.get('phase_process_isolated',True) else PhaseOwner(phase_cfg,flat,lens)
        with phase_host as owner:
            write(out/'phase_devices.json',dict(phase=owner.info,display=owner.display.audit))
            probe_c=dict(c)
            if a.action=='probe':probe_c['diagnostic_save_sensor']=True
            with Controller(probe_c) as hw:
                _,report=phase_check(owner,hw,H,white,flat,lens,out/'checks/startup')
                write(out/'devices.json',dict(phase=owner.info,amplitude_camera=hw.info,display=owner.display.audit))
            if a.action=='probe':
                write(out/'status.json',dict(status='probe_passed',physical_phase_check=report));return
            for idx,stage in enumerate(STAGES):
                stop();a.stage=stage
                _,_,state,_,_=bench.open_session(a)
                if stage in state['measured_stages']:
                    bench.audit(a);continue
                write(out/'status.json',dict(status='preparing',stage=stage))
                bench.prepare(a)
                mf=read(out/'play'/stage/'manifest.json');target=out/mf['phase_file']
                with Controller(stage_config(c,stage,STAGES)) as hw:
                    reference,check=phase_check(owner,hw,H,white,flat,target,out/'checks'/stage/'start')
                    for i,e in enumerate(mf['entries'],1):
                        stop();p=out/'ccd'/stage/(e['key']+'.png');recpath=p.with_suffix('.record.json')
                        if recpath.exists():
                            _,old=verified_ccd(out,stage,e['key'])
                            if any(old[k]!=v for k,v in dict(amplitude_sha256=e['sha256'],phase_sha256=mf['phase_sha256'],upstream_ccd_sha256=e['upstream_ccd_sha256'],hardware_sha256=state['hardware_sha256']).items()):
                                raise ValueError('Existing CCD identity mismatch')
                            continue
                        for prev,digest in e['upstream_ccd_sha256'].items():
                            _,old=verified_ccd(out,prev,e['key'])
                            if old['sha256']!=digest:raise ValueError('Upstream changed')
                        bmp=out/'play'/stage/e['bmp']
                        if sha(bmp)!=e['sha256']:raise ValueError('Input BMP changed')
                        im,meta=get_roi(hw,bmp,H)
                        try:q=gate(im)
                        except RuntimeError:
                            rejected=out/'rejected'/stage;rejected.mkdir(parents=True,exist_ok=True)
                            Image.fromarray(im).save(rejected/(e['key']+'.png'));raise
                        p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(im).save(p,compress_level=1)
                        write(recpath,dict(sha256=sha(p),amplitude_sha256=e['sha256'],phase_sha256=mf['phase_sha256'],
                            upstream_ccd_sha256=e['upstream_ccd_sha256'],hardware_sha256=state['hardware_sha256'],
                            quality=q,camera=meta,raw_saved=False,phase_confirmation='same_pc_persistent_owner_with_periodic_optical_checks'))
                        write(out/'status.json',dict(status='capturing',stage=stage,completed=i,total=len(mf['entries']),quality=q))
                        print('CAPTURED',stage,i,'/',len(mf['entries']),q,flush=True)
                        if i%50==0:
                            phase_check(owner,hw,H,white,flat,target,out/'checks'/stage/f'after_{i:05d}',reference)
                    phase_check(owner,hw,H,white,flat,target,out/'checks'/stage/'end',reference)
                state['measured_stages']=list(STAGES[:idx+1]);write(out/'session.json',state);bench.audit(a)
            write(out/'status.json',dict(status='evaluating'))
            bench.evaluate(a)
            write(out/'status.json',dict(status='complete',result='results.json'))
    except BaseException as exc:
        write(out/'status.json',dict(status='stopped',error=str(exc),existing_ccd_retained=True));raise
    finally:
        lock.unlink(missing_ok=True)
