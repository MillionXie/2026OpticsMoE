"""One requested phase ONLY: hold, capture, prepare next inputs, then wait.

No automatic phase changes, flat challenges, SDK reconnects or fabricated phase
verification. The SDK runs on the main thread as in the vendor example. Remote
camera/GPU work runs in a worker thread. A release file cancels the remote lease
before releasing the phase. Existing captures are retained by the capture CLI.
"""
import argparse,os,subprocess,threading,time
from pathlib import Path
from dual_run import Remote,STAGES
from phase_hdmi import PhaseHDMI,load_native,sha
from phase_owner import message_pump
from phase_display import DisplayOrigin
from guarded_workflow import read,write
from phase_fingerprint import digest

ROOT=Path(__file__).resolve().parent

def next_stage(stage):
    i=STAGES.index(stage)
    return STAGES[i+1] if i+1<len(STAGES) else None

def run(a):
    link=read(a.link_config);out=a.out.resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be under results')
    out.mkdir(parents=True,exist_ok=False)
    if 'blinkhdmi.exe' in subprocess.check_output(['tasklist','/FI','IMAGENAME eq BlinkHdmi.exe','/FO','CSV'],text=True).lower():
        raise RuntimeError('Close Blink GUI before SDK ownership')
    with Remote(link) as r:
        c=r.read(a.remote_config);session=c['diagnostic_session']
        state=r.read(f'sessions/{session}/session.json')
        if digest(state['hardware_config'])!=digest(c):raise ValueError('Session configuration mismatch')
        mf=r.read(f'sessions/{session}/play/{a.stage}/manifest.json')
        if mf['hardware_identity']!=state['hardware_identity'] or mf['stage']!=a.stage:raise ValueError('Manifest mismatch')
        phase=out/(a.stage+'.bmp');r.download(mf['phase_file'].replace('\\','/'),phase)
        load_native(phase,mf['phase_sha256'])
    lock=ROOT/'results/phase_sdk_owner.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    release=out/'RELEASE';worker=None;done=threading.Event()
    report=dict(status='loading_requested_phase',stage=a.stage,session=session,count=len(mf['entries']),
        next_stage=next_stage(a.stage),phase_sha256=sha(phase),release_file=str(release),
        automatic_phase_switching=False,phase_optically_certified=False,
        source_sha256=sha(__file__),phase_sdk_thread='main',config=a.remote_config)
    def save():
        report['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S');write(out/'report.json',report)
    save()
    try:
        with DisplayOrigin(bool(link.get('phase_display_align_top',False))):
            pump=message_pump()
            with PhaseHDMI(link['phase_sdk'],link['phase_lut'],link.get('phase_settle_s',1),pixel_format=link.get('phase_pixel_format','rgba')) as sdk:
                receipt=sdk.show(phase,mf['phase_sha256'],pump=pump)
                report.update(status='holding_requested_phase',receipt=receipt);save()
                print('HOLDING ONE PHASE',a.stage,str(phase),flush=True)
                def work():
                    try:
                        with Remote(link) as r:
                            report['status']='capturing';save()
                            r.job(dict(action='capture',config=a.remote_config,session=session,stage=a.stage,phase_receipt=receipt))
                            report['capture_complete']=True;save()
                            nxt=next_stage(a.stage)
                            if release.exists():return
                            if nxt:
                                report['status']='preparing_next_inputs_without_phase_change';save()
                                r.job(dict(action='prepare',config=a.remote_config,session=session,stage=nxt))
                                report['next_manifest']=r.read(f'sessions/{session}/play/{nxt}/manifest.json')
                                report['status']='next_inputs_ready_waiting_for_user_layer_change'
                            else:
                                report['status']='evaluating';save()
                                r.job(dict(action='evaluate',config=a.remote_config,session=session))
                                r.download(f'sessions/{session}/results/metrics.json',out/'metrics.json')
                                report['status']='evaluation_complete_holding_phase'
                            save();print('FINISHED REQUESTED STAGE; PHASE UNCHANGED',a.stage,flush=True)
                    except BaseException as e:
                        report.update(status='failed_holding_phase',error=str(e));save();print('ERROR',repr(e),flush=True)
                    finally:done.set()
                # Lease abort is checked in Remote.job; never release the phase
                # while a camera worker could still be acquiring.
                original_job=Remote.job
                def cancellable_job(remote,spec):
                    original_heartbeat=remote.heartbeat
                    def heartbeat(rel):
                        if release.exists():raise RuntimeError('Manual stage release requested')
                        original_heartbeat(rel)
                    remote.heartbeat=heartbeat
                    try:return original_job(remote,spec)
                    finally:remote.heartbeat=original_heartbeat
                Remote.job=cancellable_job
                worker=threading.Thread(target=work,daemon=False);worker.start();last=time.monotonic()
                try:
                    while not release.exists() or not done.is_set():
                        pump()
                        if time.monotonic()-last>=1:sdk.repeat();last=time.monotonic()
                        time.sleep(.01)
                finally:
                    release.touch()
                    worker.join(timeout=45)
                    Remote.job=original_job
                    if worker.is_alive():raise RuntimeError('Remote shutdown not acknowledged; inspect camera lease')
                report['released']=True;save()
    finally:
        if worker and worker.is_alive():
            release.touch();worker.join(timeout=45)
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--link-config',type=Path,required=True);p.add_argument('--remote-config',required=True)
    p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
