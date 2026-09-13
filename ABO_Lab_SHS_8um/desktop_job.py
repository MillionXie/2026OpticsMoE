"""Allowlisted desktop worker for the local phase coordinator. No shell commands.

Camera jobs require a heartbeat from the phase-owning computer. If the link
fails, terminate only this worker's child process tree before phase is released.
"""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for attempt in range(30):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)

def command(spec):
    action=spec['action']
    from diagnostic_config import job_config
    cfg=job_config(spec)
    if action in ('mnist_batch','exposure_batch'):
        job=(ROOT/spec['_job_path']).resolve()
        if not job.is_relative_to(ROOT/'results/dual_jobs'):raise ValueError('MNIST job path escape')
        script='mnist_raw_batch.py' if action=='mnist_batch' else 'exposure_batch.py'
        return [sys.executable,str(ROOT/script),'--spec',str(job),'--config',cfg]
    if action=='calibrate':
        return [sys.executable,str(ROOT/'calibrate.py'),'--config',cfg,'--phases']
    if action in ('capture_batch','quarantine_batch','accept_batch'):
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}',spec['session']) or not re.fullmatch('[0-9a-f]{32}',spec['batch_id']):raise ValueError('Invalid guarded batch identity')
        job=(ROOT/spec['_job_path']).resolve()
        if not job.is_relative_to(ROOT/'results/dual_jobs'):raise ValueError('Guard job path escape')
        return [sys.executable,str(ROOT/'guarded_batch.py'),'--spec',str(job)]
    if action=='probe':
        bmp=(ROOT/spec['bmp']).resolve()
        if not bmp.is_relative_to(ROOT):raise ValueError('BMP path escape')
        out=(ROOT/spec['out']).resolve()
        if not out.is_relative_to(ROOT/'results'):raise ValueError('Diagnostic output must be under results')
        return [sys.executable,str(ROOT/'slm_camera.py'),'--config',cfg,'--bmp',str(bmp),'--out',spec['out']]
    if action not in ('init','prepare','capture','evaluate'):raise ValueError('Action not allowed')
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',spec['session']):raise ValueError('Invalid session')
    cmd=[sys.executable,str(ROOT/'run.py'),action,'--session',spec['session'],'--config',cfg]
    if action in ('prepare','capture'):
        stages=['vision_router','vision_expert','vision_global','language_router','language_expert','language_global']
        if spec['stage'] not in stages:raise ValueError('Invalid stage')
        cmd+=['--stage',spec['stage']]
    if action in ('prepare','evaluate'):cmd+=['--device','cuda']
    if action=='capture':
        mf=json.loads((ROOT/'sessions'/spec['session']/'play'/spec['stage']/'manifest.json').read_text(encoding='utf-8'))
        if mf['phase_sha256']!=spec['phase_receipt']['phase_sha256']:raise ValueError('Displayed phase does not match prepared manifest')
        cmd+=['--yes']
    if action=='init':
        limit=int(spec.get('limit',4))
        if not 0<=limit<=2400:raise ValueError('Invalid limit')
        cmd+=['--limit',str(limit)]
    return cmd

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    job=a.job.resolve()
    if not job.is_relative_to(ROOT/'results/dual_jobs'):raise ValueError('Job outside controlled directory')
    spec=json.loads(job.read_text(encoding='utf-8'));status=job.with_suffix('.status.json')
    result_file=job.with_suffix('.result.json')
    lock=ROOT/'results/dual_jobs/ACTIVE.lock';owned_lock=False;child=None
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);owned_lock=True
        spec['_job_path']=str(job.relative_to(ROOT))
        cmd=command(spec);write(status,{'state':'running','command':cmd,'pid':os.getpid()})
        with job.with_suffix('.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                hb=job.with_suffix('.heartbeat')
                if job.with_suffix('.cancel').exists() or not hb.exists() or time.time()-hb.stat().st_mtime>20:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    child.wait(timeout=10);raise RuntimeError('Coordinator lease expired/cancelled; owned child tree stopped')
                time.sleep(.25)
        write(result_file,{'state':'done' if child.returncode==0 else 'failed','exit_code':child.returncode,
                      'phase_receipt':spec.get('phase_receipt'),'sdk_ack_is_not_optical_verification':True})
    except BaseException as e:
        if child and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
        write(result_file,{'state':'failed','error':str(e)});raise
    finally:
        if owned_lock:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
