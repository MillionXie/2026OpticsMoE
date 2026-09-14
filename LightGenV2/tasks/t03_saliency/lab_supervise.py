"""User-authorized local three-phase supervisor; no silent failed-stage bypass.

SDK remains in each owner's main thread. CLI errors stop the chain and retain
the phase for diagnosis. STOP stops capture through the existing phase lease.
"""
import argparse
import base64
from pathlib import Path
import subprocess
import sys
import time
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read,write

STAGES=('vision_router','vision_expert','vision_global')


def run(a):
    sys.path.insert(0,str(a.bench_root.resolve()))
    from dual_run import Remote
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',session=a.session,project=a.project,stages=list(STAGES),completed=[],pid=__import__('os').getpid())
    def save():
        report['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S');write(out/'status.json',report)
    save()
    try:
        for i,stage in enumerate(STAGES,1):
            if (out/'STOP').exists():raise RuntimeError('STOP requested before next stage')
            folder=out/f'{i:02d}_{stage}'
            phase=a.phases/f'{i:02d}_{stage}.bmp'
            command=[str(Path(sys.executable).with_name('pythonw.exe')),'-u','-m',
                     'LightGenV2.tasks.t06_video_quality_assessment.lab_manual_stage',
                     '--task','salicon','--bench-root',str(a.bench_root),'--link-config',str(a.link_config),
                     '--phase',str(phase),'--out',str(folder),'--project',a.project,'--session',a.session,
                     '--config',a.config,'--stage',stage]
            si=subprocess.STARTUPINFO();si.dwFlags=0
            with (out/f'{i:02d}_{stage}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[3],stdin=subprocess.DEVNULL,
                     stdout=log,stderr=log,startupinfo=si,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
            report.update(status='running_stage',stage=stage,phase_pid=child.pid,stage_report=str(folder/'report.json'));save()
            started=time.monotonic()
            while True:
                if (out/'STOP').exists():
                    (folder/'RELEASE').touch();raise RuntimeError('STOP requested; phase lease cancels owned capture')
                if child.poll() is not None:raise RuntimeError('Phase owner exited unexpectedly')
                try:state=read(folder/'report.json')
                except (FileNotFoundError,ValueError):state={}
                report['progress']=state.get('capture_progress')
                report['remote_action']=state.get('remote_action');save()
                if state.get('status')=='failed_holding_phase':raise RuntimeError(str(state.get('error')))
                if state.get('status') in ('next_inputs_ready_wait_for_user','evaluation_complete'):break
                if time.monotonic()-started>3900:raise TimeoutError('Stage exceeded supervision limit; phase retained')
                time.sleep(5)
            # The audit runs without camera/SLM access and checks every SHA.
            with Remote(read(a.link_config)) as r:
                code=f"import subprocess;raise SystemExit(subprocess.call([{r.root+'/.venv_gpu/Scripts/python.exe'!r},'run.py','audit','--stage',{stage!r},'--session',{a.session!r},'--config',{a.config!r}],cwd={a.project!r}))"
                payload=base64.b64encode(code.encode()).decode()
                _,stdout,stderr=r.ssh.exec_command(r.root+'/.venv_gpu/Scripts/python.exe -c "import base64;exec(base64.b64decode(\''+payload+'\'))"',timeout=180)
                text=stdout.read().decode(errors='replace');errors=stderr.read().decode(errors='replace')
                (out/f'{i:02d}_{stage}_audit.log').write_text(text+'\n'+errors,encoding='utf-8')
                if stdout.channel.recv_exit_status():raise RuntimeError('Complete-stage audit failed')
                remote=a.project+'/sessions/'+a.session
                r.sftp.get(remote+'/audits/'+stage+'.json',str(out/f'{i:02d}_{stage}_audit.json'))
                if i==len(STAGES):r.sftp.get(remote+'/results.json',str(out/'results.json'))
            report['completed'].append(stage);save()
            if i<len(STAGES):
                (folder/'RELEASE').touch()
                child.wait(timeout=60)
                if child.returncode:raise RuntimeError('Phase release returned error')
        report.update(status='complete',results=str(out/'results.json'),last_phase_held=True);save()
    except BaseException as e:
        report.update(status='stopped_needs_attention',error=str(e));save()
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bench-root','link-config','phases','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('project','session','config'):p.add_argument('--'+name,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
