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
import uuid
import html
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read,write

STAGES=('vision_router','vision_expert','vision_global')


def completed_prefix(stages,measured):
    if measured!=list(stages[:len(measured)]) or len(measured)>len(stages):
        raise ValueError('Measured stages are not a contiguous prefix')
    return len(measured)


def audit_stage(r,a,stage,index,out):
    code=f"import subprocess;raise SystemExit(subprocess.call([{r.root+'/.venv_gpu/Scripts/python.exe'!r},'run.py','audit','--stage',{stage!r},'--session',{a.session!r},'--config',{a.config!r}],cwd={a.project!r}))"
    payload=base64.b64encode(code.encode()).decode()
    _,stdout,stderr=r.ssh.exec_command(r.root+'/.venv_gpu/Scripts/python.exe -c "import base64;exec(base64.b64decode(\''+payload+'\'))"',timeout=300)
    text=stdout.read().decode(errors='replace');errors=stderr.read().decode(errors='replace')
    (out/f'{index:02d}_{stage}_audit.log').write_text(text+'\n'+errors,encoding='utf-8')
    if stdout.channel.recv_exit_status():raise RuntimeError('Complete-stage audit failed: '+stage)
    remote=a.project+'/sessions/'+a.session
    r.sftp.get(remote+'/audits/'+stage+'.json',str(out/f'{index:02d}_{stage}_audit.json'))


def start_keep_awake(remote,project,session):
    """Temporary Windows power request, scoped to this leased hardware run.

    Generated job artifact, like the phase coordinator's desktop command.
    Does not modify persistent power plans, display mode, camera or SLM state.
    """
    name='SALICON_Awake_'+uuid.uuid4().hex
    helper=f'''import ctypes,json,os,time,subprocess
from pathlib import Path
p=Path({project!r})/'sessions'/{session!r}
status=p/'keep_awake.json';lease=p/'keep_awake.lock'
fd=os.open(lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
kernel=ctypes.windll.kernel32
record=dict(pid=os.getpid(),started=time.time(),state='requested',persistent_power_plan_changed=False)
try:
 if not kernel.SetThreadExecutionState(0x80000003):raise RuntimeError('Windows power request failed')
 status.write_text(json.dumps(record,indent=2),encoding='utf-8')
 deadline=time.monotonic()+4*3600
 while time.monotonic()<deadline:
  if (p/'results.json').exists():record['reason']='evaluation_complete';break
  beats=list((p/'logs').glob('*.heartbeat'))
  newest=max((f.stat().st_mtime for f in beats),default=record['started'])
  if time.time()-newest>240:record['reason']='no_phase_lease_for_four_minutes';break
  time.sleep(10)
 else:record['reason']='four_hour_limit'
finally:
 kernel.SetThreadExecutionState(0x80000000)
 record.update(state='released',finished=time.time());status.write_text(json.dumps(record,indent=2),encoding='utf-8')
 lease.unlink(missing_ok=True)
 subprocess.run(['powershell','-NoProfile','-Command',"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false"],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
'''
    child=base64.b64encode(helper.encode()).decode()
    arguments='-c "import base64;exec(base64.b64decode(\''+child+'\'))"'
    path=project+'/sessions/'+session+'/'+name+'.task.xml'
    xml=f'<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Principals><Principal id="Author"><UserId>PS</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals><Settings><ExecutionTimeLimit>PT5H</ExecutionTimeLimit></Settings><Actions Context="Author"><Exec><Command>{html.escape(remote.root+"/.venv_gpu/Scripts/pythonw.exe")}</Command><Arguments>{html.escape(arguments)}</Arguments><WorkingDirectory>{html.escape(project)}</WorkingDirectory></Exec></Actions></Task>'
    with remote.sftp.open(path,'w') as f:f.write(xml.encode('utf-8'))
    remote.ps("$ErrorActionPreference='Stop'; $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
              +f"$xml=Get-Content -LiteralPath '{path}' -Raw -Encoding UTF8; "
              +"$xml=$xml.Replace('<UserId>PS</UserId>',('<UserId>'+$sid+'</UserId>')); "
              +f"Register-ScheduledTask -TaskName '{name}' -Xml $xml | Out-Null; Start-ScheduledTask -TaskName '{name}'")
    return dict(task_name=name,task_xml=path)


def run(a):
    stages=STAGES
    task=getattr(a,'task','salicon')
    if task=='lgvq':
        from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import STAGES as stages
    sys.path.insert(0,str(a.bench_root.resolve()))
    from dual_run import Remote
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',task=task,session=a.session,project=a.project,stages=list(stages),completed=[],pid=__import__('os').getpid())
    def save():
        report['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S');write(out/'status.json',report)
    save()
    try:
        skip=0
        with Remote(read(a.link_config)) as r:
            if getattr(a,'resume_completed',False):
                import json
                with r.sftp.open(a.project+'/sessions/'+a.session+'/session.json','rb') as f:session=json.loads(f.read().decode('utf-8'))
                skip=completed_prefix(stages,session['measured_stages'])
                if skip==len(stages):raise ValueError('All stages already captured; run audit/evaluate rather than restarting phase supervision')
                for index,stage in enumerate(stages[:skip],1):
                    report.update(status='auditing_completed_stage',stage=stage);save()
                    audit_stage(r,a,stage,index,out)
                    report['completed'].append(stage);save()
            report['remote_keep_awake']=start_keep_awake(r,a.project,a.session);save()
        for i,stage in enumerate(stages,1):
            if i<=skip:continue
            if (out/'STOP').exists():raise RuntimeError('STOP requested before next stage')
            folder=out/f'{i:02d}_{stage}'
            phase=a.phases/f'{i:02d}_{stage}.bmp'
            command=[str(Path(sys.executable).with_name('pythonw.exe')),'-u','-m',
                     'LightGenV2.tasks.t06_video_quality_assessment.lab_manual_stage',
                     '--task',task,'--bench-root',str(a.bench_root),'--link-config',str(a.link_config),
                     '--phase',str(phase),'--out',str(folder),'--project',a.project,'--session',a.session,
                     '--config',a.config,'--stage',stage,'--log-file',str(out/f'{i:02d}_{stage}.log')]
            if getattr(a,'verify_phase_optically',False):command.append('--verify-phase-optically')
            if getattr(a,'phase_reference_dir',None):command.extend(['--phase-reference-dir',str(a.phase_reference_dir)])
            si=subprocess.STARTUPINFO();si.dwFlags=0
            # Do not redirect Win32 standard handles into the SDK owner.
            # Logging is redirected at Python level inside that process instead.
            # Optical preflight, not this launch change alone, decides validity.
            child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[3],
                 startupinfo=si,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
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
                audit_stage(r,a,stage,i,out)
                remote=a.project+'/sessions/'+a.session
                if i==len(stages):r.sftp.get(remote+'/results.json',str(out/'results.json'))
            report['completed'].append(stage);save()
            if i<len(stages):
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
    p.add_argument('--task',choices=['salicon','lgvq'],default='salicon')
    p.add_argument('--resume-completed',action='store_true',help='Audit and skip completed stages in the same immutable session')
    p.add_argument('--verify-phase-optically',action='store_true')
    p.add_argument('--phase-reference-dir',type=Path)
    run(p.parse_args())

if __name__=='__main__':main()
