"""Hold ONE LGVQ phase, capture that stage, prepare next inputs, then WAIT.

Local SDK runs on its main thread. A leased scheduled task runs the existing
packaged CLI in the remote logged-in Windows desktop. No remote source edits.
"""
import argparse,base64,json,os,subprocess,sys,threading,time,uuid
from pathlib import Path
from .lab_runtime import STAGES,read,write,sha


def desktop_code(project,bench,session,config,stage,stem,stages=STAGES):
    """Fixed CLI only; phase release first cancels this job's child tree."""
    return f'''import json,subprocess,time,sys
from pathlib import Path
p=Path({project!r});stem=Path({stem!r});py={bench!r}+'/.venv_gpu/Scripts/pythonw.exe'
status=stem.with_suffix('.json');heartbeat=stem.with_suffix('.heartbeat')
commands=[['capture','--stage',{stage!r},'--phase-ready']]
stages={list(stages)!r};i=stages.index({stage!r})
commands.append(['prepare','--stage',stages[i+1],'--device','cuda'] if i<len(stages)-1 else ['evaluate','--device','cuda'])
child=None
def save(data):status.write_text(json.dumps(data,indent=2),encoding='utf-8')
try:
 with stem.with_suffix('.log').open('x',encoding='utf-8') as log:
  for command in commands:
   if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>20:raise RuntimeError('Phase lease expired before start')
   save(dict(state='running',action=command[0],stage={stage!r}))
   child=subprocess.Popen([py,'-u','run.py',*command,'--session',{session!r},'--config',{config!r},'--bench-root',{bench!r}],cwd=p,stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None:
    if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>20:
     subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait(timeout=10);raise RuntimeError('Phase lease expired; owned child stopped')
    time.sleep(.25)
   if child.returncode:raise RuntimeError('CLI failed: '+str(command))
 save(dict(state='done',stage={stage!r},phase_unchanged=True))
except BaseException as e:
 if child and child.poll() is None:
  subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait(timeout=10)
 save(dict(state='failed',error=str(e),stage={stage!r}))
'''


def run(a):
    stages=STAGES
    if getattr(a,'task','lgvq')=='salicon':
        stages=('vision_router','vision_expert','vision_global')
    if a.stage not in stages:raise ValueError('Stage not part of this task')
    sys.path.insert(0,str(a.bench_root.resolve()))
    from dual_run import Remote
    from phase_hdmi import PhaseHDMI,load_native
    from phase_owner import message_pump
    from phase_display import DisplayOrigin
    link=read(a.link_config);out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for value in (a.project,a.session,a.config):
        if any(c in value for c in "'<>\r\n\""):raise ValueError('Invalid path/session')
    if 'blinkhdmi.exe' in subprocess.check_output(['tasklist','/FI','IMAGENAME eq BlinkHdmi.exe','/FO','CSV'],text=True).lower():raise RuntimeError('Close Blink GUI')
    with Remote(link) as r:
        with r.sftp.open(a.project+'/sessions/'+a.session+'/play/'+a.stage+'/manifest.json','rb') as f:mf=json.loads(f.read().decode('utf-8'))
    load_native(a.phase,mf['phase_sha256'])
    lock=a.bench_root/'results/phase_sdk_owner.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    release=out/'RELEASE';done=threading.Event();worker=None
    report=dict(status='loading_phase',stage=a.stage,session=a.session,remote_project=a.project,phase=str(a.phase.resolve()),phase_sha256=sha(a.phase),count=len(mf['entries']),automatic_phase_switching=False,pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    def save():report['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S');write(out/'report.json',report)
    save()
    try:
        with DisplayOrigin(bool(link.get('phase_display_align_top',False))):
            pump=message_pump()
            with PhaseHDMI(link['phase_sdk'],link['phase_lut'],link.get('phase_settle_s',1),pixel_format=link.get('phase_pixel_format','rgba')) as sdk:
                report['receipt']=sdk.show(a.phase,mf['phase_sha256'],pump=pump);report['status']='holding_phase';save()
                def work():
                    name='LGVQ_'+uuid.uuid4().hex
                    try:
                        with Remote(link) as r:
                            logs=a.project+'/sessions/'+a.session+'/logs';stem=logs+'/'+name
                            r.ps(f"New-Item -ItemType Directory -Force -Path '{logs}' | Out-Null")
                            def heartbeat():
                                with r.sftp.open(stem+'.heartbeat','w') as f:f.write('phase_owner_alive')
                            heartbeat()
                            payload=base64.b64encode(desktop_code(a.project,r.root,a.session,a.config,a.stage,stem,stages).encode()).decode()
                            arguments='-u -c "import base64;exec(base64.b64decode(\''+payload+'\'))"'
                            import html
                            xml=f'<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Principals><Principal id="Author"><UserId>PS</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals><Settings><ExecutionTimeLimit>PT1H</ExecutionTimeLimit></Settings><Actions Context="Author"><Exec><Command>{html.escape(r.root+"/.venv_gpu/Scripts/pythonw.exe")}</Command><Arguments>{html.escape(arguments)}</Arguments><WorkingDirectory>{html.escape(a.project)}</WorkingDirectory></Exec></Actions></Task>'
                            # Avoid cmd.exe's 8191-character command limit. This
                            # generated task definition is an audited run artifact.
                            with r.sftp.open(stem+'.task.xml','w') as f:f.write(xml.encode('utf-8'))
                            r.ps("$ErrorActionPreference='Stop'; $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
                                 +f"$xml=Get-Content -LiteralPath '{stem}.task.xml' -Raw -Encoding UTF8; "
                                 +"$xml=$xml.Replace('<UserId>PS</UserId>',('<UserId>'+$sid+'</UserId>')); "
                                 +f"Register-ScheduledTask -TaskName '{name}' -Xml $xml | Out-Null; Start-ScheduledTask -TaskName '{name}'")
                            report.update(status='capturing',remote_log=stem+'.log',remote_job_status=stem+'.json');save()
                            started=time.monotonic();last_progress=0
                            try:
                                while True:
                                    if not release.exists():heartbeat()
                                    try:
                                        with r.sftp.open(stem+'.json','rb') as f:status=json.loads(f.read().decode('utf-8'))
                                    except (FileNotFoundError,json.JSONDecodeError):status={}
                                    if time.monotonic()-last_progress>10:
                                        try:
                                            with r.sftp.open(a.project+'/sessions/'+a.session+'/status.json','rb') as f:report['capture_progress']=json.loads(f.read().decode('utf-8'))
                                        except (FileNotFoundError,json.JSONDecodeError):pass
                                        report['remote_action']=status.get('action');save();last_progress=time.monotonic()
                                    if status.get('state') in ('done','failed'):
                                        report['remote_result']=status
                                        if status['state']=='failed':raise RuntimeError(str(status))
                                        report['status']='next_inputs_ready_wait_for_user' if a.stage!=stages[-1] else 'evaluation_complete';save();break
                                    if time.monotonic()-started>3600:raise TimeoutError('One-stage job exceeded one hour')
                                    time.sleep(2)
                            finally:
                                r.ps(f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false")
                    except BaseException as e:report.update(status='failed_holding_phase',error=str(e));save()
                    finally:done.set()
                worker=threading.Thread(target=work,daemon=False);worker.start();last=time.monotonic()
                while not release.exists() or not done.is_set():
                    pump()
                    if time.monotonic()-last>=1:sdk.repeat();last=time.monotonic()
                    time.sleep(.01)
                report['phase_released']=True;save()
    finally:
        release.touch()
        if worker:worker.join(timeout=50)
        lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bench-root','link-config','phase','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['project','session','config']:p.add_argument('--'+name,required=True)
    p.add_argument('--task',choices=['lgvq','salicon'],default='lgvq')
    p.add_argument('--stage',choices=STAGES,required=True);run(p.parse_args())
if __name__=='__main__':main()
