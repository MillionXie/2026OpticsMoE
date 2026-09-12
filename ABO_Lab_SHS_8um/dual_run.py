"""One LOCAL command: local HDMI phase + remote desktop amplitude/camera/GPU.

SSH password is prompted or read from SHS_SSH_PASSWORD, never saved.
No listening TCP phase service, no external arbitrary command endpoint.
"""
import argparse,base64,getpass,json,os,time,uuid
from pathlib import Path
import paramiko
from phase_hdmi import PhaseHDMI,sha
ROOT=Path(__file__).resolve().parent
STAGES=['vision_router','vision_expert','vision_global','language_router','language_expert','language_global']

class Remote:
    def __init__(self,c):self.c=c;self.active=None
    def __enter__(self):
        self.ssh=paramiko.SSHClient();self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.ssh.connect(self.c['host'],port=self.c['port'],username=self.c['username'],
            password=os.environ.get('SHS_SSH_PASSWORD') or getpass.getpass('SHS SSH password: '),
            timeout=20,look_for_keys=False,allow_agent=False)
        self.sftp=self.ssh.open_sftp();self.root=self.c['project'].replace('\\','/').rstrip('/');return self
    def read(self,rel):
        with self.sftp.open(self.root+'/'+rel,'r') as f:return json.loads(f.read().decode('utf-8-sig'))
    def exists(self,rel):
        try:self.sftp.stat(self.root+'/'+rel);return True
        except FileNotFoundError:return False
    def putjson(self,rel,data):
        dest=self.root+'/'+rel
        with self.sftp.open(dest+'.tmp','w') as f:f.write(json.dumps(data))
        self.sftp.rename(dest+'.tmp',dest)
    def ps(self,script):
        encoded=base64.b64encode(script.encode('utf-16le')).decode()
        _,out,err=self.ssh.exec_command('powershell -NoProfile -EncodedCommand '+encoded)
        code=out.channel.recv_exit_status()
        if code:raise RuntimeError(err.read().decode('utf-8',errors='replace'))
    def heartbeat(self,rel):
        with self.sftp.open(self.root+'/'+rel+'.heartbeat','w') as f:f.write('alive')
    def job(self,spec):
        name='ABO_Dual_'+uuid.uuid4().hex;rel='results/dual_jobs/'+name
        # Fixed project path is admin-owned config, reject shell/XML delimiters.
        if any(x in self.root for x in "'<>\r\n"):raise ValueError('Invalid remote project path')
        self.ps(f"New-Item -ItemType Directory -Force -Path '{self.root}/results/dual_jobs' | Out-Null")
        self.putjson(rel+'.json',spec);self.heartbeat(rel);self.active=(name,rel)
        exe=self.root+'/.venv_gpu/Scripts/pythonw.exe'
        script=f'''$ErrorActionPreference='Stop'
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$xml=@"
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
<Principals><Principal id="Author"><UserId>$sid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
<Settings><ExecutionTimeLimit>PT24H</ExecutionTimeLimit></Settings>
<Actions Context="Author"><Exec><Command>{exe}</Command><Arguments>desktop_job.py --job {rel}.json</Arguments><WorkingDirectory>{self.root}</WorkingDirectory></Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskName '{name}' -Xml $xml | Out-Null
Start-ScheduledTask -TaskName '{name}'
'''
        started=time.monotonic();last_print=0
        try:
            self.ps(script)
            while True:
                self.heartbeat(rel)
                if self.exists(rel+'.result.json') or self.exists(rel+'.status.json'):
                    terminal=self.exists(rel+'.result.json')
                    status=self.read(rel+('.result.json' if terminal else '.status.json'))
                    if status['state'] in ('done','failed'):
                        self.active=None
                        if status['state']!='done':raise RuntimeError(f'Job failed: {status}; inspect {rel}.log')
                        return status
                elif time.monotonic()-started>60:
                    raise TimeoutError('Desktop worker did not start in 60s; check logged-in desktop and task status')
                if time.monotonic()-started>86400:raise TimeoutError('Job exceeded 24h')
                if time.monotonic()-last_print>30:
                    print(spec['action'],spec.get('stage',''),f'{time.monotonic()-started:.0f}s',flush=True);last_print=time.monotonic()
                time.sleep(1)
        finally:
            if self.active:self.cancel()
            self.ps(f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false")
    def cancel(self):
        if not self.active:return
        _,rel=self.active
        try:
            with self.sftp.open(self.root+'/'+rel+'.cancel','w') as f:f.write('cancel')
        except Exception:pass
        # Keep the local phase displayed until the remote 20s lease has expired.
        time.sleep(30);self.active=None
    def download(self,rel,dest):
        # Only package-relative files from trusted manifests, no path traversal.
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid artifact path')
        dest.parent.mkdir(parents=True,exist_ok=True);self.sftp.get(self.root+'/'+rel,str(dest))
    def __exit__(self,*args):
        self.cancel();self.sftp.close();self.ssh.close()

def main():
    # Keep the original CLI name, but never fall back to unchecked stage loads.
    from guarded_workflow import main as guarded_main
    return guarded_main(default_action='run')

if __name__=='__main__':main()
