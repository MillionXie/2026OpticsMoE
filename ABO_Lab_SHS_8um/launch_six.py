"""Launch a persistent SDK coordinator without hiding its HDMI output window.

pythonw removes the console; it does NOT set SW_HIDE for the vendor SLM window.
Never start this host through PowerShell -WindowStyle Hidden.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def launch(a):
    if os.name!='nt':raise RuntimeError('Windows required')
    exe=Path(sys.executable).with_name('pythonw.exe')
    if not exe.is_file():raise FileNotFoundError(exe)
    out=a.out.resolve();link=a.link_config.resolve()
    if not out.is_relative_to(ROOT/'results') or not link.is_file():raise ValueError('Output under results and existing link config required')
    if (ROOT/'results/phase_sdk_owner.lock').exists():raise RuntimeError('Phase owner exists; no parallel launch')
    if not os.environ.get('SHS_SSH_PASSWORD'):raise ValueError('Set SHS_SSH_PASSWORD in the launching terminal; never store it in a file')
    if a.resume:
        r=json.loads((out/'report.json').read_text(encoding='utf-8'))
        if r['status']!='stopped':raise ValueError('Resume only a stopped run')
    elif out.exists():raise FileExistsError(out)
    logdir=out.parent;logdir.mkdir(parents=True,exist_ok=True)
    tag=('resume_' if a.resume else 'launch_')+time.strftime('%Y%m%d_%H%M%S')
    stdout=logdir/(tag+'.log');stderr=logdir/(tag+'.stderr.log')
    cmd=[str(exe),'-u',str(ROOT/'smoke_six.py'),'--link-config',str(link),
         '--remote-config',a.remote_config,'--out',str(out),'--limit',str(a.limit)]
    if a.full_dataset:cmd.append('--full-dataset')
    if a.resume:cmd.append('--resume')
    # Explicit STARTUPINFO without STARTF_USESHOWWINDOW, even if this launcher
    # was itself started hidden. Redirection only adds STARTF_USESTDHANDLES.
    si=subprocess.STARTUPINFO();si.dwFlags=0
    with stdout.open('x',encoding='utf-8') as sout,stderr.open('x',encoding='utf-8') as serr:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=sout,stderr=serr,
            startupinfo=si,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
    data=dict(pid=proc.pid,command=cmd,log=str(stdout),stderr=str(stderr),report=str(out/'report.json'),
        sdk_window_not_hidden=True,console='pythonw; no console window',started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    launch_file=logdir/(tag+'.json');launch_file.write_text(json.dumps(data,indent=2),encoding='utf-8')
    with (logdir/(tag+'.monitor.log')).open('x',encoding='utf-8') as mon:
        watcher=subprocess.Popen([str(exe),str(ROOT/'monitor_run.py'),'--launch-file',str(launch_file)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=mon,stderr=mon,startupinfo=si,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
    data['monitor_pid']=watcher.pid;data['live_page']=str(ROOT/'reports/00_current/03_live.html')
    print(json.dumps(data,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--link-config',type=Path,required=True)
    p.add_argument('--remote-config',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=4);p.add_argument('--full-dataset',action='store_true')
    p.add_argument('--resume',action='store_true');launch(p.parse_args())
