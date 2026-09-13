"""Read-only process/job supervision. Never captures, retries, or kills hardware."""
import argparse,html,json,re,time
from pathlib import Path
import win32api,win32event
from guarded_workflow import read,write
from dual_run import Remote
ROOT=Path(__file__).resolve().parent

def monitor(launch_file):
    launch=read(launch_file);current=ROOT/'reports/00_current'
    command=launch['command'];link=read(command[command.index('--link-config')+1])
    try:handle=win32api.OpenProcess(0x100000,False,launch['pid'])
    except Exception:handle=None
    try:
        while True:
            alive=handle is not None and win32event.WaitForSingleObject(handle,0)==258
            state={'checked_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'pid':launch['pid'],'process_alive':alive,
                'launch':launch,'monitor_scope':'Process, local stage report and remote worker log; no optical validation performed by this monitor'}
            try:state['inference']=read(launch['report'])
            except Exception as e:state['report_read_error']=str(e)
            text=Path(launch['log']).read_text(encoding='utf-8',errors='replace')
            state['coordinator_tail']=text.splitlines()[-12:]
            logs=re.findall(r'^REMOTE_LOG (.+)$',text,re.M)
            try:
                with Remote(link) as r:
                    if logs:
                        path=logs[-1].strip();prefix=r.root+'/results/dual_jobs/'
                        if not path.startswith(prefix) or '..' in path or not path.endswith('.log'):raise ValueError('Invalid worker log path')
                        with r.sftp.open(path,'rb') as f:
                            f.seek(max(0,f.stat().st_size-5000));state['worker_tail']=f.read().decode('utf-8',errors='replace').splitlines()[-15:]
                    r.putjson('reports/00_current/03_live.json',state)
            except Exception as e:state['monitor_connection_warning']=str(e)
            write(current/'03_live.json',state)
            page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>六层实时监督</title><style>body{max-width:1200px;margin:30px auto;font:18px/1.6 sans-serif}pre{white-space:pre-wrap}</style>'
            page+='<h1>'+('进程运行中' if alive else '进程已结束：请看最终状态/错误')+'</h1><pre>'+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>'
            (current/'03_live.html').write_text(page,encoding='utf-8')
            if not alive:break
            time.sleep(15)
    finally:
        if handle:handle.Close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch-file',type=Path,required=True);a=p.parse_args();monitor(a.launch_file)
