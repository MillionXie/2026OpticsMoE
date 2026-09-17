"""Prioritize unclaimed seed17 depths2/4 without restarting paused seed27/37 runs."""
import argparse,concurrent.futures,json,os,queue,signal,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(p.read_text())
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def command(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.read_bytes().decode().split('\0') if p.exists() else []
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);a=parser.parse_args();a.run=a.run.resolve();out=a.run/'meeting_priority';out.mkdir(exist_ok=False)
    paused=[];running={};records=[];q=queue.Queue()
    try:
        for p in a.run.glob('worker_gpu*.json'):
            w=read(p);j=w['job']
            if w['state']!='training' or j['depth']!=6 or j['seed']==17:continue
            claim=a.run/'claims'/j['name'];parent=read(claim/'worker.json')['pid']
            childfile=Path('/proc')/str(parent)/'task'/str(parent)/'children'
            if not childfile.exists():continue
            children=[int(x) for x in childfile.read_text().split()]
            for pid in children:
                cmd=command(pid)
                if '--phase' not in cmd or cmd[cmd.index('--phase')+1]!='train-one':continue
                assert any('kather_coverage.py' in x for x in cmd) and str(a.run/'jobs'/j['name']) in [str(Path(x).resolve()) for x in cmd if '/jobs/' in x],cmd
                gpu=read(claim/'worker.json')['visible_gpu'];os.kill(pid,signal.SIGSTOP);paused.append(dict(pid=pid,command=cmd,gpu=gpu,job=j['name']))
        assert len(paused)==4,paused
        jobs=[j for j in read(a.run/'jobs.json') if j['seed']==17 and j['depth'] in [2,4]]
        jobs.sort(key=lambda j:(not j['arch'].startswith('moe'),-j['depth']))
        for j in jobs:
            claim=a.run/'claims'/j['name']
            try:claim.mkdir()
            except FileExistsError:continue
            save(claim/'worker.json',dict(pid=os.getpid(),priority='user_requested_meeting',time=time.time()));q.put(j)
        save(out/'priority_plan.json',dict(paused=paused,jobs=list(q.queue),unchanged_training_config=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        def worker(gpu):
            while True:
                try:j=q.get_nowait()
                except queue.Empty:return
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('kather_coverage.py')),'--phase','train-one','--data',str(a.data),'--out',str(a.run/'jobs'/j['name']),'--candidate',j['candidate'],'--arch',j['arch'],'--depth',str(j['depth']),'--seed',str(j['seed'])]
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                with (a.run/'logs'/(j['name']+'.log')).open('x') as log:
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[gpu]=proc
                    save(out/('worker_'+gpu+'.json'),dict(job=j,pid=proc.pid,state='training'));rc=proc.wait();assert rc==0,(j,rc)
                save(a.run/'claims'/j['name']/'status.json',dict(state='complete',priority=True,time=time.time()));records.append(j['name']);save(out/('worker_'+gpu+'.json'),dict(job=j,state='complete'));q.task_done()
        def stop(sig,frame):raise KeyboardInterrupt('Priority manager interrupted')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fs=[pool.submit(worker,x['gpu']) for x in paused]
            for f in fs:f.result()
        save(out/'status.json',dict(state='priority_complete',completed=records,time=time.time()))
    finally:
        for proc in running.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait()
        resumed=[]
        for x in paused:
            if command(x['pid'])==x['command']:os.kill(x['pid'],signal.SIGCONT);resumed.append(x['pid'])
        save(out/'resume.json',dict(resumed=resumed,time=time.time()))

if __name__=='__main__':main()
