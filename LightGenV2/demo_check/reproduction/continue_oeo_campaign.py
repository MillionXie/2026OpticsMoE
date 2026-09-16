"""Finish the predeclared Kather pilot/suite, reusing GPUs only after Blood ends.

No training decisions depend on test results. At most the five named GPUs are used.
This coordinator neither reserves memory nor terminates other users' processes.
"""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(p.read_text())
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(p):return read(p/'status.json').get('state') if (p/'status.json').exists() else 'absent'
def free_gpus():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return [int(row.split(',')[0]) for row in text.strip().splitlines() if int(row.split(',')[1])<128]

def main():
    p=argparse.ArgumentParser();p.add_argument('--blood',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);assert not a.suite.exists();save(a.out/'metadata.json',dict(command=sys.argv,pid=os.getpid(),time=now(),allowed_gpus=[0,1,2,3,4],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),policy='GPU2 continues Kather. Add 0/1/3/4 only after Blood completes and device memory is below 128 MiB. Never use model performance gaps for decisions.'));proc=None
    try:
        while state(a.pilot)!='pilot_complete_test_not_read':
            assert state(a.pilot)!='failed',read(a.pilot/'status.json');assert state(a.blood)!='failed',read(a.blood/'status.json');save(a.out/'status.json',dict(state='waiting_for_pilot',time=now()));time.sleep(30)
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('kather2016_experiment.py')),'--phase','suite','--data',str(a.data),'--out',str(a.suite),'--pilot',str(a.pilot),'--gpus','2']
        with (a.out/'suite.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(a.out/'suite_process.json',dict(pid=proc.pid,command=cmd,time=now()));scheduled=[2]
            while proc.poll() is None:
                blood=state(a.blood);assert blood!='failed',read(a.blood/'status.json')
                if blood=='complete' and (a.suite/'gpu_schedule.json').exists():
                    free=free_gpus();extra=[g for g in [0,1,3,4] if g not in scheduled and g in free]
                    if extra:
                        scheduled+=extra;save(a.suite/'gpu_schedule.json',scheduled);save(a.out/'last_gpu_expansion.json',dict(gpus=scheduled,free_at_check=free,time=now()))
                save(a.out/'status.json',dict(state='suite_running',suite_state=state(a.suite),blood_state=blood,kather_gpus=scheduled,time=now()));time.sleep(30)
            assert proc.returncode==0 and state(a.suite)=='complete',(proc.returncode,state(a.suite))
        save(a.out/'status.json',dict(state='complete',time=now(),free_gpus=free_gpus(),note='Training/evaluation subprocess exited. Final process audit should distinguish other users from this campaign.'))
    except BaseException as e:
        save(a.out/'status.json',dict(state='failed',error=repr(e),suite_pid=None if proc is None else proc.pid,time=now()));raise
if __name__=='__main__':main()
