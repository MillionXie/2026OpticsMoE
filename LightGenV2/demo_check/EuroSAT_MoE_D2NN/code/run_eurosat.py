"""Sequential, resumable server queue for the explicitly approved experiment."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)

def main():
    os.chdir(ROOT);(ROOT/'runs').mkdir(exist_ok=True)
    lock=(ROOT/'runs/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(ROOT/'runs/queue_identity.json',dict(pid=os.getpid(),started_at=time.time(),python=sys.executable))
    assert json.loads((ROOT/'USER_AUTHORIZATION.json').read_text())['approved']
    while True:
        progress=json.loads((ROOT/'data_progress.json').read_text())
        if progress['state']=='failed':raise RuntimeError('Dataset preparation failed: '+str(progress))
        if progress['state']=='complete':break
        atomic(ROOT/'runs/queue_status.json',dict(state='waiting_for_data',data=progress,updated_at=time.time()));time.sleep(30)
    checks=json.loads((ROOT/'DATA_CHECKS.json').read_text());assert checks['passed']
    # Do not compete with other experiments; wait for a free GPU before initializing.
    while True:
        running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not running:break
        atomic(ROOT/'runs/queue_status.json',dict(state='waiting_for_gpu',other_gpu_pids=running,updated_at=time.time()));time.sleep(30)
    jobs=[['preflight_eurosat.py']]+[['train_eurosat.py','moe',s] for s in ('shared','expert_A','expert_B')]+[['train_eurosat.py','merge'],['train_eurosat.py','moe','router'],['train_eurosat.py','A_only'],['train_eurosat.py','B_only']]+[['train_eurosat.py','AB',s] for s in ('shared','expert_A','expert_B','router')]+[['train_eurosat.py','select_ab'],['evaluate_eurosat.py']]
    for i,job in enumerate(jobs):
        atomic(ROOT/'runs/queue_status.json',dict(state='running',job=job,job_index=i,jobs=len(jobs),updated_at=time.time()))
        print('START',job,flush=True)
        subprocess.run([sys.executable,'-u',*job],cwd=ROOT,check=True)
        print('FINISHED',job,flush=True)
    assert json.loads((ROOT/'results/PERFORMANCE.json').read_text())['state']=='complete'

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic(ROOT/'runs/queue_status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),updated_at=time.time()));raise
