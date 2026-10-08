import json,os,subprocess,sys,traceback
from experiments.expert_merge.core import ROOT,authorize,plan,atomic_json

def main():
    authorize();root=ROOT/'runs/merge_seed42';root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(ROOT/'runs/merge.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any((root/s).exists() for s in plan()['stages']):raise RuntimeError('Existing run requires explicit recovery; never overwrite it')
    subprocess.run([sys.executable,'-m','experiments.expert_merge.verify'],cwd=ROOT,check=True)
    for stage in plan()['stages']:
        if stage=='router':subprocess.run([sys.executable,'-m','experiments.expert_merge.merge'],cwd=ROOT,check=True)
        with (root/(stage+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','experiments.expert_merge.train',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(root/'queue_status.json',dict(status='training',stage=stage,pid=os.getpid(),child_pid=child.pid))
            if child.wait()!=0:raise RuntimeError('Stage failed: '+stage+'; see '+str(log.name))
    subprocess.run([sys.executable,'-m','experiments.expert_merge.report'],cwd=ROOT,check=True)
    atomic_json(root/'queue_status.json',dict(status=plan()['after_completion'],d1_started=False,summary=str(root/'results.json')))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(ROOT/'runs/failure.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()));raise
