"""CPU-only four-card pilot: two L2 hundred-epoch fits and two L16 timing fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for key in ('out', 'profile', 'data', 'probe'): p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Parent must not use CUDA'
    profile = json.loads(a.profile.read_text())
    gpu = profile['gpu_uuids']; assert len(gpu) == len(set(gpu)) == 4
    gate = json.loads(a.probe.read_text())
    for depth in (2,16):
        for rho in (0.,.3):
            assert any(x['depth']==depth and x['rho']==rho and x['microbatch']==4 and x['passed'] for x in gate['results'])
    assert profile['dataset'] == 'Caltech101_prespecified50_fewshot' and profile['maximum_epochs']==100
    a.out.mkdir(parents=True, exist_ok=False)
    task = Path(__file__).resolve().parent
    jobs = [('L2_rho0',2,0.,100,gpu[0]), ('L2_rho03',2,.3,100,gpu[3]),
            ('L16_timing_rho0',16,0.,2,gpu[1]), ('L16_timing_rho03',16,.3,2,gpu[2])]
    metadata = dict(pid=os.getpid(), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),
        scope='Only L2 hundred-epoch train/val pilot and L16 two-epoch end-to-end timing. No test; no full eight-arm suite yet.')
    save(a.out/'metadata.json', metadata)
    children = []
    try:
        for name, depth, rho, epochs, uuid in jobs:
            cmd = [sys.executable,str(task/'medicinal_pilot.py'),'--out',str(a.out/name),
                '--data',str(a.data),'--profile',str(a.profile),'--probe',str(a.probe),
                '--gpu',uuid,'--rho',str(rho),'--depth',str(depth),'--epochs',str(epochs),
                '--seconds','36000' if epochs==100 else '900','--full-budget','--micro','4','--batched','1']
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = uuid
            with (a.out/(name+'.log')).open('xb') as log:
                child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            children.append((child,dict(name=name,pid=child.pid,gpu_uuid=uuid,depth=depth,rho=rho,epochs=epochs,command=cmd)))
            save(a.out/'current_processes.json',[info for _,info in children])
        while any(child.poll() is None for child,_ in children):
            save(a.out/'current_processes.json',[info for child,info in children if child.poll() is None])
            save(a.out/'status.json',dict(state='pilot_running',test_evaluations=0,
                jobs=[dict(info,exit_code=child.poll()) for child,info in children]))
            time.sleep(5)
        failures=[info for child,info in children if child.returncode!=0]
        save(a.out/'current_processes.json',[])
        save(a.out/'status.json',dict(state='pilot_failed' if failures else 'pilot_complete_awaiting_validation_review',
            failures=failures,test_evaluations=0,jobs=[dict(info,exit_code=child.returncode) for child,info in children]))
        assert not failures, failures
    finally:
        # Never kill another project or auto-retry failed children/test claims.
        save(a.out/'current_processes.json',[info for child,info in children if child.poll() is None])


if __name__ == '__main__': main()
