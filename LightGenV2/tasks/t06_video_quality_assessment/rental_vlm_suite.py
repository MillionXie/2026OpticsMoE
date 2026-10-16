"""One finite rental pipeline; first ready fixed model, no duplicate GPU jobs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

TASK=Path(__file__).resolve().parent
RUN=TASK/"runs/simulation/lgvq_full_deepseek_kimi_rental_20261017"
DEADLINE=1792185600.0
GPU="GPU-780326a6-79bc-7a45-36f6-683c5b278bee"
PYTHONS={"deepseek":"/data/venvs/deepseek/bin/python","kimi":"/data/venvs/kimi/bin/python"}

def write(value):
    p=RUN/"suite_status.json";t=p.with_suffix(".tmp")
    t.write_text(json.dumps(value,indent=2));t.replace(p)

def ready(key):
    return ((Path("/data/models")/key/"asset_receipt.json").exists()
            and Path(f"/data/{key}_environment_ready.json").exists()
            and Path("/data/lgvq/transfer_receipt.json").exists())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--deadline",type=float,default=DEADLINE)
    args=p.parse_args();deadline=min(args.deadline,DEADLINE)
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"suite_status.json").exists():raise RuntimeError("Unique suite already attempted; preserve evidence")
    pending=["deepseek","kimi"];results={};started=time.time()
    while pending:
        if time.time()>=deadline:
            write(dict(status="deadline",pending=pending,results=results,time=time.time()));return
        key=next((k for k in pending if ready(k)),None)
        if key is None:
            write(dict(status="waiting_for_verified_assets_data_environment",pending=pending,
                       results=results,started=started,time=time.time()))
            time.sleep(15);continue
        for phase in ("cpu-preflight","train","evaluate","utilization"):
            command=[PYTHONS[key],"-u","-m",__package__+".rental_vlm_temporal",
                     "--model",key,"--phase",phase,"--deadline",str(deadline)]
            env={**os.environ,"CUDA_VISIBLE_DEVICES":"" if phase=="cpu-preflight" else GPU,
                 "HF_HOME":"/data/cache/huggingface","TMPDIR":"/data/tmp",
                 "PYTHONPYCACHEPREFIX":"/data/cache/pycache","TORCH_HOME":"/data/cache/torch",
                 "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","OMP_NUM_THREADS":"8"}
            log=RUN/f"{key}_{phase}.log"
            with log.open("x") as stream:
                child=subprocess.Popen(command,cwd=TASK.parents[2],env=env,stdout=stream,stderr=subprocess.STDOUT)
                write(dict(status="running",model=key,phase=phase,child_pid=child.pid,command=command,
                           results=results,deadline=deadline,started=started,time=time.time()))
                try:code=child.wait(timeout=max(1,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    code=124
            results[f"{key}/{phase}"]=dict(exit_code=code,completed=time.time())
            if code:break
        pending.remove(key)
    write(dict(status="complete" if all(r["exit_code"]==0 for r in results.values()) else "partial_failure",
               results=results,started=started,completed=time.time()))

if __name__=="__main__":main()
