"""Finite SALICON original-checkpoint suite; same exclusive A100 lock and MPS100."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
TASK=REPO/"LightGenV2/tasks/t03_saliency"
GPU="GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55"
PIPE="/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024"
MODEL="/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--deadline",type=float,required=True)
    p.add_argument("--methods",nargs="+",choices=("ours","qwen"),default=["ours","qwen"])
    args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to((TASK/"runs/simulation").resolve()):raise ValueError("New task run required")
    if not 0<args.deadline-time.time()<=7200 or len(set(args.methods))!=len(args.methods):raise ValueError("Finite unique plan required")
    contract=TASK/"runs/smoke/a100_nn_contract_20261024_01"
    if "ours" in args.methods:
        if json.loads((contract/"cpu_contract.json").read_text())["status"]!="passed":raise ValueError("CPU contract absent")
        if json.loads((contract/"identity.json").read_text())["checkpoint_sha256"]!="036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe":
            raise ValueError("CPU contract uses another checkpoint")
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    engine.GPU,engine.PIPE,engine.ROOT=GPU,PIPE,root
    variants={"ours":("moe_alpha40_sam_batch8_crosssample_20260913.yaml","moe_alpha40_sam_batch8_crosssample_20260913_seed42"),
              "qwen":("qwen_aligned_head_50.yaml","qwen_aligned_head_50_20260912_seed42")}
    manifest=TASK/"runs/simulation/moe_alpha40_sam_batch8_crosssample_20260913_seed42/manifests/samples.csv"
    lockfile=REPO/"LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock"
    with lockfile.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.base.guard_records(GPU):raise RuntimeError("A100 is already occupied; do not interrupt")
        validate_idle_memory(GPU);root.mkdir(parents=True)
        state=dict(status="admission",pid=os.getpid(),completed=[],deadline=args.deadline,command=sys.argv,
                   source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip())
        active=False
        try:
            for _ in range(3):
                if engine.base.guard_records(GPU):raise RuntimeError("Foreign compute in admission")
                validate_idle_memory(GPU);time.sleep(10)
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_MPS_PIPE_DIRECTORY=PIPE,
                     CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="8")
            for method in args.methods:
                config,run=variants[method]
                for repeat in (1,2,3):
                    if time.time()>args.deadline-120:raise TimeoutError("No time for next worker")
                    name=f"{method}_repeat{repeat:02d}";folder=root/name
                    command=[sys.executable,"-u",str(HERE/"measure_salicon.py"),"--system",method,"--config",str(TASK/"configs"/config),
                        "--checkpoint",str(TASK/"runs/simulation"/run/"best_checkpoint.pt"),"--manifest",str(manifest),"--model",MODEL,
                        "--output",str(folder),"--repeat",str(repeat),"--gpu-uuid",GPU]
                    write(root/(name+".command.json"),command)
                    with (root/(name+".log")).open("x") as log:
                        child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status="running",phase=name,child_pid=child.pid);write(root/"suite_status.json",state)
                        try:code=child.wait(timeout=min(1800,max(1,args.deadline-time.time())))
                        except BaseException:
                            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
                    if code:raise RuntimeError(f"{name} exit{code}; stop, do not choose another trial")
                    report=json.loads((folder/"summary.json").read_text())
                    if report["status"]!="complete" or report["valid_samples"]!=200:raise ValueError("Incomplete200 run")
                    state["completed"].append(name);write(root/"suite_status.json",state)
            state.update(status="complete",finished_unix=time.time())
        except BaseException as error:
            state.update(status="stopped",error=repr(error));raise
        finally:
            if active:
                try:engine.daemon(False);state["mps_released"]=True
                except BaseException as error:state["release_error"]=repr(error)
            write(root/"suite_status.json",state)


if __name__=="__main__":main()
