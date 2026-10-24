"""Finite serial fresh-process200x3 runner. Only one audited A100 owner.

Only the explicitly implemented T07 checkpoint pair is enabled. No incomplete
task is filled with old timings. Source publication and CPU equivalence must
precede this command; data and outputs remain under the original task's runs.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
TASK=REPO/"LightGenV2/tasks/t07_abo_image_retrieval"
GPU="GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55"
PIPE="/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024"


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--methods",nargs="+",choices=("ours","qwen"),default=["ours","qwen"])
    p.add_argument("--deadline",type=float,required=True)
    args=p.parse_args()
    root=args.output.resolve()
    if root.exists() or not root.is_relative_to((TASK/"runs/simulation").resolve()):
        raise ValueError("New original-task simulation run path required; no overwrite")
    if len(set(args.methods))!=len(args.methods) or not 0<args.deadline-time.time()<=7200:
        raise ValueError("Bounded finite unique-method plan required")
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    engine.GPU,engine.PIPE,engine.ROOT=GPU,PIPE,root
    common=["--task","abo_image_to_image","--manifest",str(TASK/"runs/simulation/abo200_enrolled_protocol_20260913/protocol.json"),
            "--data",str(REPO/"data/abo_similarity10_data"),"--gpu-uuid",GPU]
    methods={
        "ours":[sys.executable,"-u",str(HERE/"measure_ours.py"),*common,"--checkpoint",
                str(TASK/"runs/simulation/physical_bounded30_spatial_consistent_20260927/best.pt"),
                "--assets",str(REPO/".codex_tmp/t07_robust_assets_20260926")],
        "qwen":[sys.executable,"-u",str(HERE/"measure_qwen.py"),*common,"--model",
                "/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
                "--reference",str(TASK/"runs/simulation/abo200_enrolled_qwen64_20260913/final_report.json")],
    }
    # Shared with the cooperating T06 chat. Nonblocking lock never waits for or
    # interrupts another run, whether it is another suite or another user.
    lock_path=REPO/"LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.base.guard_records(GPU):raise RuntimeError("Foreign/other suite GPU process; leave it untouched")
        validate_idle_memory(GPU)
        root.mkdir(parents=True)
        identity=dict(status="admission",pid=os.getpid(),source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                      command=sys.argv,gpu_uuid=GPU,mps_percent=100,completed=[],deadline=args.deadline,
                      protocol_id="lightgen_qwen_input_inclusive_200x3_20261024_v1",explicit_warmup_forwards=0)
        active=False
        try:
            for _ in range(3):
                if engine.base.guard_records(GPU):raise RuntimeError("GPU no longer idle")
                validate_idle_memory(GPU)
                time.sleep(10)
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_MPS_PIPE_DIRECTORY=PIPE,
                     CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="8")
            for method in args.methods:
                for repeat in (1,2,3):
                    if time.time()>args.deadline-120:raise TimeoutError("No budget for another fresh worker")
                    name=f"{method}_repeat{repeat:02d}"
                    command=[*methods[method],"--repeat",str(repeat),"--output",str(root/name)]
                    write(root/(name+".command.json"),command)
                    with (root/(name+".log")).open("x") as log:
                        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        identity.update(status="running",phase=name,child_pid=child.pid)
                        write(root/"suite_status.json",identity)
                        try:code=child.wait(timeout=min(1800,max(1,args.deadline-time.time())))
                        except BaseException:
                            os.killpg(child.pid,signal.SIGTERM)
                            child.wait(timeout=15)
                            raise
                    if code:raise RuntimeError(f"{name} failed exit{code}; preserve evidence, do not retry blindly")
                    summary=json.loads((root/name/("summary.json" if method=="ours" else "report.json")).read_text())
                    if summary["status"]!="complete" or summary.get("valid_samples",summary.get("measured_samples"))!=200:
                        raise ValueError("Incomplete200-call result")
                    identity["completed"].append(name)
                    write(root/"suite_status.json",identity)
            identity.update(status="complete",finished_unix=time.time())
        except BaseException as error:
            identity.update(status="stopped",error=repr(error))
            raise
        finally:
            if active:
                try:engine.daemon(False);identity["mps_released"]=True
                except BaseException as error:identity["release_error"]=repr(error)
            write(root/"suite_status.json",identity)


if __name__=="__main__":main()
