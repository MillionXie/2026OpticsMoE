"""One owner, finite serial continuation of the approved remaining21 lab points.

Preserve the already-complete shared Qwen100 triple. Each child retains the
published T06 admission, private MPS, source/asset guards and independent audit.
No automatic retries, new rental, new model, changed precision or warming.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
MODULE="LightGenV2.tasks.t06_video_quality_assessment.measure_mps_e2e_20261024"


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--deadline",type=float,required=True);args=p.parse_args()
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    root=args.output.resolve()
    if root.exists() or not root.is_relative_to(entry.CANONICAL):raise ValueError("New canonical continuation required")
    if not 0<args.deadline-time.time()<=14400:raise ValueError("Finite at-most-four-hour continuation")
    manifest=REPO/"experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv"
    references=[entry.CANONICAL/"qwen100_repaired01/qwen_cap100_repeat01",
                entry.CANONICAL/"qwen100_remaining01/qwen_cap100_repeat02",
                entry.CANONICAL/"qwen100_remaining01/qwen_cap100_repeat03"]
    # Independent CPU verification, not trusting names or mtime as provenance.
    verified=[entry.audit_point(path,manifest) for path in references]
    if any(row["status"]!="passed" for row in verified):raise ValueError("Shared Qwen100 incomplete")
    frozen=entry.source_map()
    root.mkdir(parents=True)
    state=dict(status="ready",pid=os.getpid(),deadline=args.deadline,completed=[],
               command=sys.argv,shared_qwen100=[dict(path=str(path.relative_to(REPO)),**v) for path,v in zip(references,verified)],
               source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
               protocol_id=entry.PROTOCOL,source_map=frozen)
    state["controller_sha256"] = entry.sha(__file__)
    def save():write(root/"queue_status.json",state)
    save()
    segments=[("qwen50",["qwen"],[50]),("glm_clip_yolo",["glm","clip","yolo"],[100,50])]
    try:
        for name,models,caps in segments:
            if time.time()>=args.deadline-120:raise TimeoutError("No time for next original suite")
            if entry.source_map()!=frozen:raise ValueError("Frozen T06 source changed")
            command=[sys.executable,"-u","-m",MODULE,"--phase","suite","--models",*models,
                     "--caps",*[str(c) for c in caps],"--repeats","1","2","3","--manifest",str(manifest),
                     "--output",str(root/name),"--deadline",str(args.deadline)]
            write(root/(name+".command.json"),command)
            with (root/(name+".log")).open("x") as log:
                child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=entry.config()["profiles"]["lab"]["gpu_uuid"]))
                state.update(status="running",phase=name,child_pid=child.pid);save()
                try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise
            if code:raise RuntimeError(name+" failed; preserve raw evidence and stop")
            result=entry.read(root/name/"suite_status.json")
            if result["status"]!="complete" or not result["mps_released"]:raise ValueError("Child did not complete/release")
            state["completed"].append(name);save()
        state.update(status="complete",finished_unix=time.time())
    except BaseException as error:
        state.update(status="stopped",error=repr(error));raise
    finally:save()


if __name__=="__main__":main()
