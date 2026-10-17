"""Independent stdlib-only audit of rental B2 and Qwen raw observations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__);result=[0.]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        rank=(start+1+end)/2
        for i in order[start:end]:result[i]=rank
        start=end
    return result


def correlation(a,b):
    am,bm=statistics.mean(a),statistics.mean(b)
    return math.fsum((x-am)*(y-bm) for x,y in zip(a,b))/math.sqrt(math.fsum((x-am)**2 for x in a)*math.fsum((y-bm)**2 for y in b))


def near(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f"Numerical mismatch {a} versus {b}")


def audit_inputs(root):
    original=json.loads((root/"original_cpu_input_identity.json").read_text())
    current=json.loads((root/"rental_cpu_input_identity.json").read_text())
    assert not original["cuda_initialized"] and not current["cuda_initialized"]
    assert original["prompt_sha256"]==current["prompt_sha256"]
    assert original["frame_positions"]==current["positions"]
    assert original["input_tensor_identity"]==current["inputs"]
    return dict(status="passed",scope="First fixed TEST video only; not a claim of hashing all558 inputs",prompt_sha256=current["prompt_sha256"],inputs=current["inputs"],original_receipt_sha256=sha(root/"original_cpu_input_identity.json"),rental_receipt_sha256=sha(root/"rental_cpu_input_identity.json"))


def audit_resident(folder,batch=2):
    report=json.loads((folder/"report.json").read_text());values=[];parts=[]
    assert report["status"]=="complete" and report["batch"]==batch
    for repeat in range(1,4):
        root=folder/f"repeat{repeat:02d}";r=json.loads((root/"report.json").read_text())
        raw=read_csv(root/"telemetry.csv");selected=[x for x in raw if r["start_monotonic_s"]<=float(x["host_monotonic_s"])<r["end_monotonic_s"]]
        v=[float(x["utilization_percent"]) for x in selected]
        assert len(v)==r["samples"] and r["duration_s"]>=30
        near(statistics.mean(v),r["mean_gpu_utilization_percent"])
        groups=json.loads((root/"forward_groups.json").read_text());measured=[g for g in groups if g["phase"]=="measured"];warm=[g for g in groups if g["phase"]=="warmup"]
        assert all(g["forwards"]==8 for g in groups) and warm[-1]["end"]-warm[0]["start"]>=5
        assert sum(g["forwards"] for g in measured)==r["measured_forwards"]
        values.extend(v);parts.append(dict(samples=len(v),utilization=statistics.mean(v),duration_s=r["duration_s"]))
    near(statistics.mean(values),report["pooled_mean_utilization_percent"])
    assert len(values)==report["total_samples"]
    parity=json.loads((folder/"parity.json").read_text()) if (folder/"parity.json").exists() else None
    if batch==2:assert parity["train_samples"]==2
    return dict(status="passed",utilization_percent=statistics.mean(values),samples=len(values),repeats=parts,checkpoint_sha256=report["identity"]["checkpoint_sha256"],gpu_uuid=report["identity"]["gpu_uuid"],parity=parity,source_report_sha256=sha(folder/"report.json"))


def audit_cost(folder,test,batch):
    report=json.loads((folder/"report.json").read_text());assert report["status"]=="complete"
    predictions=read_csv(folder/"predictions.csv");assert len(predictions)==558
    assert len({p["sample_id"] for p in predictions})==558
    assert [p["sample_id"] for p in predictions]==[r["sample_id"] for r in test]
    y=[float(r["temporal"]) for r in test];x=[float(p["prediction"]) for p in predictions]
    assert all(math.isfinite(v) for v in x)
    assert all(float(p["target_mos"])==v for p,v in zip(predictions,y))
    srcc=correlation(ranks(x),ranks(y));plcc=correlation(x,y)
    near(srcc,report["performance"]["srcc"]);near(plcc,report["performance"]["plcc"])
    rows=read_csv(folder/"batch_timing.csv");cost=[r for r in rows if r["in_cost_subset"]=="True"]
    assert len(rows)==558//batch and sum(int(r["videos"]) for r in rows)==558
    assert sum(int(r["videos"]) for r in cost)==200 and all(int(r["videos"])==batch for r in rows)
    t16=math.fsum(float(r["synchronized_wall_ms"]) for r in cost)*16/200
    raw=read_csv(folder/"telemetry.csv");active=[r for r in raw if r["phase"].startswith("active:")]
    assert active and all(int(r["phase"].split(":")[1])<200 for r in active)
    watts=statistics.mean(float(r["watts"]) for r in active);energy=(338.2+watts)*t16/1000
    near(t16,report["cost"]["latency_16_videos_ms"]);near(energy,report["cost"]["energy_16_videos_j"]);near(watts,report["power"]["active_mean_w"])
    return dict(status="passed",videos=558,cost_videos=200,batch=batch,srcc=srcc,plcc=plcc,T16_ms=t16,E16_j=energy,active_board_power_w=watts,active_samples=len(active),gpu_uuid=report["identity"]["gpu_uuid"],checkpoint_sha256=report["identity"]["checkpoint_sha256"],source_report_sha256=sha(folder/"report.json"),predictions_sha256=sha(folder/"predictions.csv"))


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--partial",action="store_true");args=p.parse_args()
    assert not args.output.exists()
    manifest=args.run/"provisioning/original_manifest.csv"
    assert sha(manifest)=="607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc"
    test=[r for r in read_csv(manifest) if r["split"]=="test"];results={}
    for model in ("deepseek","kimi"):
        for phase,func in (("utilization",audit_resident),("evaluate",lambda f:audit_cost(f,test,2))):
            folder=args.run/"batch02_followup_20261017"/model/phase
            if not (folder/"report.json").exists() and args.partial:continue
            results[model+"/"+phase]=func(folder)
    qwen=args.run/"qwen_device_control_20261017/native_batch01_cost"
    if (qwen/"report.json").exists():results["qwen/evaluate"]=audit_cost(qwen,test,1)
    elif not args.partial:raise RuntimeError("Qwen native B1 not complete")
    resident=args.run/"qwen_device_control_20261017/native_batch01_resident"
    if (resident/"report.json").exists():results["qwen/utilization"]=audit_resident(resident,1)
    input_root=args.run/"qwen_device_control_20261017"
    if (input_root/"rental_cpu_input_identity.json").exists():results["qwen/input_identity"]=audit_inputs(input_root)
    result=dict(status="passed",partial=args.partial,cuda_initialized=False,time=time.time(),entry_sha256=sha(Path(__file__)),results=results)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
