"""Independent stdlib-only audit of rental B2 and Qwen raw observations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_process_identity(folder):
    identity=json.loads((folder/"identity.json").read_text())
    rows=json.loads((folder/"process_audit.json").read_text())
    assert rows and [r["time"] for r in rows]==sorted(r["time"] for r in rows)
    observed=0
    for row in rows:
        assert len(row["processes"])<=1
        for process in row["processes"]:
            assert int(process[0])==identity["pid"] and process[1].strip()==identity["gpu_uuid"]
            observed+=1
    assert observed
    return dict(status="passed",pid=identity["pid"],observations=len(rows),
                own_process_observations=observed,foreign_observations=0,
                evidence_sha256=sha(folder/"process_audit.json"),
                limitation="Periodic visible compute-process checks; not an operating-system exclusive GPU lock")


def measured_source_sha(repo,commit,name):
    if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError("Repository-relative bound source required")
    # Hash the precise measured Git blob, not a Windows checkout with CRLF conversion.
    blob=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo,timeout=15)
    return hashlib.sha256(blob).hexdigest()


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
    plan=[(start,min(start+batch,boundary)) for begin,boundary in ((0,200),(200,558)) for start in range(begin,boundary,batch)]
    assert len(rows)==len(plan) and [int(r["videos"]) for r in rows]==[end-begin for begin,end in plan]
    assert sum(int(r["videos"]) for r in cost)==200
    t16=math.fsum(float(r["synchronized_wall_ms"]) for r in cost)*16/200
    raw=read_csv(folder/"telemetry.csv");active=[r for r in raw if r["phase"].startswith("active:")]
    assert active and all(int(r["phase"].split(":")[1])<200 for r in active)
    watts=statistics.mean(float(r["watts"]) for r in active);energy=(338.2+watts)*t16/1000
    near(t16,report["cost"]["latency_16_videos_ms"]);near(energy,report["cost"]["energy_16_videos_j"]);near(watts,report["power"]["active_mean_w"])
    return dict(status="passed",videos=558,cost_videos=200,batch=batch,srcc=srcc,plcc=plcc,T16_ms=t16,E16_j=energy,active_board_power_w=watts,active_samples=len(active),gpu_uuid=report["identity"]["gpu_uuid"],checkpoint_sha256=report["identity"]["checkpoint_sha256"],source_report_sha256=sha(folder/"report.json"),predictions_sha256=sha(folder/"predictions.csv"))


def audit_device_difference(run):
    new_root=run/"qwen_device_control_20261017/native_batch01_cost"
    old_native=run.parent/"lgvq_temporal_telemetry_a100_20261014/full_remeasurement02/qwen_batch01"
    anchor_path=run.parent/"lgvq_frozen_batch_costs_a100_20261015/qwen_batch01/report.json"
    previous=read_csv(old_native/"predictions.csv");current=read_csv(new_root/"predictions.csv")
    assert len(previous)==len(current)==558
    assert [r["sample_id"] for r in previous]==[r["sample_id"] for r in current]
    differences=[abs(float(a["prediction"])-float(b["prediction"])) for a,b in zip(previous,current)]
    anchor=json.loads(anchor_path.read_text());new=json.loads((new_root/"report.json").read_text())
    assert anchor["checkpoint_sha256"]==new["identity"]["checkpoint_sha256"]
    t0=anchor["cost"]["latency_16_videos_ms"];e0=anchor["cost"]["energy_16_videos_j"]
    t1=new["cost"]["latency_16_videos_ms"];e1=new["cost"]["energy_16_videos_j"]
    return dict(status="passed",prediction_videos=558,max_abs_mos_difference=max(differences),nonidentical_mos_count=sum(d!=0 for d in differences),historical_anchor_unchanged=True,old_anchor=dict(T16_ms=t0,E16_j=e0,report_sha256=sha(anchor_path)),new_measurement=dict(T16_ms=t1,E16_j=e1),observed_time_ratio=t0/t1,observed_energy_efficiency_ratio=e0/e1,time_reduction_percent=100*(1-t1/t0),estimated_energy_reduction_percent=100*(1-e1/e0),limitation="Different GPU/host/Python environments and dates; not a pure form-factor causal experiment. Old cost is protected historical reference; new cost independently measured.",old_native_prediction_sha256=sha(old_native/"predictions.csv"),new_native_prediction_sha256=sha(new_root/"predictions.csv"))


def audit_sequential_point(folder):
    report=json.loads((folder/"report.json").read_text())
    assert report["status"]=="complete" and len(report["repeats"])==3
    raw=read_csv(folder/"telemetry.csv")
    groups=json.loads((folder/"forward_groups.json").read_text())
    assert all(g["forwards"]==8 and g["start_monotonic_s"]<g["compute_end_monotonic_s"]<=g["end_monotonic_s"] for g in groups)
    paced=report["paced"]; values=[]; repeats=[]
    for index,r in enumerate(report["repeats"],1):
        start,end=r["start_monotonic_s"],r["end_monotonic_s"]
        near(end-start,r["duration_s"]); assert end-start>=30
        selected=[v for v in raw if start<=float(v["host_monotonic_s"])<end]
        util=[float(v["utilization_percent"]) for v in selected]
        assert len(util)>=(end-start)*15
        assert float(selected[0]["host_monotonic_s"])-start<=.25 and end-float(selected[-1]["host_monotonic_s"])<=.25
        watts=statistics.mean(float(v["watts"]) for v in selected)
        assert len(util)==r["samples"] and all(0<=v<=100 for v in util)
        near(statistics.mean(util),r["mean_gpu_utilization_percent"]); near(watts,r["mean_board_power_w"])
        measured=[g for g in groups if g["phase"]==f"measured{index}"]
        warm=[g for g in groups if g["phase"]==f"warmup{index}"]
        assert warm[-1]["end_monotonic_s"]-warm[0]["start_monotonic_s"]>=5
        assert sum(g["forwards"] for g in measured)==r["forward_calls"]
        assert len(r["output"])==report["batch"] and all(math.isfinite(v) for v in r["output"])
        if paced:
            near(r["frozen_duty"],report["frozen_duty"])
            assert all(g["duty"]==report["frozen_duty"] for g in warm+measured)
            cost=r["service_cost"]; t16=(end-start)*1000*16/(r["forward_calls"]*report["batch"])
            near(t16,cost["service_time_16_videos_ms"])
            near((338.2+watts)*t16/1000,cost["estimated_energy_16_videos_j"])
            assert cost["includes_all_imposed_wait"] and cost["not_dataset_once_latency"]
        else:
            assert all(g["requested_idle_s"]==0 and g["duty"]==1 for g in warm+measured)
        repeats.append(dict(utilization_percent=statistics.mean(util),board_w=watts,duration_s=end-start,
                            calls=r["forward_calls"],samples=len(util)))
        values.extend(util)
    pooled=statistics.mean(values); near(pooled,report["pooled_mean_utilization_percent"])
    assert len(values)==report["total_samples"]
    result=dict(status="passed",batch=report["batch"],paced=paced,utilization_percent=pooled,
                samples=len(values),repeats=repeats,report_sha256=sha(folder/"report.json"))
    if paced:
        calibration=[g for g in groups if g["phase"]=="calibration"]
        assert calibration[-1]["end_monotonic_s"]-calibration[0]["start_monotonic_s"]>=30
        seconds=sum(r["duration_s"] for r in repeats); calls=sum(r["calls"] for r in repeats)
        board=sum(r["board_w"]*r["duration_s"] for r in repeats)/seconds
        cost=report["pooled_service_cost"]; t16=seconds*16000/(calls*report["batch"])
        near(t16,cost["service_time_16_videos_ms"])
        near(board,cost["measured_same_window_board_w"])
        near((338.2+board)*t16/1000,cost["estimated_energy_16_videos_j"])
        assert report["target50_reached"]==(abs(pooled-50)<=5)
        result.update(service_T16_ms=t16,service_E16_j=(338.2+board)*t16/1000,board_w=board,
                      target50_reached=report["target50_reached"],not_dataset_once_cost=True)
    return result


def audit_sequential_session(run,session,test,partial=False):
    root=run/"sequential_busy50_20261018"/session
    identity=json.loads((root/"kimi/identity.json").read_text())
    repo=Path(__file__).resolve().parents[3]
    for name,digest in identity["sources"].items(): assert measured_source_sha(repo,identity['git_commit'],name)==digest, name
    assert sha(run/"kimi/train/best_checkpoint.pt")==identity["checkpoint_sha256"]
    results={}; native=[]
    process_audits={}
    if (root/"kimi/process_audit.json").exists():
        process_audits["worker"]=audit_process_identity(root/"kimi")
    for folder in sorted((root/"kimi").glob("*")):
        if not folder.is_dir() or not (folder/"report.json").exists(): continue
        if folder.name.startswith("batch") or folder.name.startswith("busy50_batch"):
            report=json.loads((folder/"report.json").read_text())
            inp=json.loads((folder/"input_identity.json").read_text())
            assert inp["sample_ids"]==[r["sample_id"] for r in test[:report["batch"]]]
            results[folder.name]=audit_sequential_point(folder)
            if not report["paced"]:
                native.append(report)
                assert report["pid"]==identity["pid"]
        elif folder.name.startswith("evaluate_batch"):
            batch=json.loads((folder/"report.json").read_text())["batch"]
            results[folder.name]=audit_cost(folder,test,batch)
            assert results[folder.name]["checkpoint_sha256"]==identity["checkpoint_sha256"]
            assert results[folder.name]["gpu_uuid"]==identity["gpu_uuid"]
            cost_identity=json.loads((folder/"identity.json").read_text())
            assert cost_identity["sources"]==identity["sources"]
            process_audits[folder.name]=audit_process_identity(folder)
    assert native and [r["batch"] for r in native]==list(range(1,len(native)+1))
    suite=json.loads((root/"suite_status.json").read_text())
    if not partial:
        assert suite["status"]=="complete"
        worker=json.loads((root/"kimi/report.json").read_text()); assert worker["status"]=="complete"
        assert len(worker["native_batches"])==len(native)
        assert "worker" in process_audits
        for batch in sorted({1,native[-1]["batch"]}):
            assert f"evaluate_batch{batch:02d}" in results and f"busy50_batch{batch:02d}" in results
    return dict(status="passed",partial=partial,suite_status=suite["status"],results=results,process_audits=process_audits,
        checkpoint_sha256=identity["checkpoint_sha256"],gpu_uuid=identity["gpu_uuid"],
        source_identity_verified=True,highest_native_batch=native[-1]["batch"],
        highest_native_utilization_percent=native[-1]["pooled_mean_utilization_percent"],
        native_above98=native[-1]["pooled_mean_utilization_percent"]>98)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--partial",action="store_true");p.add_argument("--session");args=p.parse_args()
    assert not args.output.exists()
    manifest=args.run/"provisioning/original_manifest.csv"
    assert sha(manifest)=="607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc"
    test=[r for r in read_csv(manifest) if r["split"]=="test"];results={}
    if args.session:
        if args.session not in ("kimi_clean_morning01_20261018","kimi_context_limited01"): raise ValueError("Canonical session only")
        result=audit_sequential_session(args.run,args.session,test,args.partial)
        result.update(cuda_initialized=False,time=time.time(),entry_sha256=sha(Path(__file__)))
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2));return
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
    elif not args.partial:raise RuntimeError("Qwen resident utilization not complete")
    input_root=args.run/"qwen_device_control_20261017"
    if (input_root/"rental_cpu_input_identity.json").exists():results["qwen/input_identity"]=audit_inputs(input_root)
    if (qwen/"report.json").exists():results["qwen/device_difference"]=audit_device_difference(args.run)
    result=dict(status="passed",partial=args.partial,cuda_initialized=False,time=time.time(),entry_sha256=sha(Path(__file__)),results=results)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
