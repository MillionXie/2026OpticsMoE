"""Independent CPU-only verification of a completed scoped rental model run."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()


def load(path):return json.loads(path.read_text())
def csvrows(path):
    with path.open(newline="") as stream:return list(csv.DictReader(stream))


def close(a,b,tol=1e-9):
    if not math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol):
        raise AssertionError(f"Independent recomputation mismatch: {a} vs {b}")


def audit(key,base):
    import numpy as np
    import torch
    from scipy.stats import spearmanr,pearsonr
    from .rental_vlm_temporal import rows,WIDTHS
    if torch.cuda.is_initialized():raise RuntimeError("Audit must not initialize CUDA")
    folder=base/key;train=folder/"train"
    receipt=load(train/"training_receipt.json")
    assert receipt["status"]=="complete"
    for name,sha in receipt["artifacts"].items():assert digest(train/name)==sha,name
    repo=Path(__file__).resolve().parents[3]
    for name,sha in receipt["identity"]["source_sha256"].items():
        blob=subprocess.check_output(["git","show",receipt["identity"]["git_commit"]+":"+name],cwd=repo)
        assert hashlib.sha256(blob).hexdigest()==sha,name
    data=rows();ids=[r["sample_id"] for r in data]
    test=[r for r in data if r["split"]=="test"]
    payload=torch.load(train/"features.pt",map_location="cpu",weights_only=False)
    assert payload["sample_ids"]==ids
    assert payload["splits"]==[r["split"] for r in data]
    x=payload["features"]
    assert tuple(x.shape)==(2808,WIDTHS[key]) and x.dtype==torch.float16 and bool(x.isfinite().all())
    assert torch.equal(payload["targets"],torch.tensor([r["temporal"] for r in data]))
    pieces=sorted((train/"feature_parts").glob("*.pt"));assert len(pieces)==44
    offset=0
    for piece in pieces:
        part=torch.load(piece,map_location="cpu",weights_only=False);n=len(part["sample_ids"])
        assert n==min(64,2808-offset)
        assert part["sample_ids"]==ids[offset:offset+n]
        assert part["identity"]==payload["identity"]
        assert torch.equal(part["features"],x[offset:offset+n]);offset+=n
    assert offset==2808
    history=load(train/"train_history.json")
    assert [h["epoch"] for h in history]==list(range(1,51))
    best=max(history,key=lambda h:h["test"]["srcc"])["epoch"]
    indices=[i for i,r in enumerate(data) if r["split"]=="test"]
    cpu={}
    for name,epoch in (("best",best),("last",50)):
        pt=torch.load(train/f"{name}_checkpoint.pt",map_location="cpu",weights_only=False)
        assert pt["selected_epoch"]==epoch and pt["trainable_parameters"]==5*WIDTHS[key]
        assert set(pt["state_dict"])=={"weight"} and tuple(pt["state_dict"]["weight"].shape)==(5,WIDTHS[key])
        predicted=(x[indices].float()@pt["state_dict"]["weight"].t()).softmax(-1)@pt["level_scores"]
        saved=csvrows(train/f"cpu_{name}_predictions.csv")
        assert len(saved)==558 and [r["sample_id"] for r in saved]==[r["sample_id"] for r in test]
        values=np.array([float(r["prediction_mos"]) for r in saved])
        assert np.allclose(predicted.numpy(),values,rtol=0,atol=2e-6)
        assert np.array_equal(np.array([float(r["target_mos"]) for r in saved]),payload["targets"][indices].numpy())
        cpu[name]=dict(epoch=epoch,srcc=float(spearmanr(payload["targets"][indices].numpy(),values).statistic),checkpoint_sha256=digest(train/f"{name}_checkpoint.pt"))
        close(cpu[name]["srcc"],load(train/"cpu_verification.json")[name]["strict_cpu"]["srcc"])
    frozen=load(train/"frozen_audit.json")
    assert frozen["status"]=="passed" and frozen["before_sha256"]==frozen["after_sha256"]
    assert frozen["all_parameters_require_grad_false"] and frozen["all_backbone_gradients_none"]
    native=folder/"a100_batch01_cost";report=load(native/"report.json")
    assert report["identity"]["checkpoint_sha256"]==cpu["best"]["checkpoint_sha256"]
    predictions=csvrows(native/"predictions.csv")
    assert len(predictions)==558 and [r["sample_id"] for r in predictions]==[r["sample_id"] for r in test]
    target=np.array([float(r["target_mos"]) for r in predictions]);prediction=np.array([float(r["prediction"]) for r in predictions])
    assert np.array_equal(target,np.array([r["temporal"] for r in test]))
    srcc=float(spearmanr(target,prediction).statistic);plcc=float(pearsonr(target,prediction).statistic)
    close(srcc,report["performance"]["srcc"]);close(plcc,report["performance"]["plcc"])
    timings=csvrows(native/"batch_timing.csv")
    assert len(timings)==558 and [int(r["batch_index"]) for r in timings]==list(range(558))
    assert all(int(r["videos"])==1 and (r["in_cost_subset"]=="True")== (i<200) for i,r in enumerate(timings))
    time16=sum(float(r["synchronized_wall_ms"]) for r in timings[:200])*16/200
    samples=csvrows(native/"telemetry.csv");active=[r for r in samples if r["phase"].startswith("active:")]
    assert active and all(0<=int(r["phase"].split(":")[1])<200 for r in active)
    power=sum(float(r["watts"]) for r in active)/len(active)
    close(time16,report["cost"]["latency_16_videos_ms"]);close(power,report["cost"]["measured_active_board_w"])
    energy=(338.2+power)*time16/1000;close(energy,report["cost"]["energy_16_videos_j"])
    resident=folder/"resident_batch01";rr=load(resident/"report.json")
    assert rr["identity"]["checkpoint_sha256"]==cpu["best"]["checkpoint_sha256"]
    pooled=[];repeat_reports=[]
    for repeat in range(1,4):
        directory=resident/f"repeat{repeat:02d}";r=load(directory/"report.json")
        samples=csvrows(directory/"telemetry.csv")
        selected=[s for s in samples if r["start_monotonic_s"]<=float(s["host_monotonic_s"])<r["end_monotonic_s"]]
        assert len(selected)==r["samples"] and r["duration_s"]>=30
        util=[float(s["utilization_percent"]) for s in selected]
        close(sum(util)/len(util),r["mean_gpu_utilization_percent"])
        assert r["sample_id"]==test[0]["sample_id"]
        assert abs(r["final_mos"][0]-prediction[0])<=2e-5
        groups=load(directory/"forward_groups.json")
        assert all(g["forwards"]==8 for g in groups)
        assert sum(g["forwards"] for g in groups if g["phase"]=="measured")==r["measured_forwards"]
        pooled+=util;repeat_reports.append(r)
    util=sum(pooled)/len(pooled);close(util,rr["pooled_mean_utilization_percent"])
    assert len(pooled)==rr["total_samples"]
    for phase in (train,native,resident):assert load(phase/"status.json")["status"]=="complete"
    return dict(status="passed",model=key,train_videos=2250,test_videos=558,features=2808,parts=44,epochs=50,
        cpu_cached_development=cpu,native_b1_srcc=srcc,native_b1_plcc=plcc,latency_16_videos_ms=time16,
        energy_16_videos_j=energy,active_board_power_w=power,host_proxy_w=338.2,
        resident_utilization_percent=util,resident_samples=len(pooled),resident_repeats=repeat_reports,
        hardware=report["identity"]["gpu_name"],power_limit_w=report["identity"]["power_limit_w"],
        test_selected_development_no_validation=True,cuda_initialized=torch.cuda.is_initialized(),checked=time.time())


def main():
    from .rental_vlm_temporal import RUN
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",choices=("deepseek","kimi"),required=True)
    args=p.parse_args();destination=RUN/args.model/"independent_cpu_audit.json"
    if destination.exists():raise RuntimeError("Preserve existing independent audit")
    result=audit(args.model,RUN)
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
