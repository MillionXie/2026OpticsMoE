"""Independent CPU replay of immutable raw timings, identities and phase energy."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[4]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf8"))
def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6):raise ValueError(f"Replay mismatch: {a} versus {b}")


def audit(root,manifest):
    state=read(root/"suite_status.json")
    if state["status"]!="complete" or not state["mps_released"] or state["completed"]!=[f"ours_repeat{i:02d}" for i in (1,2,3)]:
        raise ValueError("Three complete fresh Ours processes with released MPS required")
    expected=[r for r in read(manifest)["rows"] if r["split"]=="query"][:200]
    reports=[];details=[];predictions=None;source_ids=set();dependencies=set()
    for repeat in (1,2,3):
        directory=root/f"ours_repeat{repeat:02d}"
        info=read(directory/"identity.json");summary=read(directory/"summary.json")
        rows=csv_rows(directory/"samples.csv");stages=csv_rows(directory/"stages.csv")
        files=read(directory/"files_manifest.json")
        for f in files:
            path=directory/f["file"]
            if sha(path)!=f["sha256"] or path.stat().st_size!=f["bytes"]:raise ValueError("Original file SHA/size mismatch")
        if len(rows)!=200 or len({r["sample_id"] for r in rows})!=200:raise ValueError("200 distinct IDs required")
        if [r["sample_id"] for r in rows]!=[r["sample_id"] for r in expected]:raise ValueError("Wrong TEST order")
        if info["manifest_sha256"]!=sha(manifest):raise ValueError("Wrong sealed manifest")
        if info["checkpoint_sha256"]!="0b1ff5b682b833f8c4fd25e6330147e993e974272d7d2812fa679082ef0e87bc":
            raise ValueError("Not requested .8225 checkpoint")
        if info["explicit_warmup"]!=0 or info["mps_percent"]!=100:raise ValueError("Wrong measurement protocol")
        for row,expect in zip(rows,expected):
            if row["image_sha256"]!=expect["image_sha256"] or row["expected_product_id"]!=expect["product_id"]:
                raise ValueError("Raw sample identity differs")
            ranking=json.loads(row["ranking"])
            if len(ranking)!=1600 or set(ranking)!=set(range(1600)):raise ValueError("Incomplete ranking")
            if int(row["hit1"])!=int(row["expected_product_id"]==row["predicted_product_id"]):raise ValueError("Wrong Hit@1")
        signature=[(r["sample_id"],r["predicted_product_id"]) for r in rows]
        if predictions is not None and predictions!=signature:raise ValueError("Repeated predictions differ; retain and investigate")
        predictions=signature
        for record in read(directory/"stage_ledger.json"):
            marker="/2026OpticsMoE/"
            if marker not in record["source_file"].replace("\\","/"):
                # Site-packages files were SHA-bound before/after by the worker,
                # not by repository Git; never pretend a dependency lives in Git.
                dependencies.add((record["source_file"],record["source_sha256"]))
                continue
            name=record["source_file"].replace("\\","/").split(marker,1)[-1]
            blob=subprocess.check_output(["git","show",info["source_commit"]+":"+name],cwd=REPO)
            if hashlib.sha256(blob).hexdigest()!=record["source_sha256"]:raise ValueError("Source not from bound published Git")
            source_ids.add((name,record["source_sha256"]))
        events=read(directory/"process_audit.json")
        if len(events)<2:raise ValueError("Independent GPU occupation audit missing")
        for event in events:
            allowed=set(event["servers"])|set(event["clients"])
            if len(event["clients"])!=1:raise ValueError("Foreign or missing client")
            for proc in event["processes"]:
                if int(proc[0]) not in allowed or proc[1].strip()!=state["gpu_uuid"]:raise ValueError("Foreign GPU compute")
        charged=0.;totals=defaultdict(float);counts=Counter();by_sample=defaultdict(list)
        for stage in stages:
            start,end=float(stage["start_monotonic"]),float(stage["end_monotonic"])
            if end<start:raise ValueError("Negative stage interval")
            close((end-start)*1000,float(stage["synchronized_wall_ms"]))
            by_sample[int(stage["sample_index"])].append(stage)
            counts[stage["stage"]]+=1
            totals[stage["stage"]]+=float(stage["synchronized_wall_ms"])
            if stage["charged"]=="True":charged+=float(stage["synchronized_wall_ms"])
        for index in range(200):
            ordered=by_sample[index]
            for a,b in zip(ordered,ordered[1:]):
                if float(a["end_monotonic"])>float(b["start_monotonic"])+1e-8:raise ValueError("Nested/double counted stages")
        for mode in ("vision","language"):
            for name,n in (("router_fft",200),("expert_global_fft",400),("parallel_residual1",200),("parallel_residual2",200)):
                if counts[f"{mode}.{name}"]!=n:raise ValueError("Wrong six-pass/residual occurrence contract")
        if counts["bridge.native_merger"]!=200 or counts["slm.bounded_tanh_nonlinearity"]!=1200:
            raise ValueError("Bridge or input nonlinearity omitted")
        gap=0.
        for index,row in enumerate(rows):
            measured=0. if index==0 else (float(row["input_start"])-float(rows[index-1]["cpu_result_return"]))*1000
            close(measured,float(row["internal_gap_ms"]));gap+=measured
        charged+=gap
        optical=200*6*1.0447
        energy=121.746*optical/1000+104.230*charged/1000
        close(charged,summary["charged_electronic_ms_total"])
        close(energy,summary["phase_accounted_energy_j_total"])
        close((charged+optical)/200,summary["cost"]["sample_ms"])
        close(sum(int(r["hit1"]) for r in rows)/200,summary["measured_subset_hit1"])
        reports.append(dict(repeat=repeat,valid_queries=200,subset_hit1=summary["measured_subset_hit1"],
            electronic_ms=charged/200,optical_ms=optical/200,total_ms=(charged+optical)/200,
            phase_energy_j=energy/200,raw_summary_sha256=sha(directory/"summary.json"),
            wall_event_ratio=summary["paired_neural_timers"]["wall_event_ratio"]))
        for name in counts:
            first=next(r for r in stages if r["stage"]==name)
            details.append(dict(repeat=repeat,stage=name,family=first["family"],charged=first["charged"],
                                occurrences=counts[name],wall_ms_per_query=totals[name]/200))
    return dict(status="passed",task="03 ABO image-to-image",repeats=reports,
                calls=600,unique_test_queries=200,full800_reference_hit1=.8225,
                aggregate={k:sum(r[k] for r in reports)/3 for k in ("electronic_ms","optical_ms","total_ms","phase_energy_j","subset_hit1")},
                immutable_source_identities=sorted(source_ids),external_dependency_source_bindings=sorted(dependencies),stage_details=details,
                scope="input-inclusive phase-accounted hardware projection, not measured full-device E2E",
                no_op_or_tokens_rate_until_matching_counts=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    value=audit(a.root,a.manifest)
    with a.output.open("x",encoding="utf8") as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:value[k] for k in ("status","calls","unique_test_queries","aggregate")}))


if __name__=="__main__":main()
